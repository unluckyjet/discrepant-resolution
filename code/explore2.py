import numpy as np, pandas as pd, sys, itertools
sys.path.insert(0, "code")
from drcore import *
df = pd.read_parquet("data/derived/redux_helm.parquet")
df = df[df.ref_status.isin(["ok", "wrong_gt"])].reset_index(drop=True)
models = [c for c in df.columns if "_" in c and c not in ("ref_status", "error_type")]
Y = df.Y.values
P = {m: df[m].values for m in models}

def inversions(L, F, evaluated):
    disc = P[F] != L
    D = dr_labels(L, Y, disc)
    t = {m: acc(P[m], Y) for m in evaluated}; d = {m: acc(P[m], D) for m in evaluated}
    inv = [(a, b) for a, b in itertools.combinations(evaluated, 2) if (t[a] - t[b]) * (d[a] - d[b]) < 0]
    fav = [(a, b) for a, b in inv if F in (a, b)]
    return len(inv), len(fav), disc.mean(), ((~disc) & (L != Y)).mean(), (L != Y).mean()

print("Scenario A: original MMLU labels")
L = df.L.values
for F in models:
    n_inv, n_fav, dm, pj, pe = inversions(L, F, models)
    if n_inv: print(f"  F={F[:30]:30s} inversions={n_inv} involving F={n_fav} P(J)={pj:.4f}")

print("Scenario B: labels produced by an LLM labeler (reference = Redux)")
for lab in ["mistralai_mixtral-8x7b-32kseqlen", "meta_llama-2-70b", "openai_gpt-3.5-turbo-0613", "qwen_qwen1.5-72b"]:
    L = np.where(P[lab] >= 0, P[lab], (Y + 1) % 4)  # invalid outputs become a wrong label
    ev = [m for m in models if m != lab]
    rows = []
    for F in ev:
        n_inv, n_fav, dm, pj, pe = inversions(L, F, ev)
        b, _ = bias_terms(P[F], L, Y, P[F] != L)
        rows.append((F[:26], round(acc(P[F], Y), 3), round(dm, 3), round(pj, 3), round(b, 3), n_inv, n_fav))
    r = pd.DataFrame(rows, columns=["F", "accF", "P(disc)", "P(J)", "biasF", "inv", "invF"]).sort_values("accF", ascending=False)
    print(f" labeler={lab} label error rate={(L != Y).mean():.3f}")
    print(r.head(8).to_string(index=False))
