"""CoNLL-2003 re-analysis from the released, text-free derived files (see build_conll_items.py).

Reiss et al. (2020) audit vs the independent full re-check CoNLL++ (Wang et al. 2019). Item = token;
label = entity type. Writes results/conll.json.
"""
import json, sys
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).parents[1]))
from drcore import sensitivity_bounds, sensitivity_bounds_diff

TYPES = {"O": 0, "PER": 1, "LOC": 2, "ORG": 3, "MISC": 4}
it = pd.read_csv("data/derived/conll_items.csv", keep_default_na=False)
pr = pd.read_csv("data/derived/conll_preds.csv", keep_default_na=False)
meta = json.loads(Path("data/derived/conll_build_meta.json").read_text())
L, A, Y = (it[c].map(TYPES).values for c in ("L", "A", "Y"))
Delta, Flag = it.adjudicated.values.astype(bool), it.flagged.values.astype(bool)
models = {m: pr[m].map(TYPES).values for m in pr.columns}
items = list(it.line)
excluded = range(meta["n_excluded"])
n_rows = meta["n_rows"]
changed_outside = meta["changed_outside_delta"]
is_tok = []
n = len(items)
D = np.where(Delta, A, L)                         # the published DR labels
J = (~Delta) & (L != Y)                           # hidden errors w.r.t. the reference
res = dict(n_items=n, n_excluded=meta["n_excluded"],
           n_rows=n_rows, n_adjudicated=int(Delta.sum()), n_flagged=int(Flag.sum()),
           n_incidental_only=int((Delta & ~Flag).sum()), changed_outside_delta=changed_outside,
           ref_errors=int((L != Y).sum()), ref_errors_in_delta=int(((L != Y) & Delta).sum()),
           hidden=int(J.sum()), reiss_changes=int((A != L).sum()),
           adjudicator_disagree_in_delta=int(((A != Y) & Delta).sum()),
           agree_both_change=int(((A != L) & (Y != L) & (A == Y)).sum()))
per = {}
for m, M in models.items():
    true, dr, naive = np.mean(M == Y), np.mean(M == D), np.mean(M == L)
    hidden_part = np.mean(J & (M == L)) - np.mean(J & (M == Y))
    adj_part = np.mean(Delta & (M == A)) - np.mean(Delta & (M == Y))
    assert abs((dr - true) - (hidden_part + adj_part)) < 1e-12        # Prop. 1, imperfect adjudicator
    per[m] = dict(true=true, dr=dr, naive=naive, bias=dr - true, hidden_part=hidden_part, adj_part=adj_part,
                  hidden_correct=int((J & (M == Y)).sum()), hidden_repeat=int((J & (M == L)).sum()))
res["models"] = per
names = list(models)
t = [per[m]["true"] for m in names]; d = [per[m]["dr"] for m in names]
res["kendall_tau"] = float(kendalltau(t, d).statistic)          # scipy default: tau-b (handles ties)
res["inversions"] = [(a, b) for a, b in combinations(names, 2)
                     if (per[a]["true"] - per[b]["true"]) * (per[a]["dr"] - per[b]["dr"]) < 0]
res["rank_true"] = {m: int(1 + sum(per[x]["true"] > per[m]["true"] for x in names)) for m in names}
res["rank_dr"] = {m: int(1 + sum(per[x]["dr"] > per[m]["dr"] for x in names)) for m in names}

# Theorem 6 sensitivity bounds from DR data alone (reference labels used only to check coverage)
C = ~Delta
true_rate = float(J.sum() / C.sum())
res["true_concordant_error_rate"] = true_rate
grid = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]
Ad = np.where(Delta, A, -9)
res["bounds"] = {m: [(e, *sensitivity_bounds(M, L, Delta, Ad, e * C.mean())) for e in grid] for m, M in models.items()}
order = sorted(names, key=lambda m: -per[m]["dr"])
pairs = list(zip(order[:-1], order[1:]))[:6]
res["pair_bounds"] = {f"{a}|{b}": dict(dr=per[a]["dr"] - per[b]["dr"], true=per[a]["true"] - per[b]["true"],
                                       bounds=[(e, *sensitivity_bounds_diff(models[a], models[b], L, Delta, Ad, e * C.mean(), n_classes=5)) for e in grid])
                      for a, b in pairs}
# Robustness: restrict the items to entity tokens (original or reference label is an entity). The subset is
# fixed before looking at any model and contains every reference error (L != Y implies one of them is non-O).
E = (L != 0) | (Y != 0)
sub = {}
for m, M in models.items():
    true, dr = np.mean(M[E] == Y[E]), np.mean(M[E] == D[E])
    hp = np.mean((J & E)[E] & (M[E] == L[E])) - np.mean((J & E)[E] & (M[E] == Y[E]))
    ap = np.mean(Delta[E] & (M[E] == A[E])) - np.mean(Delta[E] & (M[E] == Y[E]))
    assert abs((dr - true) - (hp + ap)) < 1e-12
    sub[m] = dict(true=true, dr=dr, bias=dr - true, hidden_part=hp, adj_part=ap)
ts = [sub[m]["true"] for m in names]; ds = [sub[m]["dr"] for m in names]
res["entity_subset"] = dict(n_items=int(E.sum()), hidden=int((J & E).sum()), models=sub,
                            kendall_tau_b=float(kendalltau(ts, ds).statistic),
                            inversions=[(a, b) for a, b in combinations(names, 2)
                                        if (sub[a]["true"] - sub[b]["true"]) * (sub[a]["dr"] - sub[b]["dr"]) < 0],
                            rank_true={m: int(1 + sum(sub[x]["true"] > sub[m]["true"] for x in names)) for m in names},
                            rank_dr={m: int(1 + sum(sub[x]["dr"] > sub[m]["dr"] for x in names)) for m in names})
res["dr_rank_ties"] = int(len(d) - len(set(np.round(d, 12))))

Path("results").mkdir(exist_ok=True)
Path("results/conll.json").write_text(json.dumps(res, indent=1, default=lambda o: o.item() if isinstance(o, np.generic) else o))
print(json.dumps({k: v for k, v in res.items() if k not in ("models", "bounds", "pair_bounds", "rank_true", "rank_dr")}, indent=1, default=str))
for m in sorted(names, key=lambda m: -per[m]["true"]):
    p = per[m]
    print(f"{m:34s} true={100*p['true']:.3f} DR={100*p['dr']:.3f} naive={100*p['naive']:.3f} bias={100*p['bias']:+.3f} "
          f"(hidden {100*p['hidden_part']:+.3f}, adj {100*p['adj_part']:+.3f}) rank {res['rank_true'][m]}->{res['rank_dr'][m]}")
