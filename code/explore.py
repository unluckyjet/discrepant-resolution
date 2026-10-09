import numpy as np, pandas as pd, sys
sys.path.insert(0, "code")
from drcore import *
df = pd.read_parquet("data/derived/redux_helm.parquet")
df = df[df.ref_status.isin(["ok", "wrong_gt"])].reset_index(drop=True)
models = [c for c in df.columns if "_" in c and c not in ("ref_status", "error_type")]
L, Y = df.L.values, df.Y.values
print("n", len(df), "label errors", (L != Y).sum(), (L != Y).mean())
true = {m: acc(df[m].values, Y) for m in models}
naive = {m: acc(df[m].values, L) for m in models}
order = sorted(models, key=lambda m: -true[m])
print(pd.DataFrame({"true": true, "naive": naive}).loc[order].round(4).head(12))
for F in order[:6]:
    disc = df[F].values != L
    J = (~disc) & (L != Y)
    D = dr_labels(L, Y, disc)
    dr = {m: acc(df[m].values, D) for m in models}
    caught = (disc & (L != Y)).sum()
    tr = pd.Series(true).rank(ascending=False); drr = pd.Series(dr).rank(ascending=False)
    print(f"F={F[:28]:28s} disc={disc.sum():4d} errors caught={caught} hidden J={J.sum()} "
          f"F true={true[F]:.4f} F DR={dr[F]:.4f} rank true={tr[F]:.0f} DR={drr[F]:.1f}")
