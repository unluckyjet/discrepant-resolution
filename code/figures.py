"""Paper figures from results/*.json (vector PDF, Okabe-Ito palette, no in-figure titles)."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from names import short

R = Path("results"); FIG = Path("paper/figures"); FIG.mkdir(parents=True, exist_ok=True)
OI = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "axes.labelsize": 9, "legend.fontsize": 8, "legend.frameon": False})
load = lambda n: json.loads((R / f"{n}.json").read_text())
SCEN_B = "B:mistralai_mixtral-8x7b-32kseqlen"


def fig_slope():
    """(a) True vs DR accuracy, scenario B, flagger Llama 3 70B. (b) flagger rank gain vs true rank, all flaggers."""
    import pandas as pd
    sys.path.insert(0, "code")
    from drcore import acc, dr_labels
    df = pd.read_parquet("data/derived/redux_helm.parquet")
    df = df[df.ref_status.isin(["ok", "wrong_gt"])].reset_index(drop=True)
    Y = df.Y.values; lab = "mistralai_mixtral-8x7b-32kseqlen"
    L = np.where(df[lab].values >= 0, df[lab].values, (Y + 1) % 4)
    F = "meta_llama-3-70b"
    D = dr_labels(L, Y, df[F].values != L)
    models = [c for c in df.columns if c not in ("qid", "subject", "L", "Y", "error_type", "ref_status", lab)]
    t = {m: acc(df[m].values, Y) for m in models}; d = {m: acc(df[m].values, D) for m in models}
    top = sorted(models, key=lambda m: -t[m])[:10]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    for m in top:
        isF = m == F
        ax.plot([0, 1], [t[m], d[m]], color=OI[6] if isF else "0.6", lw=2.0 if isF else 0.9, zorder=3 if isF else 2)
        ax.scatter([0, 1], [t[m], d[m]], color=OI[6] if isF else "0.45", s=10, zorder=4)
    # label left (true) side with rank-ordered names, offset to avoid overlap
    ys = np.array([t[m] for m in top]); ylab = ys.copy()
    for _ in range(200):   # simple 1-d label repulsion around the true positions
        for i in range(1, len(ylab)):
            gap = ylab[i - 1] - ylab[i]
            if gap < 0.0042:
                shift = (0.0042 - gap) / 2
                ylab[i - 1] += shift; ylab[i] -= shift
    for m, y0, y1 in zip(top, ys, ylab):
        ax.text(-0.05, y1, short(m), ha="right", va="center", fontsize=7,
                color=OI[6] if m == F else "0.2", fontweight="bold" if m == F else None)
    ax.text(1.05, d[F], f"{short(F)}\n(flagger)", ha="left", va="center", fontsize=7, color=OI[6], fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["true accuracy\n(MMLU-Redux)", "DR-estimated\naccuracy"])
    ax.set_xlim(-0.9, 1.6); ax.set_ylabel("accuracy")
    ax.text(-0.9, ax.get_ylim()[1], "(a)", fontweight="bold", va="top")
    # (b)
    ax = axes[1]
    b = load("bias")[SCEN_B]["flaggers"]
    tr = np.array([v["flagger_true_rank"] for v in b.values()]); gain = np.array([v["flagger_true_rank"] - v["flagger_dr_rank"] for v in b.values()])
    ax.scatter(tr, gain, s=14, color=OI[5])
    ax.axhline(0, color="0.5", lw=0.7)
    ax.set_xlabel("flagger's true rank (36 models)"); ax.set_ylabel("ranks gained by flagger under DR")
    ax.text(ax.get_xlim()[0], ax.get_ylim()[1], "(b)", fontweight="bold", va="top")
    fig.tight_layout(); fig.savefig(FIG / "fig_slope.pdf"); plt.close(fig)


def fig_lockin():
    l = load("lockin")[SCEN_B]
    ks = [r["panel_size"] for r in l["rows"]]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    ax = axes[0]
    ax.plot(ks, [r["n_hidden"] for r in l["rows"]], marker="o", color=OI[0], label="hidden errors $|J|$")
    ax.plot(ks, [r["n_adjudicated"] for r in l["rows"]], marker="s", color=OI[2], label="items adjudicated")
    ax.set_xlabel("panel size (2023 models, union rule)"); ax.set_ylabel("items"); ax.legend()
    ax.text(0.02, 0.98, "(a)", transform=ax.transAxes, fontweight="bold", va="top")
    ax = axes[1]
    for j, m in enumerate(l["new"]):
        ax.plot(ks, [100 * r["new_bias"][m] for r in l["rows"]], marker="os^Dv<>P"[j], ms=3.5, lw=1,
                color=(OI[1:] + ["0.45"])[j], label=short(m))
    ax.axhline(0, color="0.5", lw=0.7)
    ax.set_xlabel("panel size (2023 models, union rule)"); ax.set_ylabel("DR bias of 2024 model (pp)")
    ax.legend(ncol=2, fontsize=6.5, loc="lower right")
    ax.text(0.02, 0.98, "(b)", transform=ax.transAxes, fontweight="bold", va="top")
    fig.tight_layout(); fig.savefig(FIG / "fig_lockin.pdf"); plt.close(fig)


def fig_two_phase():
    t = load("two_phase")
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
    for j, (scen, name) in enumerate([("A", "A: MMLU labels"), (SCEN_B, "B: LLM labels")]):
        rows = t[scen]["rows"]
        pc = [r["pi_C"] for r in rows if r["pi_C"] > 0]
        dr = [r for r in rows if r["pi_C"] == 0][0]
        two = [r for r in rows if r["pi_C"] > 0]
        axes[0].plot(pc, [100 * r["rmse_mean"] for r in two], marker="o", color=OI[5 + j], label=name)
        axes[0].axhline(100 * dr["rmse_mean"], color=OI[5 + j], ls="--", lw=0.8)
        axes[1].plot(pc, [r["coverage_wald_mean"] for r in two], marker="o", color=OI[5 + j], label=name + " (Wald)")
        axes[1].plot(pc, [r["coverage_cp_mean"] for r in two], marker="^", ls=":", color=OI[5 + j], label=name + " (conservative)")
        axes[2].plot(pc, [100 * r["width_wald_mean"] for r in two], marker="o", color=OI[5 + j])
        axes[2].plot(pc, [100 * r["width_cp_mean"] for r in two], marker="^", ls=":", color=OI[5 + j])
    for ax in axes:
        ax.set_xscale("log"); ax.set_xlabel(r"concordant sampling rate $\pi_C$")
    axes[0].set_ylabel("RMSE, mean over models (pp)"); axes[0].legend(fontsize=7)
    axes[1].axhline(0.95, color="0.5", lw=0.7); axes[1].set_ylabel("coverage of 95% interval"); axes[1].legend(fontsize=6)
    axes[2].set_ylabel("mean interval width (pp)"); axes[2].set_yscale("log")
    for k, ax in enumerate(axes):
        ax.text(0.02, 0.98, "(" + "abc"[k] + ")", transform=ax.transAxes, fontweight="bold", va="top")
    fig.tight_layout(); fig.savefig(FIG / "fig_two_phase.pdf"); plt.close(fig)


def fig_rate():
    r = load("rate")
    m = np.array([x["m"] for x in r["rows"]])
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.loglog(m, [x["rmse"] for x in r["rows"]], "o", color=OI[5], label="empirical RMSE")
    ax.loglog(m, [x["theory_se"] for x in r["rows"]], "-", color=OI[0], lw=1, label=r"$\gamma\sqrt{\varepsilon(1-\pi_C)/m}$ (design SE)")
    ax.loglog(m, [x["lower_bound"] for x in r["rows"]], "--", color=OI[6], lw=1, label="minimax lower bound")
    ax.set_xlabel("expected concordant adjudications $m$"); ax.set_ylabel("error in flagger accuracy")
    ax.legend(fontsize=6.5)
    fig.tight_layout(); fig.savefig(FIG / "fig_rate.pdf"); plt.close(fig)


def fig_sensitivity():
    """Identified set for theta_k - theta_F (k = model ranked just above the flagger) vs the assumed bound eps."""
    s = load("sensitivity")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for k, scen in enumerate(["A", SCEN_B]):
        v = s[scen]; ax = axes[k]
        eps = np.array([r["eps"] for r in v["rows"]])
        a, b = v["rows"][0]["pair"]
        lo = np.array([100 * r["pair_bounds"][0] for r in v["rows"]]); hi = np.array([100 * r["pair_bounds"][1] for r in v["rows"]])
        ax.fill_between(eps, lo, hi, color=OI[5], alpha=0.25, lw=0, label="identified set")
        ax.plot(eps, lo, color=OI[5], lw=1); ax.plot(eps, hi, color=OI[5], lw=1)
        ax.axhline(100 * v["pair_true"], color=OI[6], ls="--", lw=1, label="true difference")
        ax.axhline(0, color="0.4", lw=0.7)
        ax.axvline(v["true_hidden_rate"], color="0.4", ls=":", lw=0.8, label="true concordant error rate")
        ax.set_xlabel(r"assumed bound $\varepsilon$ on concordant error rate")
        ax.set_ylabel(f"{short(a)} $-$ {short(b)} (pp)")
        ax.text(0.02, 0.98, "(" + "ab"[k] + ") " + ("A: MMLU labels" if k == 0 else "B: LLM labels") + f"; flagger {short(b)}",
                transform=ax.transAxes, fontweight="bold", va="top", fontsize=7.5)
        if k == 1: ax.legend(fontsize=6.5, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "fig_sensitivity.pdf"); plt.close(fig)


def fig_sweep():
    """Labeler-strength sweep: every model in turn supplies the labels."""
    s = load("sweep")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for k, v in s.items():
        isA = k == "A"
        x = 100 * v["label_error_rate"]
        c = OI[6] if isA else OI[5]
        axes[0].plot([x, x], [100 * v["median_inflation"], 100 * v["max_inflation"]], color=c, lw=0.8, alpha=0.6)
        axes[0].scatter([x], [100 * v["median_inflation"]], color=c, s=18 if isA else 12, zorder=3,
                        marker="D" if isA else "o")
        axes[1].scatter([x], [v["max_rank_gain"]], color=c, s=18 if isA else 12, marker="D" if isA else "o")
    axes[0].annotate("original MMLU labels", (100 * s["A"]["label_error_rate"], 100 * s["A"]["median_inflation"]),
                     xytext=(8, 40), textcoords="offset points", fontsize=7, color=OI[6],
                     arrowprops=dict(arrowstyle="-", color=OI[6], lw=0.6))
    axes[0].set_xlabel("label error rate of the labeler (%)"); axes[0].set_ylabel("flagger inflation (pp)\nmedian (dot) to max (bar)")
    axes[1].set_xlabel("label error rate of the labeler (%)"); axes[1].set_ylabel("max ranks gained by a flagger")
    for j, ax in enumerate(axes):
        ax.text(0.02, 0.98, "(" + "ab"[j] + ")", transform=ax.transAxes, fontweight="bold", va="top")
    fig.tight_layout(); fig.savefig(FIG / "fig_sweep.pdf"); plt.close(fig)


if __name__ == "__main__":
    for f in (fig_slope, fig_lockin, fig_two_phase, fig_rate, fig_sensitivity, fig_sweep):
        f(); print("ok", f.__name__)
