"""All experiments for the paper. Writes results/*.json and results/*.csv; deterministic seeds.

Scenario A: original MMLU labels L, reference Y from MMLU-Redux (random human re-annotation).
Scenario B: an LLM-labelled benchmark: L = one HELM model's answers, reference Y from MMLU-Redux.
"""
import itertools, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, beta

sys.path.insert(0, str(Path(__file__).parent))
from drcore import acc, dr_labels, bias_terms, sensitivity_bounds, sensitivity_bounds_diff, hypergeom_upper, simultaneous_bounds

OUT = Path("results"); OUT.mkdir(exist_ok=True)
REPS = 2000
ALPHA = 0.05

df = pd.read_parquet("data/derived/redux_helm.parquet")
df = df[df.ref_status.isin(["ok", "wrong_gt"])].reset_index(drop=True)
MODELS = [c for c in df.columns if c not in ("qid", "subject", "L", "Y", "error_type", "ref_status")]
P = {m: df[m].values.astype(int) for m in MODELS}
Y = df.Y.values.astype(int)
N = len(df)
RES = {"n_items": N, "n_models": len(MODELS), "n_label_errors_A": int((df.L.values != Y).sum()),
       "n_invalid": {m: int((P[m] < 0).sum()) for m in MODELS}}

LABELERS_B = ["mistralai_mixtral-8x7b-32kseqlen", "meta_llama-2-70b", "openai_gpt-3.5-turbo-0613", "qwen_qwen1.5-72b"]
OLD = ["openai_gpt-3.5-turbo-0613", "meta_llama-2-70b", "anthropic_claude-2.1", "google_text-bison@001",
       "anthropic_claude-instant-1.2", "mistralai_mixtral-8x7b-32kseqlen"]          # released 2023
NEW = ["openai_gpt-4o-2024-05-13", "anthropic_claude-3-opus-20240229", "google_gemini-1.5-pro-preview-0409",
       "meta_llama-3-70b", "mistralai_mixtral-8x22b", "qwen_qwen1.5-72b",
       "google_gemini-1.5-flash-preview-0514", "anthropic_claude-3-sonnet-20240229"]  # released 2024


def labels_for(scenario):
    if scenario == "A":
        return df.L.values.astype(int), MODELS
    lab = scenario.split(":")[1]
    L = np.where(P[lab] >= 0, P[lab], (Y + 1) % 4)  # an invalid labeler output becomes a wrong label
    return L, [m for m in MODELS if m != lab]


def dr_summary(L, evaluated, disc, flaggers):
    D = dr_labels(L, Y, disc)
    J = (~disc) & (L != Y)
    t = {m: acc(P[m], Y) for m in evaluated}
    d = {m: acc(P[m], D) for m in evaluated}
    nv = {m: acc(P[m], L) for m in evaluated}
    pairs = list(itertools.combinations(evaluated, 2))
    inv = [(a, b) for a, b in pairs if (t[a] - t[b]) * (d[a] - d[b]) < 0]
    fav = [(a, b) for a, b in inv
           if (a in flaggers) != (b in flaggers) and ((d[a] > d[b]) == (a in flaggers))]
    tau = kendalltau([t[m] for m in evaluated], [d[m] for m in evaluated]).statistic
    bias = {m: d[m] - t[m] for m in evaluated}
    # exact check of Prop. 1 on every model
    for m in evaluated:
        b, _ = bias_terms(P[m], L, Y, disc)
        assert abs(b - bias[m]) < 1e-12
    return dict(n_adjudicated=int(disc.sum()), n_errors=int((L != Y).sum()),
                n_caught=int((disc & (L != Y)).sum()), n_hidden=int(J.sum()), pJ=float(J.mean()),
                n_inversions=len(inv), n_flagger_favoring=len(fav), kendall_tau=float(tau),
                bias=bias, true=t, dr=d, naive=nv, inversions=inv)


# ---------------------------------------------------------------- Exp 1/2: DR bias and rankings
def exp_bias():
    out = {}
    for scen in ["A"] + [f"B:{l}" for l in LABELERS_B]:
        L, ev = labels_for(scen)
        order = sorted(ev, key=lambda m: -acc(P[m], Y))
        rows = {}
        for F in ev:
            s = dr_summary(L, ev, P[F] != L, {F})
            rows[F] = {k: v for k, v in s.items() if k not in ("bias", "true", "dr", "naive", "inversions")}
            rows[F]["flagger_bias"] = s["bias"][F]
            # ranks: 1 + number of models strictly ahead, under both true and DR accuracy
            rows[F]["flagger_true_rank"] = 1 + sum(acc(P[m], Y) > acc(P[F], Y) for m in ev)
            rows[F]["flagger_dr_rank"] = 1 + sum(s["dr"][m] > s["dr"][F] for m in ev)
            rows[F]["max_nonflagger_bias"] = max(v for m, v in s["bias"].items() if m != F)
        panels = {}
        for k in (3, 5, 10, len(ev)):
            pan = order[:k]
            disc = np.any(np.stack([P[m] for m in pan]) != L, axis=0)
            s = dr_summary(L, ev, disc, set(pan))
            panels[f"top{k}"] = {kk: v for kk, v in s.items() if kk not in ("bias", "true", "dr", "naive", "inversions")}
            panels[f"top{k}"]["max_panel_bias"] = max(s["bias"][m] for m in pan)
        naive = {m: acc(P[m], L) - acc(P[m], Y) for m in ev}
        # independence benchmark for P(J): F agrees with a wrong L by choosing the same wrong option
        indep = {F: float(np.mean(L != Y) * np.mean((P[F] != Y)) / 3) for F in ev}
        out[scen] = dict(label_error_rate=float(np.mean(L != Y)), flaggers=rows, panels=panels,
                         naive_bias=naive, pJ_independence=indep,
                         true_acc={m: acc(P[m], Y) for m in ev})
    return out


# ---------------------------------------------------------------- Exp 3: two-phase estimation
def cp_interval(k, n, a):
    lo = beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


def two_phase_sim(L, F, ev, pi_C, pi_D=1.0, reps=REPS, seed=0, target_pairs=()):
    """Monte Carlo over the phase-2 adjudication draw (Poisson sampling). Items are fixed (finite population)."""
    rng = np.random.default_rng(seed)
    disc = P[F] != L
    pi = np.where(disc, pi_D, pi_C)
    Mstack = np.stack([P[m] for m in ev])                      # K x N
    r = (Mstack == Y).astype(float) - (Mstack == L)            # K x N, nonzero only where L != Y
    base = (Mstack == L).mean(axis=1)
    truth = (Mstack == Y).mean(axis=1)
    C = ~disc
    NC = C.sum()
    est = np.empty((reps, len(ev))); se = np.empty_like(est); cov_w = np.empty_like(est, dtype=bool)
    cov_cp = np.empty_like(cov_w); n_adj = np.empty(reps)
    width_w = np.empty_like(est); width_cp = np.empty_like(est)
    pair_idx = [(ev.index(a), ev.index(b)) for a, b in target_pairs]
    pair_cov = np.empty((reps, len(pair_idx)), dtype=bool); pair_sign = np.empty_like(pair_cov)
    for t in range(reps):
        R = rng.random(N) < pi
        n_adj[t] = R.sum()
        w = np.where(R, 1 / np.maximum(pi, 1e-300), 0.0)
        e = base + (r * w).sum(axis=1) / N
        v = (r**2 * np.where(R, (1 - pi) / np.maximum(pi, 1e-300)**2, 0.0)).sum(axis=1) / N**2
        est[t], se[t] = e, np.sqrt(v)
        cov_w[t] = np.abs(e - truth) <= 1.959964 * np.sqrt(v)
        width_w[t] = 2 * 1.959964 * np.sqrt(v)
        # conservative interval: discordant stratum exact when pi_D = 1; concordant correction bounded with
        # Clopper-Pearson on the positive and negative parts (Bonferroni), conditional on the sampled count.
        RC = R & C
        m = RC.sum()
        disc_part = (r[:, disc] * (1 / pi_D) * R[disc]).sum(axis=1) / N
        for k in range(len(ev)):
            kp = int(((r[k] > 0) & RC).sum()); kn = int(((r[k] < 0) & RC).sum())
            lp, hp = cp_interval(kp, m, ALPHA / 2); ln, hn = cp_interval(kn, m, ALPHA / 2)
            lo = base[k] + disc_part[k] + NC / N * (lp - hn)
            hi = base[k] + disc_part[k] + NC / N * (hp - ln)
            cov_cp[t, k] = lo <= truth[k] <= hi
            width_cp[t, k] = hi - lo
        for j, (a, b) in enumerate(pair_idx):
            dr_ = r[a] - r[b]
            ed = base[a] - base[b] + (dr_ * w).sum() / N
            vd = (dr_**2 * np.where(R, (1 - pi) / np.maximum(pi, 1e-300)**2, 0.0)).sum() / N**2
            td = truth[a] - truth[b]
            pair_cov[t, j] = abs(ed - td) <= 1.959964 * np.sqrt(vd)
            pair_sign[t, j] = np.sign(ed) == np.sign(td)
    err = est - truth
    return dict(pi_C=pi_C, pi_D=pi_D, mean_adjudicated=float(n_adj.mean()),
                max_abs_bias=float(np.abs(err.mean(0)).max()), mean_abs_bias=float(np.abs(err.mean(0)).mean()),
                rmse_mean=float(np.sqrt((err**2).mean(0)).mean()),
                rmse_flagger=float(np.sqrt((err[:, ev.index(F)]**2).mean())),
                coverage_wald_mean=float(cov_w.mean()), coverage_wald_min=float(cov_w.mean(0).min()),
                coverage_cp_mean=float(cov_cp.mean()), coverage_cp_min=float(cov_cp.mean(0).min()),
                width_wald_mean=float(width_w.mean()), width_cp_mean=float(width_cp.mean()),
                coverage_wald_flagger=float(cov_w[:, ev.index(F)].mean()),
                pair_coverage=[float(x) for x in pair_cov.mean(0)], pair_sign_correct=[float(x) for x in pair_sign.mean(0)])


def exp_two_phase():
    out = {}
    settings = {"A": "openai_gpt-4o-2024-05-13", "B:mistralai_mixtral-8x7b-32kseqlen": "meta_llama-3-70b"}
    for scen, F in settings.items():
        L, ev = labels_for(scen)
        order = sorted(ev, key=lambda m: -acc(P[m], Y))
        pairs = list(zip(order[:5], order[1:6]))   # adjacent pairs among the top six
        if F not in order[:6]:
            pairs.append((order[order.index(F) - 1], F))
        rows = []
        for pc in [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
            reps = 1 if pc == 0.0 else REPS   # DR (pi_C = 0) is deterministic
            rows.append(two_phase_sim(L, F, ev, pc, reps=reps, seed=int(pc * 1000), target_pairs=pairs))
        out[scen] = dict(flagger=F, pairs=pairs, rows=rows)
    return out


# ---------------------------------------------------------------- Exp 4: allocation at fixed budget
def exp_allocation():
    """RMSE (mean over models) at equal expected budget: uniform random audit vs two-phase (pi_D=1) vs Neyman."""
    out = {}
    settings = {"A": "openai_gpt-4o-2024-05-13", "B:mistralai_mixtral-8x7b-32kseqlen": "meta_llama-3-70b"}
    for scen, F in settings.items():
        L, ev = labels_for(scen)
        disc = P[F] != L
        ND, NC = int(disc.sum()), int((~disc).sum())
        Mstack = np.stack([P[m] for m in ev])
        r2 = (((Mstack == Y).astype(float) - (Mstack == L)) ** 2).sum(axis=0)   # summed over models
        A = {"D": r2[disc].sum(), "C": r2[~disc].sum()}
        rows = []
        for mult in [0.25, 0.5, 1.0, 1.25, 1.5, 2.0]:
            B = mult * ND

            def exact_var(pD, pC):   # exact design variance summed over models, divided by K
                return ((1 / pD - 1) * A["D"] + (1 / pC - 1) * A["C"]) / N**2 / len(ev)

            pu = B / N
            designs = {"uniform": (pu, pu)}
            if B > ND:
                designs["two_phase"] = (1.0, (B - ND) / NC)
            # Neyman: pi_h proportional to sqrt(A_h / N_h), capped at 1 (oracle A_h; see paper for pilot)
            s = {"D": np.sqrt(A["D"] / ND), "C": np.sqrt(A["C"] / NC)}
            lam = B / (ND * s["D"] + NC * s["C"])
            pD, pC = lam * s["D"], lam * s["C"]
            if pD > 1:
                pD, pC = 1.0, (B - ND) / NC
            designs["neyman"] = (pD, pC)
            rows.append(dict(budget_mult=mult, budget=B, **{f"{k}_pi": list(v) for k, v in designs.items()},
                             **{f"{k}_rmse": float(np.sqrt(exact_var(*v))) for k, v in designs.items()}))
        out[scen] = dict(flagger=F, ND=ND, NC=NC, eD=float(A["D"] / ND / len(ev)), eC=float(A["C"] / NC / len(ev)), rows=rows)
    return out


# ---------------------------------------------------------------- Exp 5: lock-in
def exp_lockin():
    out = {}
    for scen in ["A", "B:mistralai_mixtral-8x7b-32kseqlen"]:
        L, ev = labels_for(scen)
        old = [m for m in OLD if m in ev]
        new = [m for m in NEW if m in ev]
        rows = []
        for k in range(1, len(old) + 1):
            pan = old[:k]
            disc = np.any(np.stack([P[m] for m in pan]) != L, axis=0)
            J = (~disc) & (L != Y)
            D = dr_labels(L, Y, disc)
            rows.append(dict(panel_size=k, n_adjudicated=int(disc.sum()), n_hidden=int(J.sum()),
                             panel_bias={m: acc(P[m], D) - acc(P[m], Y) for m in pan},
                             new_bias={m: acc(P[m], D) - acc(P[m], Y) for m in new},
                             new_correct_on_J={m: int((J & (P[m] == Y)).sum()) for m in new},
                             new_repeat_on_J={m: int((J & (P[m] == L)).sum()) for m in new}))
        out[scen] = dict(old=old, new=new, rows=rows)
    return out


# ---------------------------------------------------------------- Exp 6: sensitivity bounds
def exp_sensitivity():
    out = {}
    for scen, F in {"A": "openai_gpt-4o-2024-05-13", "B:mistralai_mixtral-8x7b-32kseqlen": "meta_llama-3-70b"}.items():
        L, ev = labels_for(scen)
        disc = P[F] != L
        Yd = np.where(disc, Y, -9)          # only discordant reference labels are used
        order = sorted(ev, key=lambda m: -acc(P[m], Y))
        top = order[:6] if F in order[:6] else order[:5] + [F]
        rows = []
        pC = float((~disc).mean())
        for eps in [0.0, 0.001, 0.002, 0.004, 0.006, 0.01, 0.02, 0.035, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2]:
            eta = eps * pC
            b = {m: sensitivity_bounds(P[m], L, disc, Yd, eta) for m in top}
            pair = (order[0], order[1]) if scen == "A" else (order[order.index(F) - 1], F)
            bd = sensitivity_bounds_diff(P[pair[0]], P[pair[1]], L, disc, Yd, eta)
            rows.append(dict(eps=eps, eta=eta, bounds=b, pair=pair, pair_bounds=bd))
        out[scen] = dict(flagger=F, top=top, rows=rows, true={m: acc(P[m], Y) for m in top},
                         true_hidden_rate=float(((~disc) & (L != Y)).sum() / (~disc).sum()),
                         pair_true=acc(P[rows[0]["pair"][0]], Y) - acc(P[rows[0]["pair"][1]], Y))
    return out


# ---------------------------------------------------------------- Exp 7: rate in m (flagger, scenario B)
def exp_rate():
    L, ev = labels_for("B:mistralai_mixtral-8x7b-32kseqlen")
    F = "meta_llama-3-70b"
    disc = P[F] != L
    C = ~disc
    NC = C.sum()
    eps = float(((L != Y) & C).sum() / NC)
    rows = []
    rng = np.random.default_rng(7)
    rF = (P[F] == Y).astype(float) - (P[F] == L)
    truth = acc(P[F], Y)
    for m in [10, 20, 50, 100, 200, 500, 1000]:
        pc = m / NC
        errs = []
        for _ in range(REPS):
            R = (rng.random(N) < np.where(disc, 1.0, pc))
            e = np.mean(P[F] == L) + (rF * np.where(R, 1 / np.where(disc, 1.0, pc), 0)).sum() / N
            errs.append(e - truth)
        errs = np.array(errs)
        rows.append(dict(m=m, rmse=float(np.sqrt((errs**2).mean())), mae=float(np.abs(errs).mean()),
                         theory_se=float(NC / N * np.sqrt(eps * (1 - pc) / m)),
                         # Theorem (minimax lower bound): gamma * min(eps, sqrt(eps/m)) / (16 e)
                         lower_bound=float(NC / N * min(eps, np.sqrt(eps / m)) / (16 * np.e))))
    return dict(eps=eps, gamma=float(NC / N), rows=rows)


# ---------------------------------------------------------------- Exp 9: simultaneous exact intervals
def exp_simultaneous(reps=REPS, alpha=ALPHA):
    """Coverage and width of the simultaneous exact intervals (Theorem: simultaneous intervals) in the Table 4 setup."""
    out = {}
    settings = {"A": "openai_gpt-4o-2024-05-13", "B:mistralai_mixtral-8x7b-32kseqlen": "meta_llama-3-70b"}
    for scen, F in settings.items():
        L, ev = labels_for(scen)
        order = sorted(ev, key=lambda m: -acc(P[m], Y))
        adjacent = list(zip(order[:-1], order[1:]))              # all adjacent pairs of the true ranking
        models = {m: P[m] for m in ev}
        truth = {m: acc(P[m], Y) for m in ev}
        tdiff = {pr: truth[pr[0]] - truth[pr[1]] for pr in adjacent}
        disc = P[F] != L
        C = ~disc
        NC, K = int(C.sum()), int((C & (L != Y)).sum())
        rows = []
        for pc in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
            rng = np.random.default_rng(1000 + int(pc * 1000))
            ev_ok = sim_single = sim_pairs = 0
            wid_single, wid_pair, cert, wrong_cert, top_cov, zero_err = [], [], [], [], [], 0
            for _ in range(reps):
                R = (rng.random(N) < pc) & C
                m = int(R.sum()); h = int((R & (L != Y)).sum())
                zero_err += h == 0
                U = hypergeom_upper(h, NC, m, alpha)
                ev_ok += K <= U
                single, diff = simultaneous_bounds(models, L, Y, disc | R, C & ~R, U - h, pairs=adjacent)
                cov_s = [single[k][0] - 1e-12 <= truth[k] <= single[k][1] + 1e-12 for k in ev]
                cov_p = [diff[pr][0] - 1e-12 <= tdiff[pr] <= diff[pr][1] + 1e-12 for pr in adjacent]
                sim_single += all(cov_s); sim_pairs += all(cov_p)
                wid_single.append(np.mean([single[k][1] - single[k][0] for k in ev]))
                wid_pair.append(np.mean([diff[pr][1] - diff[pr][0] for pr in adjacent]))
                # certified = interval excludes 0 (all true adjacent differences are >= 0)
                cert.append(np.mean([diff[pr][0] > 0 for pr in adjacent]))
                wrong_cert.append(np.mean([diff[pr][1] < 0 for pr in adjacent]))
                top_cov.append(np.mean(cov_p[:5]))
            rows.append(dict(pi_C=pc, event_prob=ev_ok / reps, simultaneous_single=sim_single / reps,
                             simultaneous_adjacent_pairs=sim_pairs / reps, width_single=float(np.mean(wid_single)),
                             width_pair=float(np.mean(wid_pair)), certified_adjacent=float(np.mean(cert)),
                             wrongly_certified=float(np.mean(wrong_cert)), frac_reps_no_sampled_error=zero_err / reps))
        out[scen] = dict(flagger=F, K=K, NC=NC, n_adjacent=len(adjacent), alpha=alpha,
                         true_certifiable=float(np.mean([tdiff[pr] > 0 for pr in adjacent])), rows=rows)
    return out


# ---------------------------------------------------------------- Exp 8: labeler-strength sweep
def exp_sweep():
    """Every model in turn supplies the benchmark labels; summary over all single-model flaggers."""
    out = {}
    scen_list = ["A"] + [f"B:{m}" for m in MODELS]
    for scen in scen_list:
        L, ev = labels_for(scen)
        t = {m: acc(P[m], Y) for m in ev}
        infl, gains, fav = [], [], 0
        for F in ev:
            disc = P[F] != L
            D = dr_labels(L, Y, disc)
            d = {m: acc(P[m], D) for m in ev}
            infl.append(d[F] - t[F])
            true_rank = 1 + sum(t[m] > t[F] for m in ev)
            dr_rank = 1 + sum(d[m] > d[F] for m in ev)
            gains.append(true_rank - dr_rank)
            fav += any(t[m] > t[F] and d[m] < d[F] for m in ev)
        out[scen] = dict(label_error_rate=float(np.mean(L != Y)), median_inflation=float(np.median(infl)),
                         max_inflation=float(np.max(infl)), max_rank_gain=int(max(gains)),
                         median_rank_gain=float(np.median(gains)), n_flaggers_favoured=int(fav), n_flaggers=len(ev))
    return out


if __name__ == "__main__":
    which = sys.argv[1:] or ["bias", "two_phase", "allocation", "lockin", "sensitivity", "rate", "sweep", "simultaneous"]
    fns = dict(bias=exp_bias, two_phase=exp_two_phase, allocation=exp_allocation, lockin=exp_lockin,
               sensitivity=exp_sensitivity, rate=exp_rate, sweep=exp_sweep, simultaneous=exp_simultaneous)
    (OUT / "meta.json").write_text(json.dumps(RES, indent=1))
    for w in which:
        res = fns[w]()
        (OUT / f"{w}.json").write_text(json.dumps(res, indent=1, default=lambda o: o if not isinstance(o, np.generic) else o.item()))
        print("done", w)
