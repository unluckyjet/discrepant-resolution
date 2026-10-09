"""Core quantities for discrepant resolution (DR) and two-phase estimation.

Conventions: arrays are per item. L = original label, Y = reference (true) label,
M = model prediction (-1 = invalid output, always wrong), F = flagger prediction.
"""
import numpy as np

def acc(M, T):
    return np.mean(M == T)

def dr_labels(L, Y, discordant):
    """DR-resolved labels: reference label on discordant items, original label elsewhere."""
    return np.where(discordant, Y, L)

def panel_discordant(L, panel_preds):
    """Union rule: an item is discordant if any panel member disagrees with L."""
    return np.any(np.stack(panel_preds) != L, axis=0)

def bias_terms(M, L, Y, discordant):
    """Prop. 1: DR bias = P(J, M=L) - P(J, M=Y), with J = concordant & L != Y."""
    J = (~discordant) & (L != Y)
    return np.mean(J & (M == L)) - np.mean(J & (M == Y)), np.mean(J)

def two_phase(M, L, Y, discordant, R, pi):
    """Difference (bias-corrected) estimator and its design-based SE.

    R: adjudication indicator; pi: per-item inclusion probability (>0).
    theta_hat = mean(1{M=L}) + mean(R/pi * (1{M=Y} - 1{M=L})).
    Y is only read where R == 1.
    """
    r = np.where(R, (M == Y).astype(float) - (M == L), 0.0)
    n = len(M)
    w = np.where(R, 1 / np.where(pi > 0, pi, 1.0), 0.0)   # pi = 0 items are never sampled
    est = np.mean(M == L) + np.sum(r * w) / n
    var = np.sum((1 - pi) * w**2 * r**2) / n**2
    return est, np.sqrt(var)

def two_phase_diff(Ma, Mb, L, Y, discordant, R, pi):
    """Same estimator for the accuracy difference theta_a - theta_b."""
    ra = (Ma == Y).astype(float) - (Ma == L)
    rb = (Mb == Y).astype(float) - (Mb == L)
    r = np.where(R, ra - rb, 0.0)
    n = len(Ma)
    w = np.where(R, 1 / np.where(pi > 0, pi, 1.0), 0.0)
    est = np.mean(Ma == L) - np.mean(Mb == L) + np.sum(r * w) / n
    var = np.sum((1 - pi) * w**2 * r**2) / n**2
    return est, np.sqrt(var)

def sensitivity_bounds(M, L, discordant, Yd, eta):
    """Thm (sharp bounds): theta in [DR - min(eta, P(C, M=L)), DR + min(eta, P(C, M valid, M != L))].

    eta = bound on P(J) (hidden-error mass). Yd: labels known on discordant items.
    """
    C = ~discordant
    D = np.where(discordant, Yd, L)
    dr = np.mean(M == D)
    down = min(eta, np.mean(C & (M == L)))
    up = min(eta, np.mean(C & (M >= 0) & (M != L)))
    return dr - down, dr + up

def sensitivity_bounds_diff(Ma, Mb, L, discordant, Yd, eta, n_classes=4):
    """Sharp bounds on theta_a - theta_b given P(J) <= eta (fractional knapsack over concordant items)."""
    C = ~discordant
    D = np.where(discordant, Yd, L)
    base = np.mean(Ma == D) - np.mean(Mb == D)
    # moving item i into J with true label y != L changes (theta_a - theta_b) by
    # [1{Ma=y} - 1{Ma=L}] - [1{Mb=y} - 1{Mb=L}]
    # vectorised over concordant items and candidate true labels y != L
    Ma_, Mb_, L_ = Ma[C], Mb[C], L[C]
    ys = np.arange(n_classes)[None, :]
    g = (((Ma_[:, None] == ys).astype(int) - (Ma_ == L_)[:, None])
         - ((Mb_[:, None] == ys).astype(int) - (Mb_ == L_)[:, None])).astype(float)
    g[ys == L_[:, None]] = np.nan                      # y must differ from L
    gains = np.stack([np.nanmax(g, axis=1), np.nanmin(g, axis=1)], axis=1)
    n, cap = len(Ma), int(np.floor(eta * len(Ma) + 1e-9))
    up = np.sort(gains[:, 0])[::-1][:cap]
    lo = np.sort(gains[:, 1])[:cap]
    return base + lo[lo < 0].sum() / n, base + up[up > 0].sum() / n


# ---------------------------------------------------------------- simultaneous exact intervals
def hypergeom_upper(h, N, m, alpha, _cache={}):
    """Exact (1-alpha) upper confidence bound for the number K of 'successes' in a population of N,
    after observing h successes in a simple random sample of m without replacement:
    U = max{K : P(H <= h; K) > alpha}. P(H <= h; K) is non-increasing in K."""
    from scipy.stats import hypergeom
    key = (int(h), int(N), int(m), float(alpha))
    if key in _cache:
        return _cache[key]
    if m == 0:
        _cache[key] = int(N); return int(N)
    K = np.arange(h, N - (m - h) + 1)
    ok = hypergeom.cdf(h, N, K, m) > alpha
    U = int(K[ok].max())
    _cache[key] = U
    return U


def simultaneous_bounds(models, L, Y, known, cand, budget, pairs=()):
    """Bounds for every accuracy and pairwise difference, valid simultaneously whenever the number
    of label errors among the candidate (unadjudicated concordant) items is at most `budget`.

    known: items whose reference label is observed (discordant + sampled concordant).
    cand: unadjudicated concordant items (their label is L unless they are hidden errors).
    Returns ({k: (lo, hi)}, {(a, b): (lo, hi)}) for accuracies and differences theta_a - theta_b.
    """
    n = len(L)
    D = np.where(known, Y, L)
    single = {}
    for k, M in models.items():
        base = np.mean(M == D)
        down = min(budget, int((cand & (M == L)).sum()))
        up = min(budget, int((cand & (M >= 0) & (M != L)).sum()))
        single[k] = (base - down / n, base + up / n)
    diff = {}
    for a, b in pairs:
        Ma, Mb = models[a], models[b]
        base = np.mean(Ma == D) - np.mean(Mb == D)
        # gain of making candidate item i a hidden error with true label y != L:
        # g = [1{Ma=y} - 1{Ma=L}] - [1{Mb=y} - 1{Mb=L}], values in {-2,...,2}
        aL, bL = (Ma == L), (Mb == L)
        aV, bV = (Ma >= 0) & ~aL, (Mb >= 0) & ~bL           # valid answers different from L
        same = aV & bV & (Ma == Mb)
        # max over y != L: choose y = Ma if valid (and != Mb), else any y avoiding Mb
        gmax = np.where(aV & ~same, 1, 0) - aL.astype(int) + bL.astype(int)
        gmin = -np.where(bV & ~same, 1, 0) - aL.astype(int) + bL.astype(int)
        # when a and b give the same valid non-L answer, y = that answer gives 0 and y = other gives 0 too
        gmax = np.where(same, 0, gmax); gmin = np.where(same, 0, gmin)
        c = cand
        up_counts = [int((c & (gmax == v)).sum()) for v in (2, 1)]
        lo_counts = [int((c & (gmin == v)).sum()) for v in (-2, -1)]
        B, up = budget, 0
        for v, cnt in zip((2, 1), up_counts):
            take = min(B, cnt); up += v * take; B -= take
        B, lo = budget, 0
        for v, cnt in zip((-2, -1), lo_counts):
            take = min(B, cnt); lo += v * take; B -= take
        diff[(a, b)] = (base + lo / n, base + up / n)
    return single, diff
