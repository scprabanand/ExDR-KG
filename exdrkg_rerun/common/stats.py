"""Statistics shared by several modules."""
import numpy as np


def krippendorff_alpha_ordinal(M):
    """M: items x raters array with np.nan for missing. Ordinal metric, Krippendorff (2011)."""
    vals = np.unique(M[~np.isnan(M)])
    if len(vals) < 2: return float("nan")
    idx = {v: i for i, v in enumerate(vals)}; k = len(vals)
    coinc = np.zeros((k, k))
    for row in M:
        r = row[~np.isnan(row)]; m = len(r)
        if m < 2: continue
        for i in range(m):
            for j in range(m):
                if i != j: coinc[idx[r[i]], idx[r[j]]] += 1.0 / (m - 1)
    n_c = coinc.sum(1); n = n_c.sum()
    def d2(c, g):   # ordinal squared difference
        lo, hi = min(c, g), max(c, g)
        return (n_c[lo:hi + 1].sum() - (n_c[lo] + n_c[hi]) / 2) ** 2
    Do = sum(coinc[c, g] * d2(c, g) for c in range(k) for g in range(k)) / n
    De = sum(n_c[c] * n_c[g] * d2(c, g) for c in range(k) for g in range(k)) / (n * (n - 1))
    return float(1 - Do / De) if De > 0 else float("nan")
