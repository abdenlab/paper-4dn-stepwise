# /// script
# requires-python = ">=3.10"
# dependencies = ["polars", "numpy", "scipy", "pyarrow"]
# ///
"""limma-trend differential accessibility on the cCRE insertion matrix.

Pipeline:
1. Filter: log2-CPM >= -2 in >= 2 samples (~60% of cCREs survive).
2. Apply per-sample loess offsets (already computed) to log-CPM.
3. Fit cell-means OLS per cCRE on the 5-stage design (no intercept).
4. Compute residual variance sigma2 per cCRE.
5. Fit a smooth mean-variance trend g(mean_lcpm) with binned smoother.
6. Estimate prior degrees of freedom d0 via Smyth (2004) moment-matching
   on the trigamma scale; compute moderated variance:
       v_post(g) = (d0 * g(mean_g) + d * sigma2(g)) / (d0 + d)
   where d = residual df (= 10 - 5 = 5).
7. For each of 6 contrasts, compute moderated t = c'beta / sqrt(v_post * c'(X'X)^-1 c),
   p-value from t distribution with df = d + d0, BH-adjusted FDR per contrast.
8. Write a single wide Parquet with cCRE coords + per-contrast log2FC/p/padj.

Pure-Python limma-trend (no R, no rpy2). scipy.special is used for digamma/trigamma
in d0 estimation, scipy.stats.t for p-values.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats
from scipy.special import polygamma

COUNT_PATH = Path(__file__).resolve().parent / "results" / "ccre_insertion_matrix.parquet"
OFFSET_PATH = Path(__file__).resolve().parent / "results" / "loess_offsets.parquet"
OUT_PATH = Path(__file__).resolve().parent / "results" / "ccre_da_results.parquet"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

SAMPLES = [
    "01_ESC.R1", "01_ESC.R2",
    "02_DE.R1",  "02_DE.R2",
    "03_HB.R1",  "03_HB.R2",
    "04_iHEP.R1", "04_iHEP.R2",
    "05_mHEP.R1", "05_mHEP.R2",
]
STAGES = ["01_ESC", "02_DE", "03_HB", "04_iHEP", "05_mHEP"]

# Each contrast vector multiplies a 5-vector of stage means in STAGES order.
CONTRASTS: dict[str, list[float]] = {
    "DE_vs_ESC":         [-1, 1, 0, 0, 0],
    "HB_vs_DE":          [0, -1, 1, 0, 0],
    "iHEP_vs_HB":        [0, 0, -1, 1, 0],
    "mHEP_vs_iHEP":      [0, 0, 0, -1, 1],
    "hepatic_vs_pre":    [-1/2, -1/2, 1/3, 1/3, 1/3],
    "mature_vs_immature": [0, 0, -1/2, -1/2, 1],
}


def trigamma(x: np.ndarray | float) -> np.ndarray | float:
    return polygamma(1, x)


def trigamma_inverse(y: float) -> float:
    """Solve trigamma(x) = y, x > 0, by Newton's method."""
    if y > 1e6:  # very high variance -> small x
        return 1.0 / y
    if y <= 0:
        return float("inf")
    x = 0.5 + 1.0 / y
    for _ in range(50):
        d = (trigamma(x) - y) / polygamma(2, x)
        x_new = x - d
        if abs(x_new - x) < 1e-10 * max(abs(x_new), 1):
            return float(x_new)
        x = x_new
    return float(x)


def fit_smooth_trend(
    x: np.ndarray, y: np.ndarray,
    *, n_bins: int = 200, trim: float = 0.10, smoothing: int = 7,
) -> np.ndarray:
    """Smooth y as a function of x via quantile-binned trimmed mean +
    moving-average + linear interpolation. Returns smoothed y at each x."""
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    edges = np.maximum.accumulate(edges)
    edges[-1] += 1e-9
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    starts = np.searchsorted(xs, edges[:-1], side="left")
    ends = np.searchsorted(xs, edges[1:], side="left")
    centers = np.empty(n_bins)
    fitted = np.empty(n_bins)
    for b in range(n_bins):
        s, e = starts[b], ends[b]
        if e - s < 5:
            centers[b] = (edges[b] + edges[b + 1]) / 2
            fitted[b] = np.nan
            continue
        a_in = xs[s:e]
        m_in = np.sort(ys[s:e])
        lo = int(len(m_in) * trim)
        hi = len(m_in) - lo
        centers[b] = float(np.median(a_in))
        fitted[b] = float(np.mean(m_in[lo:hi])) if hi > lo else float(np.median(m_in))
    valid = ~np.isnan(fitted)
    if not valid.all():
        fitted[~valid] = np.interp(centers[~valid], centers[valid], fitted[valid])
    if smoothing > 1:
        pad_l = smoothing // 2
        pad_r = smoothing - pad_l - 1
        padded = np.concatenate([
            np.full(pad_l, fitted[0]), fitted, np.full(pad_r, fitted[-1]),
        ])
        kernel = np.ones(smoothing) / smoothing
        fitted = np.convolve(padded, kernel, mode="valid")
    return np.interp(x, centers, fitted)


def main() -> None:
    df = pl.read_parquet(COUNT_PATH)
    offsets_df = pl.read_parquet(OFFSET_PATH)
    counts = df.select(SAMPLES).to_numpy().astype(np.float64)
    offsets = offsets_df.select(SAMPLES).to_numpy()
    lib = counts.sum(axis=0)
    log_cpm = np.log2((counts + 1) / lib[None, :] * 1e6)

    # 1. Filter
    mask = (log_cpm >= -2.0).sum(axis=1) >= 2
    n_in, n_out = len(mask), int(mask.sum())
    print(f"Filter (log2-CPM >= -2 in >= 2 samples): {n_out:,} of {n_in:,} cCREs ({n_out/n_in*100:.1f}%)")
    df_f = df.filter(pl.Series(mask))
    log_cpm = log_cpm[mask]
    offsets = offsets[mask]

    # 2. Apply loess offsets
    log_cpm_n = log_cpm - offsets
    Y = log_cpm_n  # (n_genes, 10) in log-CPM space, normalized

    # 3. Cell-means design matrix
    X = np.zeros((len(SAMPLES), len(STAGES)))
    for i, s in enumerate(SAMPLES):
        X[i, STAGES.index(s.split(".")[0])] = 1.0
    XtX_inv = np.linalg.inv(X.T @ X)

    # OLS per cCRE: beta = (X'X)^-1 X' Y'  -> shape (5, n_genes)
    beta = XtX_inv @ X.T @ Y.T
    fitted = X @ beta              # (10, n_genes)
    resid = Y.T - fitted           # (10, n_genes)
    rss = (resid ** 2).sum(axis=0) # (n_genes,)
    d_resid = X.shape[0] - X.shape[1]  # 10 - 5 = 5
    sigma2 = rss / d_resid
    mean_lcpm = Y.mean(axis=1)

    # 4-5. Trend on log-sigma2 vs mean_lcpm (limma's choice for log-CPM data)
    log_sigma2 = np.log(np.maximum(sigma2, 1e-300))
    trend_log = fit_smooth_trend(mean_lcpm, log_sigma2, n_bins=200, trim=0.10, smoothing=7)
    trend = np.exp(trend_log)
    print(f"\nResidual df (per cCRE): {d_resid}")
    print(f"Median sigma2: {np.median(sigma2):.4f}   median trend(mean): {np.median(trend):.4f}")

    # 6. Estimate d0 via Smyth (2004) moment-matching on z = log(sigma2) - log(trend)
    z = log_sigma2 - trend_log
    z_var = float(np.var(z, ddof=1))
    diff = z_var - float(trigamma(d_resid / 2))
    if diff <= 0:
        d0 = float("inf")
    else:
        d0 = 2.0 * trigamma_inverse(diff)
    print(f"Empirical Var[log sigma^2]: {z_var:.4f} (theoretical chi-sq term: {trigamma(d_resid/2):.4f})")
    print(f"Estimated prior df d0: {d0:.3f}")

    if np.isinf(d0):
        # Full shrinkage to trend
        v_post = trend.copy()
        df_total = float("inf")
    else:
        v_post = (d0 * trend + d_resid * sigma2) / (d0 + d_resid)
        df_total = d_resid + d0
    print(f"Total df for moderated t: {df_total}")
    print(f"Median posterior variance: {np.median(v_post):.4f}\n")

    # 7. Per-contrast moderated t, p, padj
    out_cols = {
        "chrom":   df_f["chrom"],
        "start":   df_f["start"],
        "end":     df_f["end"],
        "ccre_id": df_f["ccre_id"],
        "ccre_class": df_f["ccre_class"],
        "mean_lcpm": pl.Series(mean_lcpm),
        "sigma2":     pl.Series(sigma2),
        "post_var":   pl.Series(v_post),
    }

    print(f"{'contrast':<25} {'sig (FDR<0.05)':>14} {'sig (FDR<0.01)':>16} {'sig (FDR<0.001)':>17}")
    print("-" * 75)
    for name, c in CONTRASTS.items():
        c_arr = np.asarray(c)
        c_var_factor = float(c_arr @ XtX_inv @ c_arr)
        log2fc = c_arr @ beta                                     # (n_genes,)
        se = np.sqrt(v_post * c_var_factor)
        t_stat = log2fc / se
        if np.isinf(df_total):
            # As df -> inf, t -> standard normal
            p = 2 * stats.norm.sf(np.abs(t_stat))
        else:
            p = 2 * stats.t.sf(np.abs(t_stat), df=df_total)
        # BH FDR
        order = np.argsort(p)
        n_t = len(p)
        ranks = np.empty(n_t, dtype=np.int64)
        ranks[order] = np.arange(1, n_t + 1)
        padj_sorted = np.minimum.accumulate(
            (p[order] * n_t / np.arange(1, n_t + 1))[::-1]
        )[::-1]
        padj = np.empty(n_t)
        padj[order] = np.clip(padj_sorted, 0, 1)

        out_cols[f"log2FC_{name}"] = pl.Series(log2fc)
        out_cols[f"p_{name}"]      = pl.Series(p)
        out_cols[f"padj_{name}"]   = pl.Series(padj)

        n_05  = int((padj < 0.05).sum())
        n_01  = int((padj < 0.01).sum())
        n_001 = int((padj < 0.001).sum())
        print(f"{name:<25} {n_05:>14,} {n_01:>16,} {n_001:>17,}")

    pl.DataFrame(out_cols).write_parquet(OUT_PATH)
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
