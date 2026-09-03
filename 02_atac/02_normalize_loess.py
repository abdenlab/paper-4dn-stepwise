# /// script
# requires-python = ">=3.10"
# dependencies = ["polars", "numpy", "matplotlib", "pyarrow"]
# ///
"""Per-sample loess M-vs-A normalization for the cCRE insertion matrix.

The csaw/voom style: for each sample j, fit a smooth (M ~ A) bias trend
against a reference (here: geometric mean of all samples on log2-CPM scale).
The fitted curve is the per-cCRE-per-sample offset; subtract it from
log-CPM to remove abundance-dependent scale bias that scalar TMM can't fix.

Loess is approximated with binned trimmed-mean smoothing (numpy only):
  - quantile-bin A into N bins (so each bin holds ~equal cCRE counts)
  - per bin: trimmed mean of M (30% trim by default — same as TMM)
  - moving-average smooth across bins, then linear-interpolate back to
    every cCRE's A
This is fast (sub-second per sample on 2.35M cCREs) and gives a robust
non-parametric trend without needing statsmodels/lowess on millions of points.

Constraint: offsets are zero-meaned per cCRE so the overall log-CPM scale is
preserved (csaw convention).

Reads:
  results/ccre_insertion_matrix.parquet

Outputs:
  results/loess_offsets.parquet   ccre_id + 10 offset cols (~190MB)
  results/ccre_norm_lcpm.parquet  per-replicate normalized log2-CPM on the
                                  expressed-cCRE filter (log2-CPM >= -2 in >= 2
                                  samples). This is just log-CPM minus the
                                  offsets fitted here, so it is written in the
                                  same pass.
  results/loess_curves.parquet    (sample, A, fitted_M) for diagnostics
  figs/02_ma_per_sample_vs_geomean_postloess.png
  figs/02_ma_adjacent_stages_loess_compare.png  (pre vs post, 2x4 grid)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

COUNT_PATH = Path(__file__).resolve().parent / "results" / "ccre_insertion_matrix.parquet"
OUT_DIR = Path(__file__).resolve().parent
RESULTS = OUT_DIR / "results"; RESULTS.mkdir(parents=True, exist_ok=True)
FIGS = OUT_DIR / "figs"; FIGS.mkdir(parents=True, exist_ok=True)


SAMPLES = [
    "01_ESC.R1", "01_ESC.R2",
    "02_DE.R1",  "02_DE.R2",
    "03_HB.R1",  "03_HB.R2",
    "04_iHEP.R1", "04_iHEP.R2",
    "05_mHEP.R1", "05_mHEP.R2",
]
STAGES = ["01_ESC", "02_DE", "03_HB", "04_iHEP", "05_mHEP"]


def binned_trimmed_loess(
    A: np.ndarray, M: np.ndarray,
    *, n_bins: int = 400, trim: float = 0.30, smoothing: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a smooth E[M | A] curve via quantile-binned trimmed mean.

    Returns (fitted_per_point, bin_centers, fitted_per_bin) — the first is the
    interpolated trend evaluated at every input A; the latter two are the
    summarized curve for diagnostic plotting.
    """
    n = A.size
    edges = np.quantile(A, np.linspace(0, 1, n_bins + 1))
    # Make edges strictly increasing in case of repeated quantile values
    edges = np.maximum.accumulate(edges)
    edges[-1] += 1e-9

    centers = np.empty(n_bins)
    fitted = np.empty(n_bins)

    # Sort by A once; bin via searchsorted on sorted A is O(n)
    order = np.argsort(A, kind="stable")
    A_sorted = A[order]
    M_sorted = M[order]
    starts = np.searchsorted(A_sorted, edges[:-1], side="left")
    ends = np.searchsorted(A_sorted, edges[1:], side="left")

    for b in range(n_bins):
        s, e = starts[b], ends[b]
        if e - s < 5:
            centers[b] = (edges[b] + edges[b + 1]) / 2
            fitted[b] = np.nan
            continue
        a_in = A_sorted[s:e]
        m_in = M_sorted[s:e]
        # Trim by M
        m_sorted = np.sort(m_in)
        lo = int(len(m_sorted) * trim)
        hi = len(m_sorted) - lo
        centers[b] = float(np.median(a_in))
        fitted[b] = float(np.mean(m_sorted[lo:hi])) if hi > lo else float(np.median(m_sorted))

    valid = ~np.isnan(fitted)
    if not valid.all():
        fitted[~valid] = np.interp(centers[~valid], centers[valid], fitted[valid])

    if smoothing > 1:
        kernel = np.ones(smoothing) / smoothing
        # Edge-pad for moving average so endpoints don't pull toward 0
        padded = np.concatenate([
            np.full(smoothing // 2, fitted[0]),
            fitted,
            np.full(smoothing - smoothing // 2 - 1, fitted[-1]),
        ])
        fitted = np.convolve(padded, kernel, mode="valid")

    fitted_per_point = np.interp(A, centers, fitted)
    return fitted_per_point, centers, fitted


def fit_per_sample_offsets(
    log_cpm: np.ndarray, **kw,
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """For each sample, fit the M-vs-A bias trend against the geometric mean
    of all samples (in log-CPM space). Returns:
      offsets : (n_ccres, n_samples) — to subtract from log_cpm
      curves  : list per sample of (bin_centers, fitted_M) for plotting
    Offsets are zero-meaned per cCRE (preserves the geomean reference).
    """
    log_geom = log_cpm.mean(axis=1)
    n_ccres, n_samples = log_cpm.shape
    offsets = np.zeros_like(log_cpm)
    curves: list[tuple[np.ndarray, np.ndarray]] = []
    for j in range(n_samples):
        M = log_cpm[:, j] - log_geom
        A = 0.5 * (log_cpm[:, j] + log_geom)
        offset_j, centers, fitted = binned_trimmed_loess(A, M, **kw)
        offsets[:, j] = offset_j
        curves.append((centers, fitted))
    offsets -= offsets.mean(axis=1, keepdims=True)
    return offsets, curves


def hexbin_panel(ax, A, M, title: str) -> None:
    ax.hexbin(A, M, gridsize=80, bins="log", cmap="viridis", mincnt=1)
    ax.axhline(0, color="red", lw=0.6, alpha=0.8)
    med = float(np.median(M))
    ax.axhline(med, color="orange", lw=0.6, alpha=0.8,
               label=f"median M = {med:+.3f}")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="lower right", fontsize=7, framealpha=0.7)


def main() -> None:
    df = pl.read_parquet(COUNT_PATH)
    counts = df.select(SAMPLES).to_numpy().astype(np.float64)
    lib_sizes = counts.sum(axis=0)
    log_cpm = np.log2((counts + 1) / lib_sizes[None, :] * 1e6)

    print(f"Fitting loess offsets on {log_cpm.shape[0]:,} cCREs x {log_cpm.shape[1]} samples...")
    offsets, curves = fit_per_sample_offsets(log_cpm, n_bins=400, trim=0.3, smoothing=5)
    log_cpm_post = log_cpm - offsets
    print("Done.\n")

    # Quick stratified bias check: median M (vs geomean) per sample, in
    # low/mid/high abundance tiers (same as plot_postnorm.py).
    gm_pre  = log_cpm.mean(axis=1)
    gm_post = log_cpm_post.mean(axis=1)
    mean_lcpm_post = gm_post
    tier_idx = np.digitize(mean_lcpm_post, [-2.0, 1.0])  # 0=low, 1=mid, 2=high
    tier_names = ["low", "mid", "high"]

    print(f"{'sample':12} | {'tier':>5} | {'med M (pre)':>12} | {'med M (loess)':>14}")
    print("-" * 60)
    for i, s_ in enumerate(SAMPLES):
        for t in range(3):
            mask = tier_idx == t
            mp = float(np.median(log_cpm[mask, i]      - gm_pre[mask]))
            mq = float(np.median(log_cpm_post[mask, i] - gm_post[mask]))
            print(f"{s_:12} | {tier_names[t]:>5} | {mp:>+12.3f} | {mq:>+14.3f}")
        print("-" * 60)

    # Save offset matrix
    ccre_id = df["ccre_id"]
    offset_df = pl.DataFrame({"ccre_id": ccre_id})
    for i, s_ in enumerate(SAMPLES):
        offset_df = offset_df.with_columns(pl.Series(s_, offsets[:, i]))
    offset_df.write_parquet(RESULTS / "loess_offsets.parquet")
    print(f"\n-> {RESULTS / 'loess_offsets.parquet'}")

    # Save the fitted curves for diagnostics
    curve_rows = []
    for s_, (cs, fs) in zip(SAMPLES, curves):
        for c, f in zip(cs, fs):
            curve_rows.append({"sample": s_, "A": float(c), "fitted_M": float(f)})
    pl.DataFrame(curve_rows).write_parquet(RESULTS / "loess_curves.parquet")
    print(f"-> {RESULTS / 'loess_curves.parquet'}")

    # Per-replicate normalized log2-CPM on the expressed-cCRE filter. Same
    # quantity the DA model uses; persisted here so 06, 07 and 20 can reuse it.
    mask = (log_cpm >= -2.0).sum(axis=1) >= 2
    print(f"\nExpressed-cCRE filter: {mask.sum():,} of {len(mask):,} "
          f"({mask.sum()/len(mask)*100:.1f}%)")
    Y = log_cpm_post[mask]
    (df.filter(pl.Series(mask))
       .select(["chrom", "start", "end", "ccre_id", "ccre_class"])
       .with_columns([pl.Series(s_, Y[:, i]) for i, s_ in enumerate(SAMPLES)])
       .write_parquet(RESULTS / "ccre_norm_lcpm.parquet"))
    print(f"-> {RESULTS / 'ccre_norm_lcpm.parquet'}  "
          f"({int(mask.sum()):,} cCREs x {len(SAMPLES)} samples)")

    # Per-sample-vs-geomean MA after loess
    fig, axes = plt.subplots(2, 5, figsize=(22, 9), sharex=True, sharey=True)
    for i, s_ in enumerate(SAMPLES):
        M = log_cpm_post[:, i] - gm_post
        A = 0.5 * (log_cpm_post[:, i] + gm_post)
        hexbin_panel(axes.flat[i], A, M, f"{s_} (post-loess)")
    for ax in axes[-1]:
        ax.set_xlabel("A")
    for ax in axes[:, 0]:
        ax.set_ylabel("M = log₂(sample / geomean)")
    fig.suptitle("MA per sample vs geomean — after per-sample loess offsets", fontsize=11)
    fig.tight_layout()
    out = FIGS / "02_ma_per_sample_vs_geomean_postloess.png"
    fig.savefig(out, dpi=120)
    print(f"-> {out}")

    # Adjacent-stage MA pre vs post (2x4)
    def stage_log(arr: np.ndarray) -> np.ndarray:
        return np.column_stack([
            0.5 * (arr[:, SAMPLES.index(f"{stg}.R1")]
                   + arr[:, SAMPLES.index(f"{stg}.R2")])
            for stg in STAGES
        ])
    sl_pre  = stage_log(log_cpm)
    sl_post = stage_log(log_cpm_post)

    fig, axes = plt.subplots(2, 4, figsize=(22, 9), sharex=True, sharey=True)
    for i in range(4):
        b, a = sl_pre[:, i],  sl_pre[:, i + 1]
        M, A = a - b, 0.5 * (a + b)
        hexbin_panel(axes[0, i], A, M, f"{STAGES[i+1]} vs {STAGES[i]} (pre)")
        b, a = sl_post[:, i], sl_post[:, i + 1]
        M, A = a - b, 0.5 * (a + b)
        hexbin_panel(axes[1, i], A, M, f"{STAGES[i+1]} vs {STAGES[i]} (post-loess)")
    for ax in axes[-1]:
        ax.set_xlabel("A = ½ log₂(later · earlier)")
    axes[0, 0].set_ylabel("pre-norm  M")
    axes[1, 0].set_ylabel("post-loess M")
    fig.suptitle("Adjacent-stage MA: pre vs post per-sample loess (R1+R2 pooled)", fontsize=11)
    fig.tight_layout()
    out = FIGS / "02_ma_adjacent_stages_loess_compare.png"
    fig.savefig(out, dpi=120)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
