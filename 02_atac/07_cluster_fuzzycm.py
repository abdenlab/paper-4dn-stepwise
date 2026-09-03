# /// script
# requires-python = ">=3.10"
# dependencies = ["polars", "numpy", "pyarrow"]
# ///
"""Clustered epigenetic heatmap over the full DA cCRE union.

Clustering: fuzzy c-means (c=8, chosen by the validity sweep in 06_cluster_fuzzycm_select.py) on
the 5-stage ATAC accessibility trajectory only.

Inputs (both already cached):
  results/ccre_mark_fc_matrix.parquet   union cCREs x 20 mean-fc columns
  results/ccre_norm_lcpm.parquet        per-replicate normalized log2-CPM

Outputs:
  results/ccre_epigenetic_clusters.parquet   ccre_id + cluster + features
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

DIR = Path(__file__).resolve().parent

MAT_PATH = DIR / "results" / "ccre_mark_fc_matrix.parquet"
LCPM_PATH = DIR / "results" / "ccre_norm_lcpm.parquet"
OUT_CLUST = DIR / "results" / "ccre_epigenetic_clusters.parquet"
OUT_CLUST.parent.mkdir(parents=True, exist_ok=True)

STAGES = ["ESC", "DE", "HB", "iHEP", "mHEP"]
MARKS = ["H3K4me3", "H3K27ac", "H3K27me3", "H3K9me3"]
SAMPLES = [  # per-replicate columns in ccre_norm_lcpm.parquet
    "01_ESC.R1", "01_ESC.R2", "02_DE.R1", "02_DE.R2", "03_HB.R1", "03_HB.R2",
    "04_iHEP.R1", "04_iHEP.R2", "05_mHEP.R1", "05_mHEP.R2",
]
C = 8           # fuzzy c-means clusters (chosen by the sweep in 06)
M_FUZZ = 2.0
SEED = 0
# manual archetype display order (1-based, over the peak-stage labelling)
DISPLAY_ORDER = [1, 4, 3, 2, 5, 7, 8, 6]


def fcm(X, c, m=M_FUZZ, n_iter=200, tol=1e-5, seed=SEED):
    """Fuzzy c-means. Returns (centroids c x d, memberships c x n)."""
    rng = np.random.default_rng(seed)
    U = rng.random((c, X.shape[0])); U /= U.sum(0, keepdims=True)
    cent = None
    for _ in range(n_iter):
        Um = U ** m
        cent = (Um @ X) / Um.sum(1, keepdims=True)
        d2 = np.maximum(((X[None] - cent[:, None]) ** 2).sum(2), 1e-12)
        p = 1.0 / (m - 1)
        Un = np.empty_like(U)
        for i in range(c):
            Un[i] = 1.0 / ((d2[i][None] / d2) ** p).sum(0)
        if np.abs(Un - U).max() < tol:
            U = Un; break
        U = Un
    return cent, U


def main():
    mat = pl.read_parquet(MAT_PATH)
    n = mat.height
    print(f"Union DA cCREs: {n:,}")

    # 20 mark columns (4 histones x 5 stages); log2 fold-change for display
    mark_cols = [f"{m}.{s}" for m in MARKS for s in STAGES]
    M = mat.select(mark_cols).to_numpy()
    logM = np.log2(np.clip(M, 2 ** -4, None))

    # 5 ATAC stage means (normalized log2-CPM), aligned by ccre_id
    lcpm = pl.read_parquet(LCPM_PATH)
    lc = (mat.select("ccre_id").join(lcpm, on="ccre_id", how="left"))
    atac = np.empty((n, 5))
    for j, s in enumerate(STAGES):
        stg = SAMPLES[2 * j].split(".")[0]            # e.g. 01_ESC
        r1 = lc[f"{stg}.R1"].to_numpy(); r2 = lc[f"{stg}.R2"].to_numpy()
        atac[:, j] = (r1 + r2) / 2
    if np.isnan(atac).any():
        miss = int(np.isnan(atac).any(axis=1).sum())
        print(f"  warning: {miss} cCREs missing ATAC (filling with row min)")
        atac = np.where(np.isnan(atac), np.nanmin(atac, axis=1, keepdims=True), atac)

    # ---- fuzzy c-means on the 5-stage ATAC trajectory (magnitude-preserving) ----
    # The dynamic class = accessibility trajectory; all marks are display-only.
    X = (atac - atac.mean(0)) / atac.std(0)          # per-stage column z (keeps magnitude)
    print(f"Fuzzy c-means: c={C} on the 5-stage ATAC trajectory over {n:,} cCREs...")
    cent, U = fcm(X, C)
    lab = U.argmax(0)
    memb = U.max(0)

    # order clusters by peak ATAC stage (ESC-high -> mHEP-high), then mean level
    peak_stage = np.array([atac[lab == c].mean(0).argmax() for c in range(C)])
    mean_atac = np.array([atac[lab == c].mean() for c in range(C)])
    order = sorted(range(C), key=lambda c: (peak_stage[c], -mean_atac[c]))
    remap = {old: new for new, old in enumerate(order)}
    lab = np.array([remap[c] for c in lab])

    # apply the manual archetype display order on top of the peak-stage labelling
    if sorted(DISPLAY_ORDER) == list(range(1, C + 1)):
        reorder = {old - 1: new for new, old in enumerate(DISPLAY_ORDER)}
        lab = np.array([reorder[c] for c in lab])

    # within each cluster, order rows (descending) by a cluster-specific leading
    # stage average: early clusters (1-3) by ESC+DE, the DE-high cluster (4) by all
    # 5 stages, the hepatic-gain clusters (5-8) by HB+iHEP+mHEP.
    esc_de = atac[:, [0, 1]].mean(1)        # ESC, DE
    all5 = atac.mean(1)
    hep = atac[:, [2, 3, 4]].mean(1)        # HB, iHEP, mHEP
    key = np.where(np.isin(lab, [0, 1, 2]), esc_de,
                   np.where(lab == 3, all5, hep))
    row_order = np.lexsort((-key, lab))
    lab_s = lab[row_order]; atac_s = atac[row_order]; logM_s = logM[row_order]
    memb_s = memb[row_order]

    sizes = [int((lab_s == c).sum()) for c in range(C)]
    print("cluster sizes:", "  ".join(f"{c+1}:{n:,}" for c, n in enumerate(sizes)))

    save_clusters(mat, row_order, lab_s, atac_s, logM_s, memb_s)


def save_clusters(mat, row_order, lab_s, atac_s, logM_s, memb_s):
    base = mat.select(["chrom", "start", "end", "ccre_id"])[row_order]
    out = base.with_columns(cluster=pl.Series(lab_s + 1),
                            membership=pl.Series(memb_s))
    out = out.with_columns([pl.Series(f"atac_{s}", atac_s[:, j])
                            for j, s in enumerate(STAGES)])
    mark_cols = [f"{m}.{s}" for m in MARKS for s in STAGES]
    out = out.with_columns([pl.Series(f"log2fc_{mark_cols[j]}", logM_s[:, j])
                            for j in range(logM_s.shape[1])])
    out.write_parquet(OUT_CLUST)
    print(f"-> {OUT_CLUST}")


if __name__ == "__main__":
    main()
