# /// script
# requires-python = ">=3.10"
# dependencies = ["polars", "numpy", "scikit-learn", "evoc", "pyarrow", "matplotlib"]
# ///
"""Histone sub-clustering within each ATAC archetype (k-free).

Same primary partition as 07_cluster_fuzzycm.py (8 fuzzy-c-means ATAC
trajectory archetypes, same display order), but we sub-cluster them on the
4 HISTONE marks (H3K4me3/H3K27ac/H3K27me3/H3K9me3 x 5 stages = 20 features)
with EVoC (Tutte Institute) — a k-free, ensemble-of-clusterings successor to
HDBSCAN. Low-confidence rows are labelled noise; sub-clusters below
MIN_SUB_SIZE are folded into noise for readability.

Output:
  results/ccre_histsub_clusters.parquet
"""
from __future__ import annotations

from pathlib import Path

import evoc
import numpy as np
import polars as pl

DIR = Path(__file__).resolve().parent
RESULTS = DIR / "results"; RESULTS.mkdir(parents=True, exist_ok=True)
CLUST_PATH = RESULTS / "ccre_epigenetic_clusters.parquet"
OUT_CLUST = RESULTS / "ccre_histsub_clusters.parquet"

STAGES = ["ESC", "DE", "HB", "iHEP", "mHEP"]
HISTONES = ["H3K4me3", "H3K27ac", "H3K27me3", "H3K9me3"]
MIN_SUB_SIZE = 400    # fold EVoC sub-clusters smaller than this into noise (readability)


def main():
    df = pl.read_parquet(CLUST_PATH)
    n = df.height
    cl_arr = df["cluster"].to_numpy()
    hist = np.column_stack([df[f"log2fc_{m}.{s}"].to_numpy() for m in HISTONES for s in STAGES])
    histz = ((hist - hist.mean(0)) / hist.std(0)).astype(np.float32)   # column z (magnitude-preserving)

    def block(a, mark):
        """The 5 stage columns belonging to one mark."""
        j = HISTONES.index(mark) * len(STAGES)
        return a[:, j:j + len(STAGES)]

    h27 = block(hist, "H3K27ac").mean(1)                 # mean H3K27ac (ordering key, cl 4-8)
    # active -> Polycomb axis (H3K27me3 - H3K27ac, z-scored) for the early cl 1-3
    biv = block(histz, "H3K27me3").mean(1) - block(histz, "H3K27ac").mean(1)

    n_clusters = int(cl_arr.max())
    row_order, sub_lab = [], []
    print(f"{'ATAC cl':>8} {'n':>7} {'#sub':>5} {'noise%':>7}")
    for cl in range(1, n_clusters + 1):
        idx = np.where(cl_arr == cl)[0]
        lab = evoc.EVoC(random_state=0).fit_predict(histz[idx])
        # fold tiny sub-clusters into noise (-1) for readability
        for s in np.unique(lab):
            if s != -1 and (lab == s).sum() < MIN_SUB_SIZE:
                lab[lab == s] = -1
        # within-archetype ordering key: active->Polycomb (H3K27me3 - H3K27ac,
        # ascending) for the early clusters 1-3; descending H3K27ac for 4-8.
        okey, sgn = (biv, 1) if cl <= 3 else (h27, -1)
        real = [s for s in np.unique(lab) if s != -1]
        real.sort(key=lambda s: sgn * okey[idx[lab == s]].mean())   # order sub-clusters
        for si, s in enumerate(real):
            sidx = idx[lab == s]
            sidx = sidx[np.argsort(sgn * okey[sidx])]     # rows within sub-cluster
            row_order.extend(sidx.tolist()); sub_lab.extend([si] * len(sidx))
        if (lab == -1).any():
            sidx = idx[lab == -1]; sidx = sidx[np.argsort(sgn * okey[sidx])]
            row_order.extend(sidx.tolist()); sub_lab.extend([99] * len(sidx))
        print(f"{cl:>8} {len(idx):>7,} {len(real):>5} {float((lab == -1).mean()):>6.0%}")

    row_order = np.array(row_order)
    cl_s = cl_arr[row_order]; sub_s = np.array(sub_lab)

    (df[row_order]
     .with_columns(atac_cluster=pl.Series(cl_s), hist_subcluster=pl.Series(sub_s))
     .write_parquet(OUT_CLUST))
    print(f"-> {OUT_CLUST}")


if __name__ == "__main__":
    main()
