# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas", "numpy", "bioframe", "scikit-learn", "pyarrow"]
# ///
"""K-means clustering of the jointly-PCA score tracks -> IPT bin classes.

K-means over a sweep of k, each relabeled and row-ordered by median GC,
then absolute distance from the centromere.

Reads : data/jointly_pca.parquet   (PCA embedding + mask)
        data/hg38.bins.gc.50000.pq (GC / centel / armlen)
Writes: results/hepdiff.jointly_pca.norm.kmeans.pq
          chrom, start, end + per-k `kmeans_<k>` and `kmeans_<k>_order` columns.
"""
from __future__ import annotations

from pathlib import Path

import bioframe
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

DIR = Path(__file__).resolve().parent
PCA_PQ = DIR / "data" / "jointly_pca.parquet"
GC_PQ = DIR / "data" / "hg38.bins.gc.50000.pq"
OUT = DIR / "results" / "hepdiff.jointly_pca.norm.kmeans.pq"

BINSIZE = 50_000
STAGES = ["ESC", "DE", "HB", "iHLC", "HLC"]   # column suffixes in the GEO file
N_COMPONENTS = 6
N_CLUSTERS_LIST = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 32, 64]
CLUSTER_SORT_KEY = "GC"


def relabel_clusters(labels, n_clusters, sorting_tracks, sort_key):
    """Re-order the bins and re-label the cluster IDs based on a set of
    sorting tracks.

    1. User-defined sorting key.
    2. Absolute distance from centromere.
    3. Length of corresponding chromosome arm.

    `sorting_tracks` must be row-aligned with the clustered frame (same bins in
    the same order) -- `labels` is assigned positionally.
    """
    # Assign the cluster IDs and extra data to temporary dataframe
    df = sorting_tracks[['chrom', 'start', 'end', sort_key, 'centel', 'armlen']].copy()
    df['centel_abs'] = df['centel'] * df['armlen']
    df['cluster'] = labels

    # Relabel the clusters using median of sorting column. The unassigned
    # sentinel (== n_clusters) is forced to +inf so it always sorts last.
    df.loc[df['cluster'] == n_clusters, sort_key] = np.inf
    clusters_ordered = (
        df
        .groupby('cluster')
        [sort_key]
        .median()
        .sort_values()
        .index
        .tolist()
    )
    cluster_dtype = pd.CategoricalDtype(clusters_ordered, ordered=True)
    df['cluster_relabeled'] = df['cluster'].astype(cluster_dtype).cat.codes

    # Reorder the bins for plotting
    bin_ranks = (
        df
        .sort_values(
            ['cluster_relabeled', 'centel_abs'],
            ascending=[True, True]
        )
        .index
        .values
    )
    return df['cluster_relabeled'].values, bin_ranks


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    chromsizes = bioframe.fetch_chromsizes("hg38")
    chromosomes = list(chromsizes[:'chr22'].index)

    sorting_tracks = pd.read_parquet(GC_PQ)
    sorting_tracks = sorting_tracks[sorting_tracks['chrom'].isin(chromosomes)]

    eigvecs = pd.read_parquet(PCA_PQ)
    eigvecs = eigvecs[eigvecs['chrom'].isin(chromosomes)]
    if len(eigvecs) != len(sorting_tracks):
        raise SystemExit(f"bin grids differ: {len(eigvecs)} PCA vs "
                         f"{len(sorting_tracks)} GC rows — relabel_clusters "
                         f"aligns them positionally")
    if "mask" not in eigvecs.columns:
        raise SystemExit(f"{PCA_PQ.name} has no `mask` column")
    all_pc_cols = [name for name in eigvecs.columns if name.startswith("PCA")]
    excluded = ~eigvecs["mask"].to_numpy(bool)
    eigvecs.loc[excluded, all_pc_cols] = np.nan
    print(f"masked {int(excluded.sum()):,} of {len(eigvecs):,} bins "
          f"across {len(all_pc_cols)} PCA columns")

    # Per-stage L2 renormalization onto the overall norm.
    pc_cols = [f"PCA{i}_{stage}" for stage in STAGES for i in range(1, N_COMPONENTS + 1)]
    finite = eigvecs.loc[:, pc_cols].notnull().all(axis=1)
    overall_norm = np.linalg.norm(eigvecs.loc[finite, pc_cols])
    for stage in STAGES:
        pc_cols_stage = [f"PCA{i}_{stage}" for i in range(1, N_COMPONENTS + 1)]
        norm = np.linalg.norm(eigvecs.loc[finite, pc_cols_stage])
        eigvecs.loc[:, pc_cols_stage] = eigvecs.loc[:, pc_cols_stage] / norm * overall_norm

    print(f"clustering {int(finite.sum()):,} of {len(eigvecs):,} bins on "
          f"{N_COMPONENTS} components x {len(STAGES)} stages")

    out = eigvecs[['chrom', 'start', 'end']].copy()
    X = eigvecs.loc[:, pc_cols].values
    keep = np.all(~np.isnan(X), axis=1)
    x = X[keep, :]

    for n_clusters in N_CLUSTERS_LIST:
        colname = f'kmeans_{n_clusters}'
        print(colname)

        model = KMeans(
            n_clusters=n_clusters,
            init='k-means++',
            n_init=100,
            max_iter=10000,
            tol=0.00001,
            random_state=42,
            copy_x=True,
            verbose=0,
        )

        labels = np.full(len(keep), n_clusters)
        labels[keep] = model.fit_predict(x)
        new_labels, bin_ranks = relabel_clusters(
            labels, n_clusters, sorting_tracks, CLUSTER_SORT_KEY
        )

        out[colname] = new_labels
        out[colname + '_order'] = bin_ranks

    out.to_parquet(OUT, index=False)
    print(f"-> {OUT}  {out.shape}")


if __name__ == "__main__":
    main()
