# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas", "numpy", "bioframe", "pyarrow"]
# ///
"""Collapse the k=10 K-means clustering into the 7 named IPT classes.

Reads : results/hepdiff.jointly_pca.norm.kmeans.pq
Writes: results/hepdiff.jointly_pca.norm.kmeans_10_7.labeled.tsv
        results/hepdiff.jointly_pca.norm.kmeans_10_7.labeled.dense.bed
"""
from __future__ import annotations

from pathlib import Path

import bioframe
import numpy as np
import pandas as pd

DIR = Path(__file__).resolve().parent
IN_PQ = DIR / "results" / "hepdiff.jointly_pca.norm.kmeans.pq"
RESULTS = DIR / "results"

CLUSTER_COL = "kmeans_10"
N_LABELS = 7

apple_vintage = {
    'green': '#5ebd3e',
    'yellow': '#ffb900',
    'orange': '#f78200',
    'red': '#e23838',
    'violet': '#973999',
    'blue': '#009cdf',
}

# k=10 cluster id -> canonical class. A1/B1/B4 each absorb two clusters.
cluster_labels = {
    0: "Inactive",
    1: "Quies",
    2: "A3",
    3: "B4",
    4: "B1",
    5: "A2",
    6: "B1",
    7: "B4",
    8: "A1",
    9: "A1",
}
colors = {
    "B4": apple_vintage['violet'],
    "Inactive": "darkgray",
    "Quies": "lightsteelblue",
    "B1": apple_vintage['blue'],
    "A3": apple_vintage['yellow'],
    "A2": apple_vintage['orange'],
    "A1": apple_vintage['red'],
    np.nan: "black",
}


def main() -> None:
    if not IN_PQ.exists():
        raise SystemExit(f"missing {IN_PQ} — run 01_ipt_clustering.py first")
    RESULTS.mkdir(parents=True, exist_ok=True)
    ipt = pd.read_parquet(IN_PQ)

    df = ipt[["chrom", "start", "end", CLUSTER_COL]].copy()
    df['name'] = df[CLUSTER_COL].map(cluster_labels)
    df["color"] = df["name"].map(colors)
    df["itemRgb"] = df["color"].apply(bioframe.to_ucsc_colorstring)

    n_named = int(df["name"].notna().sum())
    print(f"{n_named:,} of {len(df):,} bins labelled "
          f"({len(df) - n_named:,} unassigned) -> {df['name'].nunique()} classes")

    stem = f"hepdiff.jointly_pca.norm.{CLUSTER_COL}_{N_LABELS}.labeled"

    df_dense = bioframe.merge_runs(df.dropna(subset=["name"]), "name",
                                   agg={"itemRgb": ("itemRgb", "first")})
    bioframe.to_bed(df_dense, RESULTS / f"{stem}.dense.bed", schema='bed9')
    print(f"-> {RESULTS / f'{stem}.dense.bed'}  ({len(df_dense):,} runs)")

    df.to_csv(RESULTS / f"{stem}.tsv", index=False, header=True, sep="\t",
              columns=["chrom", "start", "end", "name", "color", "itemRgb"])
    print(f"-> {RESULTS / f'{stem}.tsv'}  ({len(df):,} bins)")


if __name__ == "__main__":
    main()
