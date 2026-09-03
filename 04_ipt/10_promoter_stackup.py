# /// script
# requires-python = ">=3.10"
# dependencies = ["polars", "numpy", "pybigtools", "pyarrow", "scikit-learn", "oxbow"]
# ///
"""Joint H3K27me3 + H3K27ac stackup at T.pcg gene promoters, clustered jointly.

Takes every expressed gene whose TSS falls in a T.pcg (B1) 50-kb bin, builds a
+/-5 kb signal matrix for both marks across the 5 stages, and k-means clusters
the elements on the centre signal of both marks at once. Rows are ordered by
cluster (descending H3K27me3 at HB) and, within a cluster, by descending mean
H3K27me3 — so the figure reads as a Polycomb gradient.

Reads : data/genes.gtf.gz
        data/{cutnrun,chip}/
        results/hepdiff.jointly_pca.norm.kmeans_10_7.labeled.tsv
        ../01_rna/results/gene_vst.parquet        expressed-gene set
        ../01_rna/results/de/de_5stage.parquet    HB_vs_DE LFC for the row annotation
Writes: results/promoter_joint_cluster_{stackup,metaprofiles}.npz
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import oxbow
import polars as pl

import stackup as S

DIR = Path(__file__).resolve().parent
GENCODE_GTF = str(DIR / "data" / "genes.gtf.gz")
LABELS = DIR / "results" / "hepdiff.jointly_pca.norm.kmeans_10_7.labeled.tsv"
RNA = DIR.parent / "01_rna" / "results"
TAG = "promoter_joint_cluster"
ME3_VMAX = 25.0          # promoters reach much higher H3K27me3 than bulk cCREs


def gencode_tss() -> pl.DataFrame:
    """Per-gene TSS from GENCODE v50 gene rows: + strand -> start, - strand -> end.
    oxbow only auto-detects BGZF; this GTF is plain gzip, so pass compression='gzip'
    explicitly (oxbow also 0-bases start)."""
    g = (oxbow.from_gtf(GENCODE_GTF, compression="gzip").with_attributes().pl()
         .unnest("attributes"))
    g = (g.filter(pl.col("type") == "gene")
          .rename({"gene_name": "gene", "seqid": "chrom"}).drop_nulls("gene"))
    g = g.with_columns(
        pl.when(pl.col("strand") == "+").then(pl.col("start"))
          .otherwise(pl.col("end")).alias("tss"))
    return g.select("gene", "chrom", "tss", "strand").unique(subset="gene", keep="first")


def main() -> None:
    for pth in (LABELS, RNA / "gene_vst.parquet", RNA / "de" / "de_5stage.parquet"):
        if not pth.exists():
            raise SystemExit(f"missing {pth} — run 02_ipt_tracks.py and 01_rna first")

    genes = gencode_tss().with_columns(
        (pl.col("tss") // S.BINSIZE * S.BINSIZE).alias("bin_start"))
    labels = (pl.read_csv(LABELS, separator="\t")[["chrom", "start", "name"]]
              .rename({"start": "bin_start", "name": "cluster"}))
    genes = genes.join(labels, on=["chrom", "bin_start"], how="left")

    lfc = (pl.read_parquet(RNA / "de" / "de_5stage.parquet")
             .filter(pl.col("contrast") == "HB_vs_DE")
             .select(pl.col("gene_name").alias("gene"),
                     pl.col("log2FoldChange").alias("lfc_HB_vs_DE"))
             .unique(subset="gene", keep="first"))
    expressed = (pl.read_parquet(RNA / "gene_vst.parquet")
                   .select(pl.col("gene_name").alias("gene")).unique())

    g = (genes.filter(pl.col("cluster") == "B1")
              .join(expressed, on="gene", how="inner")
              .join(lfc, on="gene", how="left"))
    chrom = g["chrom"].to_list(); tss = g["tss"].to_list(); strand = g["strand"].to_list()
    lfc_arr = g["lfc_HB_vs_DE"].to_numpy()
    print(f"{g.height} T.pcg promoters (GENCODE v50 TSS)")

    mats = S.build_matrices(chrom, tss, strand=strand)
    keep, mats, lab_k, order, bounds, seq, krank = S.cluster_and_order(mats, clip=True)
    lfc_arr = lfc_arr[keep]

    S.dump(TAG, mats, lab_k, order, bounds, seq, ME3_VMAX, lfc_ord=lfc_arr[order])
    S.summarize(mats, lab_k, seq, krank,
                extra=("meanLFC(HB/DE)", lambda m: f"{np.nanmean(lfc_arr[m]):+.2f}"))


if __name__ == "__main__":
    main()
