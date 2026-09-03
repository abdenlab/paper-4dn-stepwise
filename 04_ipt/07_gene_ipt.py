# /// script
# requires-python = ">=3.10"
# dependencies = ["polars", "numpy", "pyarrow", "oxbow"]
# ///
"""Assign each gene's promoter (TSS) to its 50-kb IPT class and join RNA results.

Reads : data/genes.gtf.gz
        results/hepdiff.jointly_pca.norm.kmeans_10_7.labeled.tsv
        ../01_rna/results/gene_vst.parquet           VST, 2 reps x 5 stages
        ../01_rna/results/gene_tpm.parquet           TPM, 2 reps x 5 stages
        ../01_rna/results/de/de_5stage.parquet       long DE table; pivoted to one lfc_ column per sequential contrast
        ../01_rna/results/de/lrt_5stage.parquet      across-stage omnibus LRT (padj)

All RNA tables are the 5-stage series (unsuffixed names in 01_rna/results).

Writes: results/gene_ipt_expression.tsv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import oxbow
import polars as pl

DIR = Path(__file__).resolve().parent
GTF = DIR / "data" / "genes.gtf.gz"
RESULTS = DIR / "results"
RNA = DIR.parent / "01_rna" / "results"
LABELS = RESULTS / "hepdiff.jointly_pca.norm.kmeans_10_7.labeled.tsv"
OUT = RESULTS / "gene_ipt_expression.tsv"
STAGES = ["ESC", "DE", "HB", "iHEP", "mHEP"]
LFC_COLS = ["DE_vs_ESC", "HB_vs_DE", "iHEP_vs_HB", "mHEP_vs_iHEP"]
BINSIZE = 50_000
ORDER = ["A1", "A2", "A3", "B1", "Quies", "Inactive", "B4"]


def gene_tss() -> pl.DataFrame:
    g = oxbow.from_gtf(str(GTF), compression="gzip").with_attributes().pl().unnest("attributes")
    g = (g.filter(pl.col("type").is_in(["exon", "transcript", "CDS"]))
          .rename({"gene_name": "gene"}).drop_nulls("gene"))
    agg = g.group_by("gene").agg(
        pl.col("seqid").first().alias("chrom"), pl.col("strand").first(),
        pl.col("start").min().alias("gmin"), pl.col("end").max().alias("gmax"))
    return agg.with_columns(
        pl.when(pl.col("strand") == "+").then(pl.col("gmin")).otherwise(pl.col("gmax")).alias("tss")
    ).select("gene", "chrom", "tss", "strand")


def mean_reps(path, stages):
    d = pl.read_parquet(path)
    out = d.select(pl.col("gene_name").alias("gene"))
    for s in stages:
        out = out.with_columns(((d[f"{s}_REP1"] + d[f"{s}_REP2"]) / 2).alias(s))
    return out


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    for pth in (GTF, LABELS, RNA / "gene_vst.parquet", RNA / "gene_tpm.parquet",
                RNA / "de" / "de_5stage.parquet", RNA / "de" / "lrt_5stage.parquet"):
        if not pth.exists():
            raise SystemExit(f"missing {pth} — run 02_ipt_tracks.py and 01_rna first")

    genes = gene_tss().with_columns((pl.col("tss") // BINSIZE * BINSIZE).alias("bin_start"))
    labels = (pl.read_csv(LABELS, separator="\t")[["chrom", "start", "name"]]
              .rename({"start": "bin_start", "name": "cluster"}))
    genes = genes.join(labels, on=["chrom", "bin_start"], how="left")

    uniq = lambda d: d.unique(subset="gene", keep="first")
    vst = uniq(mean_reps(RNA / "gene_vst.parquet", STAGES).rename({s: f"vst_{s}" for s in STAGES}))
    tpm = uniq(mean_reps(RNA / "gene_tpm.parquet", STAGES).rename({s: f"tpm_{s}" for s in STAGES}))

    # de_5stage is long (gene x contrast) -> pivot to one lfc_ column per contrast.
    # Pivot on gene_id (unique) and carry gene_name along, then dedup by name.
    de5 = pl.read_parquet(RNA / "de" / "de_5stage.parquet")
    missing = set(LFC_COLS) - set(de5["contrast"].unique().to_list())
    if missing:
        raise SystemExit(f"de_5stage.parquet is missing contrasts: {sorted(missing)}")
    lfc = uniq(
        de5.filter(pl.col("contrast").is_in(LFC_COLS))
           .select("gene_id", "gene_name", "contrast", "log2FoldChange")
           .pivot(values="log2FoldChange", index=["gene_id", "gene_name"], on="contrast")
           .select(pl.col("gene_name").alias("gene"),
                   *[pl.col(c).alias(f"lfc_{c}") for c in LFC_COLS]))
    # lrt_5stage already carries gene_name, so no name map is needed.
    lrt = uniq(pl.read_parquet(RNA / "de" / "lrt_5stage.parquet")
               .select(pl.col("gene_name").alias("gene"), pl.col("padj").alias("lrt_padj")))

    df = (genes.join(vst, on="gene", how="inner")
               .join(tpm, on="gene", how="left")
               .join(lfc, on="gene", how="left")
               .join(lrt, on="gene", how="left"))
    df.write_csv(OUT, separator="\t")
    print(f"-> {OUT}  ({df.height} genes with TSS + VST)")

    sub = df.filter(pl.col("cluster").is_in(ORDER))
    print("\nper-cluster mean VST trajectory (canonical):")
    print(f"  {'cluster':9s}{'n':>6}  " + "  ".join(f"{s:>6}" for s in STAGES)
          + f"   {'HBvsDE':>7}{'%down':>8}")
    for cl in ORDER:
        d = sub.filter(pl.col("cluster") == cl)
        traj = [d[f"vst_{s}"].mean() for s in STAGES]
        hb = d["lfc_HB_vs_DE"].drop_nulls()
        pct = 100 * (hb < 0).mean() if hb.len() else float("nan")
        print(f"  {cl:9s}{d.height:>6}  " + "  ".join(f"{t:6.2f}" for t in traj)
              + f"   {hb.mean():7.2f}{pct:7.0f}%")


if __name__ == "__main__":
    main()
