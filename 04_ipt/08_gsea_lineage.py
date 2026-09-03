# /// script
# requires-python = ">=3.10"
# dependencies = ["polars", "numpy", "pandas", "gseapy", "pyarrow"]
# ///
"""Within-IPT lineage GSEA (PanglaoDB, HB_vs_DE).

Within each IPT class, rank ITS genes by the HB_vs_DE Wald statistic and run
preranked GSEA against PanglaoDB cell-type markers. Negative NES = that lineage's
genes are coordinately DOWN at HB within the class, i.e. lineage restriction.

The PanglaoDB library is fetched live.

Reads : results/gene_ipt_expression.tsv
        ../01_rna/results/de/de_5stage.parquet      Wald `stat`, filtered to HB_vs_DE
Writes: results/gsea_lineage_<IPT>__PanglaoDB_Augmented_2021__HB_vs_DE.tsv
"""
from __future__ import annotations

from pathlib import Path

import gseapy as gp
import pandas as pd
import polars as pl

DIR = Path(__file__).resolve().parent
RESULTS = DIR / "results"
RNA = DIR.parent / "01_rna" / "results"
CLUSTERS = ["A1", "A2", "A3", "B1", "Quies", "Inactive", "B4"]
CONTRAST = "HB_vs_DE"
LIB = "PanglaoDB_Augmented_2021"
PRERANK_KW = dict(min_size=5, max_size=1000, permutation_num=1000, seed=42, verbose=False)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    expr = RESULTS / "gene_ipt_expression.tsv"
    de5 = RNA / "de" / "de_5stage.parquet"
    for pth in (expr, de5):
        if not pth.exists():
            raise SystemExit(f"missing {pth} — run 07_gene_ipt.py and 01_rna first")

    df = pl.read_csv(expr, separator="\t")
    gmt = gp.get_library(name=LIB, organism="Human")
    print(f"PanglaoDB: {len(gmt)} sets")
    # de_5stage is long over contrasts; take the Wald stat for this one.
    stat = (pl.read_parquet(de5).filter(pl.col("contrast") == CONTRAST)
              .select(pl.col("gene_name").alias("gene"), "stat"))
    cdf = df.join(stat, on="gene", how="left")
    for cl in CLUSTERS:
        d = cdf.filter((pl.col("cluster") == cl) & pl.col("stat").is_not_null())
        rnk = pd.Series(d["stat"].to_numpy(), index=d["gene"].to_list())
        rnk = rnk[~rnk.index.duplicated()].sort_values(ascending=False)
        res = gp.prerank(rnk=rnk, gene_sets=gmt, outdir=None, **PRERANK_KW).res2d
        r = pl.from_pandas(res).with_columns(
            pl.col("NES").cast(float), pl.col("FDR q-val").cast(float))
        r.write_csv(RESULTS / f"gsea_lineage_{cl}__{LIB}__{CONTRAST}.tsv", separator="\t")
        nup = r.filter((pl.col("NES") > 0) & (pl.col("FDR q-val") < 0.05)).height
        ndn = r.filter((pl.col("NES") < 0) & (pl.col("FDR q-val") < 0.05)).height
        print(f"  {cl:9s} {len(rnk):5d} genes: {nup:3d} UP / {ndn:3d} DOWN  (FDR<0.05)")


if __name__ == "__main__":
    main()
