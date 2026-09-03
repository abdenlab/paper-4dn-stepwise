"""Aggregate RNA-seq into per-gene features for Phase 2 driver scoring.

Inputs (from 01_rna):
  ../01_rna/results/gene_tpm.parquet   -- gene_id, gene_name, one column per 5-stage
                                         library (<STAGE>_REP{1,2}).
  ../01_rna/results/de/de_5stage.parquet -- LONG DESeq2 table: gene_id, gene_name,
                                         baseMean, log2FoldChange, lfcSE, stat,
                                         pvalue, padj, contrast. Pivoted to one
                                         lfc_ column per contrast below,

Outputs (under data/):
  gene_features.parquet
      per-gene table with: stage TPMs (5 columns), tau, max_stage,
      log2FC per contrast (4 adjacent + 2 macro).
  gene_features.meta.json
      stage list, contrast list, tau formula, filter thresholds.

The tau (Yanai 2005) stage-specificity score:
    tau = sum( (1 - x_i / x_max) ) / (n - 1)
where x_i is the (log-1p TPM) at stage i, x_max the max, n=5.
We use log1p(TPM) rather than raw TPM because tau is dominated by
absolute scale and one outlier stage otherwise.

Conventions used downstream:
  - "Active stage" of (contrast=X_vs_Y, direction=up)   is X
  - "Active stage" of (contrast=X_vs_Y, direction=down) is Y
  - Direction match: sign(log2FC of TF, in that contrast)
                     == +1 (for up) or -1 (for down)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

DIR = Path(__file__).resolve().parent
RNA_OUT = DIR.parent / "01_rna" / "results"
OUT = DIR / "data"
OUT.mkdir(parents=True, exist_ok=True)

STAGES = ["ESC", "DE", "HB", "iHEP", "mHEP"]
CONTRASTS = [
    "DE_vs_ESC",
    "HB_vs_DE",
    "iHEP_vs_HB",
    "mHEP_vs_iHEP",
    "hepatic_vs_prehepatic",
    "maturation_vs_prematuration",
]
EXPRESSION_FLOOR_TPM = 5.0


def per_stage_mean_tpm(tpm: pl.DataFrame) -> pl.DataFrame:
    """Collapse REP1/REP2 columns into one per-stage mean column.

    Input columns expected: gene_id, gene_name, <STAGE>_REP1, <STAGE>_REP2 for
    each stage.
    """
    aggs = []
    for s in STAGES:
        cols = [c for c in tpm.columns if c.startswith(f"{s}_REP")]
        if not cols:
            raise ValueError(f"No replicate columns found for stage {s}")
        aggs.append(pl.mean_horizontal(*[pl.col(c) for c in cols]).alias(f"tpm_{s}"))
    return tpm.select("gene_id", "gene_name", *aggs)


def tau(values: np.ndarray) -> float:
    """Yanai stage-specificity tau on a 1D vector. Computed on log1p(TPM)."""
    v = np.log1p(values)
    vmax = v.max()
    if vmax <= 0:
        return np.nan
    return float(((1.0 - v / vmax).sum()) / (len(v) - 1))


def main():
    tpm_raw = pl.read_parquet(RNA_OUT / "gene_tpm.parquet")
    de = pl.read_parquet(RNA_OUT / "de" / "de_5stage.parquet")

    tpm = per_stage_mean_tpm(tpm_raw)

    # tau on per-stage means (log1p).
    tpm_mat = tpm.select([f"tpm_{s}" for s in STAGES]).to_numpy()
    tpm_log = np.log1p(tpm_mat)
    vmax = tpm_log.max(axis=1)
    safe_vmax = np.where(vmax > 0, vmax, np.nan)
    tau_vec = (1.0 - tpm_log / safe_vmax[:, None]).sum(axis=1) / (len(STAGES) - 1)

    max_idx = tpm_log.argmax(axis=1)
    max_stage = np.array(STAGES)[max_idx]

    tpm = tpm.with_columns(
        tau=pl.Series(tau_vec, dtype=pl.Float64),
        max_stage=pl.Series(max_stage, dtype=pl.Utf8),
    )

    # de_5stage is long (one row per gene x contrast) -> pivot to one lfc_ column
    # per contrast. 13_de_contrasts.R emits apeglm-shrunken LFCs for the four
    # sequential contrasts and unshrunken LFCs for the two combined ones.
    missing = set(CONTRASTS) - set(de["contrast"].unique().to_list())
    if missing:
        raise ValueError(f"de_5stage.parquet is missing contrasts: {sorted(missing)}")
    lfc = (
        de.filter(pl.col("contrast").is_in(CONTRASTS))
        .select("gene_id", "contrast", "log2FoldChange")
        .pivot(values="log2FoldChange", index="gene_id", on="contrast")
        .rename({c: f"lfc_{c}" for c in CONTRASTS})
    )
    merged = tpm.join(lfc, on="gene_id", how="left")

    out_pq = OUT / "gene_features.parquet"
    merged.write_parquet(out_pq)
    print(f"[gene features] {out_pq}  rows={merged.height}")

    meta = {
        "stages": STAGES,
        "contrasts": CONTRASTS,
        "tau_formula": "sum(1 - log1p(tpm_i)/max(log1p(tpm))) / (n_stages - 1)",
        "expression_floor_tpm": EXPRESSION_FLOOR_TPM,
        "tpm_columns": [f"tpm_{s}" for s in STAGES],
        "lfc_columns": [f"lfc_{c}" for c in CONTRASTS],
        "max_stage_column": "max_stage",
        "tau_column": "tau",
    }
    (OUT / "gene_features.meta.json").write_text(json.dumps(meta, indent=2))

    # Print a quick sanity panel — the TFs we'll explicitly validate later.
    sanity_genes = [
        "POU5F1", "GATA4", "HNF4A", "HNF1A", "HNF1B",
        "CTCF", "CTCFL",
        "BACH1", "BACH2", "NFE2L2",
        "JUN", "FOS", "FOSL1", "FOSL2",
        "AR", "NR3C1", "NR1H3", "NR1H4",
    ]
    cols = ["gene_name", "max_stage", "tau", *[f"tpm_{s}" for s in STAGES]]
    sub = merged.filter(pl.col("gene_name").is_in(sanity_genes)).select(cols)
    print()
    with pl.Config(tbl_rows=30, fmt_str_lengths=20, tbl_cols=10, float_precision=2):
        print(sub.sort("gene_name"))


if __name__ == "__main__":
    main()
