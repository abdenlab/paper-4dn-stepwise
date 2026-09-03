# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "polars",
#     "numpy",
#     "pandas",
#     "gseapy",
# ]
# ///
"""Preranked GSEA of every 5-stage contrast against KEGG.

Contrasts:
  - 4 sequential: DE_vs_ESC, HB_vs_DE, iHEP_vs_HB, mHEP_vs_iHEP
  - 2 combined: hepatic_vs_prehepatic, maturation_vs_prematuration

Genes are ranked by the DESeq2 Wald statistic (= LFC / SE), which incorporates
both effect size and precision.

The library name is resolved from Enrichr's LIVE catalogue rather than hardcoded,
because it is version-stamped and drifts (KEGG_2021_Human -> KEGG_2026); the
newest human KEGG library wins.

Reads  results/de/de_5stage.parquet
Writes results/gsea_results/<contrast>__<library>.pq   per-contrast, resumable
       results/gsea_all_results.pq                     combined table
"""
import re
import warnings
from pathlib import Path

import gseapy as gp
import numpy as np
import pandas as pd
import polars as pl

warnings.filterwarnings("ignore")

DIR = Path(__file__).resolve().parent
DE_FILE = DIR / "results" / "de" / "de_5stage.parquet"
GSEA_OUTDIR = DIR / "results"
RESULTS_DIR = GSEA_OUTDIR / "gsea_results"

CONTRASTS = [
    "DE_vs_ESC", "HB_vs_DE", "iHEP_vs_HB", "mHEP_vs_iHEP",
    "hepatic_vs_prehepatic", "maturation_vs_prematuration",
]
NON_HUMAN = ("Mouse", "mouse", "Yeast", "yeast", "Worm", "worm", "Fly", "fly",
             "Fish", "fish", "Drosophila", "C_elegans", "E_coli")
PRERANK_KW = dict(outdir=None, min_size=5, max_size=1000,
                  permutation_num=1000, seed=42, verbose=False)


def kegg_library() -> str:
    """Newest human KEGG library in Enrichr's live catalogue."""
    all_libs = gp.get_library_name()
    kegg = [l for l in all_libs
            if re.match(r"KEGG_\d{4}", l) and not any(kw in l for kw in NON_HUMAN)]
    if not kegg:
        raise SystemExit("Enrichr serves no KEGG_<year> human library "
                         f"(catalogue has {len(all_libs)} entries)")
    lib = max(kegg, key=lambda l: int(re.search(r"KEGG_(\d{4})", l).group(1)))
    print(f"KEGG library: {lib}  (candidates: {sorted(kegg)})")
    return lib


def rankings(de: pl.DataFrame) -> dict[str, pd.Series]:
    """Wald-statistic ranking per contrast, keyed on gene symbol."""
    have = set(de.get_column("contrast").unique().to_list())
    if missing := [c for c in CONTRASTS if c not in have]:
        raise SystemExit(f"contrasts absent from {DE_FILE.name}: {missing} "
                         f"(have {sorted(have)})")
    out = {}
    for label in CONTRASTS:
        res = de.filter(pl.col("contrast") == label)
        genes = np.array(res.get_column("gene_name").to_list())
        stats = res.get_column("stat").to_numpy()
        valid = ~np.isnan(stats)
        rnk = pd.Series(stats[valid], index=genes[valid])
        rnk = rnk[~rnk.index.duplicated(keep="first")].sort_values(ascending=False)
        out[label] = rnk
        print(f"{label}: {len(rnk)} genes ranked")
    return out


def main() -> None:
    if not DE_FILE.exists():
        raise SystemExit(f"missing {DE_FILE} — run 13_de_contrasts.R first")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    de = pl.read_parquet(DE_FILE)
    rnks = rankings(de)
    lib = kegg_library()

    for contrast, rnk in rnks.items():
        out_file = RESULTS_DIR / f"{contrast}__{lib}.pq"
        if out_file.exists():
            print(f"  {contrast}: cached")
            continue
        res_df = gp.prerank(rnk=rnk, gene_sets=lib, **PRERANK_KW).res2d
        if res_df.empty:
            raise SystemExit(f"{contrast}: GSEA returned no terms for {lib}")
        res_df = res_df.copy()
        res_df["contrast"] = contrast
        res_df["library"] = lib
        pl.from_pandas(res_df).write_parquet(out_file)
        n_sig = (res_df["FDR q-val"].astype(float) < 0.05).sum()
        print(f"  {contrast}: {len(res_df)} terms, {n_sig} significant (FDR < 0.05)")

    combined = pl.concat([pl.read_parquet(f) for f in sorted(RESULTS_DIR.glob("*.pq"))])
    combined.write_parquet(GSEA_OUTDIR / "gsea_all_results.pq")
    sig = combined.filter(pl.col("FDR q-val").cast(float) < 0.05)
    print(f"\n-> gsea_all_results.pq: {combined.height} rows, "
          f"{sig.height} significant (FDR < 0.05)")
    for contrast in CONTRASTS:
        print(f"  {contrast}: {sig.filter(pl.col('contrast') == contrast).height}")


if __name__ == "__main__":
    main()
