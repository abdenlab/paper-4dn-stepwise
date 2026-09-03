# /// script
# requires-python = ">=3.10"
# dependencies = ["polars", "numpy", "pandas", "gseapy", "lxml"]
# ///
"""Preranked GSEA against native MSigDB collections (not Enrichr libraries).

Fetches the real MSigDB GMTs via gseapy.Msigdb and runs gp.prerank for each
collection x contrast, using the same DESeq2 Wald-statistic rankings and
prerank parameters as 20_gsea_kegg.py.

Input:
  results/de/de_5stage.parquet

Collections (human, MSigDB 2024.1.Hs):
  H          h.all     Hallmark
  C2:CP      c2.cp     canonical pathways (BioCarta/KEGG/PID/Reactome/WikiPathways)
  C2:CGP     c2.cgp    chemical and genetic perturbation signatures
  C3:TFT     c3.tft    transcription factor targets
  C8         c8.all    cell type signature gene sets

Scope: the 4 sequential stage transitions.

Outputs:
  results/msigdb_gmt/<cat>.gmt           cached gene set definitions
  results/gsea_msigdb/<contrast>__<lib>.pq   per-run results (resumable)
  results/gsea_msigdb_results.pq         combined table (drop-in companion to
                                        gsea_all_results.pq: same columns +
                                        contrast + library)
"""
from __future__ import annotations

import warnings
from pathlib import Path

import gseapy as gp
import numpy as np
import pandas as pd
import polars as pl
from gseapy import Msigdb

warnings.filterwarnings("ignore")

DIR = Path(__file__).resolve().parent
DE_FILE = DIR / "results" / "de" / "de_5stage.parquet"
OUTDIR = DIR / "results"
GMT_DIR = OUTDIR / "msigdb_gmt"
RES_DIR = OUTDIR / "gsea_msigdb"
GMT_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

DBVER = "2024.1.Hs"
CONTRASTS = ["DE_vs_ESC", "HB_vs_DE", "iHEP_vs_HB", "mHEP_vs_iHEP"]

# (Enrichr-style library label, MSigDB category)
COLLECTIONS = [
    ("MSigDB_H", "h.all"),
    ("MSigDB_C2_CP", "c2.cp"),
    ("MSigDB_C2_CGP", "c2.cgp"),
    ("MSigDB_C3_TFT", "c3.tft"),
    ("MSigDB_C8", "c8.all"),
]

PRERANK_KW = dict(min_size=5, max_size=1000, permutation_num=1000,
                  seed=42, threads=16, verbose=False)


def load_gmt(msig, category):
    """Fetch a collection (cached on disk as a GMT) -> {set_name: [genes]}."""
    path = GMT_DIR / f"{category}.{DBVER}.gmt"
    if path.exists():
        gmt = {}
        for line in path.read_text().splitlines():
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                gmt[parts[0]] = parts[2:]
        return gmt
    gmt = msig.get_gmt(category=category, dbver=DBVER)
    with open(path, "w") as f:
        for name, genes in gmt.items():
            f.write("\t".join([name, ""] + list(genes)) + "\n")
    return gmt


def build_rankings():
    if not DE_FILE.exists():
        raise SystemExit(f"missing {DE_FILE} — run 13_de_contrasts.R first")
    de = pl.read_parquet(DE_FILE)
    have = set(de.get_column("contrast").unique().to_list())
    if missing := [c for c in CONTRASTS if c not in have]:
        raise SystemExit(
            f"contrasts absent from {DE_FILE.name}: {missing} (have {sorted(have)})")

    rankings = {}
    for label in CONTRASTS:
        res = de.filter(pl.col("contrast") == label)
        genes = res.get_column("gene_name").to_list()
        stats = res.get_column("stat").to_numpy()
        valid = ~np.isnan(stats)
        s = pd.Series(stats[valid], index=np.array(genes)[valid])
        s = s[~s.index.duplicated(keep="first")].sort_values(ascending=False)
        rankings[label] = s
        print(f"  {label}: {len(s)} genes ranked")
    return rankings


def main():
    print("Building rankings (DESeq2 Wald stat)...")
    rankings = build_rankings()

    print("Fetching MSigDB collections...")
    msig = Msigdb()
    gmts = {}
    for label, cat in COLLECTIONS:
        g = load_gmt(msig, cat)
        gmts[label] = g
        print(f"  {label:16s} ({cat}): {len(g)} sets")

    for label, _cat in COLLECTIONS:
        gmt = gmts[label]
        for contrast in CONTRASTS:
            out = RES_DIR / f"{contrast}__{label}.pq"
            if out.exists():
                print(f"skip  {contrast} x {label} (exists)")
                continue
            print(f"run   {contrast} x {label} ({len(gmt)} sets) ...", flush=True)
            pre = gp.prerank(rnk=rankings[contrast], gene_sets=gmt,
                             outdir=None, **PRERANK_KW)
            res = pre.res2d.copy()
            res["contrast"] = contrast
            res["library"] = label
            pl.from_pandas(res).write_parquet(out)
            nsig = (res["FDR q-val"].astype(float) < 0.05).sum()
            print(f"      -> {len(res)} sets, {nsig} sig (FDR<0.05)", flush=True)

    print("Combining...")
    files = sorted(RES_DIR.glob("*.pq"))
    combined = pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")
    combined.write_parquet(OUTDIR / "gsea_msigdb_results.pq")
    print(f"-> {OUTDIR / 'gsea_msigdb_results.pq'}: {combined.height} rows")


if __name__ == "__main__":
    main()
