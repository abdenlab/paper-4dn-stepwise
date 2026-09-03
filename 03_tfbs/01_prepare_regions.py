"""Prepare foreground and background region sets for pycisTarget enrichment.

Foreground: 4 adjacent contrasts x {up, down} = 8 sets
Source : da_peak_set.parquet  (padj<0.05, |log2FC|>1, stage-aware sa>1)
Filter : direction = sign(log2FC). Already class-filtered upstream.

Background: peak-supported cCRE universe (369,321 cCREs)
Source : data/ccre_universe.parquet

Outputs (BED 3-column, sorted, deduped):
    data/regions/fg_<contrast>_<direction>.bed
    data/regions/bg_universe.bed
    data/regions/region_set_manifest.parquet  -- per-set counts + cCRE class breakdown
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

DIR = Path(__file__).resolve().parent
ATAC_RESULTS = DIR.parent / "02_atac" / "results"
UNIVERSE = DIR / "data" / "ccre_universe.parquet"
OUT = DIR / "data" / "regions"
OUT.mkdir(parents=True, exist_ok=True)

CONTRASTS = ["DE_vs_ESC", "HB_vs_DE", "iHEP_vs_HB", "mHEP_vs_iHEP"]


def write_bed(df: pl.DataFrame, path: Path) -> int:
    """Write 3-column BED, sorted by chrom/start, deduped."""
    bed = (
        df.select(["chrom", "start", "end"])
        .unique()
        .sort(["chrom", "start", "end"])
    )
    bed.write_csv(path, separator="\t", include_header=False)
    return bed.height


def main():
    da = pl.read_parquet(ATAC_RESULTS / "da_peak_set.parquet")
    uni = pl.read_parquet(UNIVERSE)

    print(f"DA peak set : {da.height:>9} rows")
    print(f"Universe    : {uni.height:>9} cCREs")
    print()

    # Background universe
    bg_path = OUT / "bg_universe.bed"
    n_bg = write_bed(uni, bg_path)
    print(f"[bg]  bg_universe              n={n_bg:>7}  -> {bg_path.name}")

    manifest_rows = [
        {
            "set": "bg_universe",
            "kind": "background",
            "contrast": None,
            "direction": None,
            "n_regions": n_bg,
            "path": str(bg_path),
        }
    ]

    # Foreground sets
    for contrast in CONTRASTS:
        for direction, expr in [("up", pl.col("log2FC") > 0), ("down", pl.col("log2FC") < 0)]:
            sub = da.filter((pl.col("contrast") == contrast) & expr)
            name = f"fg_{contrast}_{direction}"
            path = OUT / f"{name}.bed"
            n = write_bed(sub, path)
            print(f"[fg]  {name:<28} n={n:>7}  -> {path.name}")
            manifest_rows.append(
                {
                    "set": name,
                    "kind": "foreground",
                    "contrast": contrast,
                    "direction": direction,
                    "n_regions": n,
                    "path": str(path),
                }
            )

    manifest = pl.DataFrame(manifest_rows)
    manifest_path = OUT / "region_set_manifest.parquet"
    manifest.write_parquet(manifest_path)
    print()
    print(f"Manifest -> {manifest_path}")
    print()
    print(manifest)

    # Class breakdown per foreground set (just informational)
    print()
    print("cCRE class breakdown per foreground (% of set):")
    class_summary = (
        da.with_columns(
            direction=pl.when(pl.col("log2FC") > 0).then(pl.lit("up")).otherwise(pl.lit("down"))
        )
        .group_by(["contrast", "direction", "ccre_class"])
        .len()
        .with_columns(total=pl.col("len").sum().over(["contrast", "direction"]))
        .with_columns(pct=(100 * pl.col("len") / pl.col("total")).round(1))
        .sort(["contrast", "direction", "len"], descending=[False, False, True])
    )
    with pl.Config(tbl_rows=200, tbl_cols=10):
        print(class_summary)


if __name__ == "__main__":
    main()
