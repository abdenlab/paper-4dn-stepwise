"""Build AME inputs: per-contrast×direction foreground FASTA + GC-matched control FASTA.

Inputs (all written by 01_prepare_regions.py):
  data/regions/bg_universe.bed     the peak-supported cCRE universe (369,321)
  data/regions/fg_*_{up,down}.bed  8 foreground sets

Steps:
  1. Compute GC% for every region in the universe via `bedtools nuc`.
  2. For each foreground set (8 of them):
     a. Subtract foreground regions from the universe → eligible control pool.
     b. Bin both foreground and eligible-control by GC (2% bins).
     c. Sample N_CONTROL_PER_FG × (foreground count) from the control pool with
        per-bin weights matching the foreground GC histogram.
     d. Extract FASTA for foreground + sampled background via `bedtools getfasta`.
  3. Write under data/regions/ame/<contrast>_<direction>/{fg.fa,bg.fa,bg.bed,gc_stats.tsv}

Outputs are ready inputs for `ame --control <bg.fa> ... <fg.fa>`.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import polars as pl

DIR = Path(__file__).resolve().parent
REG = DIR / "data" / "regions"
OUT_ROOT = REG / "ame"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

HG38_FA = str(DIR / "data" / "hg38.fa")

CONTRASTS = ["DE_vs_ESC", "HB_vs_DE", "iHEP_vs_HB", "mHEP_vs_iHEP"]
DIRECTIONS = ["up", "down"]

# Sampling parameters
N_CONTROL_PER_FG = 5
GC_BIN_WIDTH = 0.02  # 2% GC bins
RNG_SEED = 1729


def run(cmd: list[str], **kw):
    """Run a command, raise on non-zero. Quiet by default."""
    res = subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)
    return res


def compute_universe_gc() -> pl.DataFrame:
    """Run `bedtools nuc` on bg_universe.bed and return a polars DataFrame
    with columns chrom, start, end, gc, length.
    """
    bg = REG / "bg_universe.bed"
    out_tsv = OUT_ROOT / "universe_gc.tsv"
    if out_tsv.exists():
        print(f"  (cached) {out_tsv}")
    else:
        print(f"  computing GC for {bg.name} ...")
        # bedtools nuc appends: 4_pct_at 5_pct_gc 6_num_A 7_num_C 8_num_G 9_num_T 10_num_N 11_num_oth 12_seq_len
        with open(out_tsv, "w") as fh:
            subprocess.run(
                ["bedtools", "nuc", "-fi", HG38_FA, "-bed", str(bg)],
                check=True, stdout=fh,
            )

    # Parse — first 3 columns are chrom/start/end; bedtools writes a header line
    # starting with '#'. The columns we care about: pct_gc (col 5, 0-indexed 4)
    # and seq_len (col 12, 0-indexed 11).
    df = pl.read_csv(
        out_tsv, separator="\t",
        has_header=True, comment_prefix=None,
        # bedtools nuc emits a header line beginning with '#'; polars treats it
        # as the column header (the '#' prefix is fine on the first column name).
    )
    # Normalize column names: the bedtools header tokens like '4_usercol' etc.
    # are messy. Rename by position.
    df = df.rename({df.columns[0]: "chrom",
                    df.columns[1]: "start",
                    df.columns[2]: "end"})
    # pct_gc is column index 4 (5th col), seq_len is column index 11 (12th col)
    gc_col = df.columns[4]
    len_col = df.columns[11]
    df = df.select([
        pl.col("chrom"),
        pl.col("start").cast(pl.Int64),
        pl.col("end").cast(pl.Int64),
        pl.col(gc_col).cast(pl.Float64).alias("gc"),
        pl.col(len_col).cast(pl.Int64).alias("length"),
    ])
    return df


def gc_bin(df: pl.DataFrame, width: float) -> pl.DataFrame:
    """Assign each region to a GC bin index based on `gc` column."""
    return df.with_columns(
        ((pl.col("gc") / width).floor().cast(pl.Int32)).alias("gc_bin")
    )


def sample_matched_controls(
    fg: pl.DataFrame,
    pool: pl.DataFrame,
    n_per_fg: int,
    rng: np.random.Generator,
) -> pl.DataFrame:
    """For each GC bin in fg, sample n_per_fg × fg_count from the eligible
    control pool's same bin. If the pool is short in a bin (rare), take all
    available — sampling is without replacement within a bin.
    """
    fg_counts = (fg.group_by("gc_bin").len().rename({"len": "fg_n"}))
    pool_by_bin = {row["gc_bin"]: row["indices"] for row in
                   pool.with_row_index("idx").group_by("gc_bin")
                       .agg(pl.col("idx").alias("indices")).iter_rows(named=True)}
    chosen_idxs = []
    short = []
    for row in fg_counts.iter_rows(named=True):
        b = row["gc_bin"]
        need = row["fg_n"] * n_per_fg
        avail = pool_by_bin.get(b, [])
        if len(avail) == 0:
            short.append((b, need, 0))
            continue
        if len(avail) <= need:
            chosen_idxs.extend(avail)
            short.append((b, need, len(avail)))
        else:
            chosen = rng.choice(avail, size=need, replace=False)
            chosen_idxs.extend(chosen.tolist())
    if short:
        print(f"    short bins (need vs got): " +
              ", ".join(f"bin{b}: {n}/{g}" for b, n, g in short[:5]) +
              (f" (+{len(short)-5} more)" if len(short) > 5 else ""))
    return pool[chosen_idxs]


def main():
    print("[1/3] Universe GC stats")
    uni = compute_universe_gc()
    print(f"  universe: {uni.height} regions, "
          f"GC range [{uni['gc'].min():.3f}, {uni['gc'].max():.3f}], "
          f"length range [{uni['length'].min()}, {uni['length'].max()}]")
    uni = gc_bin(uni, GC_BIN_WIDTH)

    rng = np.random.default_rng(RNG_SEED)

    print("\n[2/3] Per-contrast foreground + matched background")
    for contrast in CONTRASTS:
        for direction in DIRECTIONS:
            fg_bed = REG / f"fg_{contrast}_{direction}.bed"
            if not fg_bed.exists():
                continue
            tag = f"{contrast}_{direction}"
            out_dir = OUT_ROOT / tag
            out_dir.mkdir(exist_ok=True)
            print(f"  {tag}")

            # Load fg regions and join GC from universe
            fg = pl.read_csv(fg_bed, separator="\t", has_header=False,
                             new_columns=["chrom", "start", "end"])
            fg_gc = fg.join(uni, on=["chrom", "start", "end"], how="left")
            n_missing = fg_gc["gc"].is_null().sum()
            if n_missing > 0:
                print(f"    WARN {n_missing} fg regions missing from universe GC "
                      f"(coordinate mismatch); dropping them")
                fg_gc = fg_gc.drop_nulls("gc")
            print(f"    fg: {fg_gc.height} regions, "
                  f"mean GC = {fg_gc['gc'].mean():.3f}")

            # Eligible control pool = universe MINUS fg regions
            fg_keys = fg_gc.select(["chrom", "start", "end"])
            pool = uni.join(fg_keys, on=["chrom", "start", "end"], how="anti")
            print(f"    pool: {pool.height} eligible control regions")

            # Sample matched controls
            ctrl = sample_matched_controls(fg_gc, pool, N_CONTROL_PER_FG, rng)
            print(f"    bg: {ctrl.height} sampled "
                  f"(target {fg_gc.height * N_CONTROL_PER_FG}), "
                  f"mean GC = {ctrl['gc'].mean():.3f}")

            # Persist sampled bg BED + GC stats
            bg_bed = out_dir / "bg.bed"
            (ctrl.select(["chrom", "start", "end"])
                 .write_csv(bg_bed, separator="\t", include_header=False))

            gc_stats = pl.DataFrame({
                "set": ["fg", "bg"],
                "n_regions": [fg_gc.height, ctrl.height],
                "mean_gc": [fg_gc["gc"].mean(), ctrl["gc"].mean()],
                "median_gc": [fg_gc["gc"].median(), ctrl["gc"].median()],
                "mean_length": [fg_gc["length"].mean(), ctrl["length"].mean()],
            })
            gc_stats.write_csv(out_dir / "gc_stats.tsv", separator="\t")

            # Extract fastas
            fg_fa = out_dir / "fg.fa"
            bg_fa = out_dir / "bg.fa"
            for src_bed, dst_fa in [(fg_bed, fg_fa), (bg_bed, bg_fa)]:
                subprocess.run(
                    ["bedtools", "getfasta", "-fi", HG38_FA, "-bed", str(src_bed),
                     "-fo", str(dst_fa)],
                    check=True, capture_output=True, text=True,
                )

            print(f"    -> {fg_fa.name} ({fg_fa.stat().st_size//1024} KB), "
                  f"{bg_fa.name} ({bg_fa.stat().st_size//1024} KB)")

    print(f"\n[3/3] Done. Output rooted at: {OUT_ROOT}")


if __name__ == "__main__":
    main()
