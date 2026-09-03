"""Run AME for the family-level PWM bundle (139 motifs × 8 region sets).

Inputs:
  data/pwms/family_bundle.meme         139 motifs (138 HOCOMOCO + 1 JASPAR dimer)
  data/regions/ame/<contrast>_<direction>/  fg.fa, bg.fa

Output:
  data/regions/ame_family/<contrast>_<direction>/ame_output/ame.tsv
  results/ame_family_long.parquet
"""
from __future__ import annotations

import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import polars as pl

DIR = Path(__file__).resolve().parent
AME_IN_ROOT = DIR / "data" / "regions" / "ame"
AME_OUT_ROOT = DIR / "data" / "regions" / "ame_family"
AME_OUT_ROOT.mkdir(parents=True, exist_ok=True)
PWMS = DIR / "data" / "pwms" / "family_bundle.meme"
RESULTS = DIR / "results"
RESULTS.mkdir(exist_ok=True)

CONTRASTS = ["DE_vs_ESC", "HB_vs_DE", "iHEP_vs_HB", "mHEP_vs_iHEP"]
DIRECTIONS = ["up", "down"]

N_PARALLEL = 4


def run_ame_for(tag: str) -> tuple[str, str | None]:
    in_dir = AME_IN_ROOT / tag
    fg = in_dir / "fg.fa"
    bg = in_dir / "bg.fa"
    out_dir = AME_OUT_ROOT / tag / "ame_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = out_dir / "ame.tsv"
    cmd = [
        "ame", "--text",
        "--control", str(bg),
        "--scoring", "avg", "--method", "fisher",
        "--evalue-report-threshold", "100000",
        str(fg), str(PWMS),
    ]
    try:
        with open(tsv_path, "w") as fh:
            subprocess.run(cmd, check=True, stdout=fh, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        return tag, f"AME failed: {e.stderr[:300]}"
    return tag, None


def parse_ame_tsv(path: Path, tag: str) -> pl.DataFrame:
    lines = [ln for ln in path.read_text().splitlines() if ln and not ln.startswith("#")]
    if not lines:
        return pl.DataFrame()
    df = pl.read_csv("\n".join(lines).encode(), separator="\t", has_header=True)
    contrast, direction = tag.rsplit("_", 1)
    return df.with_columns(
        pl.lit(contrast).alias("contrast"),
        pl.lit(direction).alias("direction"),
    )


def main():
    tags = [f"{c}_{d}" for c in CONTRASTS for d in DIRECTIONS]
    print(f"AME family bundle: {len(tags)} sets, parallelism={N_PARALLEL}")

    errs = []
    with ProcessPoolExecutor(N_PARALLEL) as pool:
        futures = {pool.submit(run_ame_for, t): t for t in tags}
        for fut in as_completed(futures):
            tag, err = fut.result()
            if err:
                errs.append((tag, err))
                print(f"  [FAIL] {tag}: {err}")
            else:
                print(f"  [ok]   {tag}")
    if errs:
        print(f"\n{len(errs)} runs failed — aborting parse step")
        return

    frames = []
    for tag in tags:
        path = AME_OUT_ROOT / tag / "ame_output" / "ame.tsv"
        if not path.exists():
            print(f"  WARN missing: {path}"); continue
        df = parse_ame_tsv(path, tag)
        if df.is_empty():
            print(f"  WARN empty: {path}"); continue
        frames.append(df)
        print(f"  {tag}: {df.height} motif rows")

    all_df = pl.concat(frames, how="diagonal_relaxed")
    out = RESULTS / "ame_family_long.parquet"
    all_df.write_parquet(out)
    print(f"\n-> {out}  ({all_df.height} rows)")


if __name__ == "__main__":
    main()
