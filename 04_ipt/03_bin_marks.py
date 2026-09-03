# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas", "numpy", "bioframe", "pybigtools", "pyarrow"]
# ///
"""Bin the histone-mark fold-change tracks at 50 kb.

Writes: results/marks.50kb.pq   chrom,start,end + 25 <stage>.<mark> columns
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import bioframe
import numpy as np
import pybigtools

DIR = Path(__file__).resolve().parent
CNR_DIR = DIR / "data/cutnrun"
CHIP_DIR = DIR / "data/chip"
RESULTS = DIR / "results"

STAGES = ["ESC", "DE", "HB", "iHLC", "HLC"]
STAGE_PREFIX = {s: f"{i:02d}-{s}" for i, s in enumerate(STAGES, start=1)}

BINSIZE = 50_000
N_THREADS = 12
OUT = "marks.50kb.pq"

chromsizes = bioframe.fetch_chromsizes("hg38")
CHROMS = chromsizes.loc[:"chr22"].index.tolist()

# (mark, assay tag in the filename, directory) -- in heatmap display order
TRACKS = [
    ("H3K27ac",  "ChIP", CHIP_DIR),
    ("H3K27me3", "CnR",  CNR_DIR),
    ("H3K9me2",  "CnR",  CNR_DIR),
    ("H3K9me3",  "CnR",  CNR_DIR),
    ("H4K20me3", "CnR",  CNR_DIR),
]


def resolve(root: Path, assay: str, mark: str, stage: str) -> tuple[Path, bool]:
    """(bigwig, is_pooled) -- the pooled track if present, else the first replicate."""
    stem = f"{STAGE_PREFIX[stage]}_{assay}-{mark}"
    pooled = root / f"{stem}.fc.signal.bigwig"
    if pooled.exists():
        return pooled, True
    reps = sorted(root.glob(f"{stem}_*.fc.signal.bigwig"))
    if not reps:
        raise FileNotFoundError(f"no bigwig for {stem} under {root}")
    return reps[0], False


def bin_track(path: Path) -> np.ndarray:
    """Exact 50-kb binning across chr1-22 (one call per chromosome)."""
    bw = pybigtools.open(str(path))
    have = bw.chroms()
    out = []
    for chrom in CHROMS:
        L = int(chromsizes[chrom])
        clen_rounded = int(np.ceil(L / BINSIZE)) * BINSIZE
        n_bins = clen_rounded // BINSIZE
        if chrom not in have:
            out.append(np.full(n_bins, np.nan))
        else:
            out.append(np.asarray(bw.values(chrom, 0, clen_rounded, bins=n_bins,
                                            summary="mean", exact=True, fillna=0.0)))
    bw.close()
    return np.concatenate(out)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    for _, _, root in TRACKS:
        if not root.is_dir():
            raise SystemExit(f"not a directory: {root}")

    bins = bioframe.binnify(chromsizes.loc[:"chr22"], BINSIZE)   # chrom,start,end
    resolved = [(f"{s}.{m}", *resolve(root, assay, m, s))
                for m, assay, root in TRACKS for s in STAGES]
    specs = [(name, path) for name, path, _ in resolved]
    n_pooled = sum(is_pooled for _, _, is_pooled in resolved)
    print(f"Exact-binning {len(specs)} tracks ({n_pooled} pooled, "
          f"{len(specs) - n_pooled} single-replicate) over {len(bins):,} 50-kb bins")

    with ThreadPoolExecutor(max_workers=N_THREADS) as pool:
        arrays = list(pool.map(lambda sp: bin_track(sp[1]), specs))
    for (name, _), arr in zip(specs, arrays):
        assert len(arr) == len(bins), f"{name}: {len(arr)} != {len(bins)}"
        bins[name] = arr

    out = RESULTS / OUT
    bins.to_parquet(out)
    print(f"-> {out}  {bins.shape}")


if __name__ == "__main__":
    main()
