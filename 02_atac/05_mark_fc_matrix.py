# /// script
# requires-python = ">=3.10"
# dependencies = ["polars", "numpy", "matplotlib", "pyarrow", "pybigtools"]
# ///
"""Epigenetic heatmaps of differentially-accessible cCREs.

For each DA contrast, rows are significant cCREs ordered by signed ATAC log2FC;
columns are the mean fold-change ChIP/CUT&RUN signal of 4
histone marks across all 5 stages (4 marks x 5 stages = 20 columns).

Marks / sources:
  H3K4me3   CUT&RUN
  H3K27ac   ChIP-seq
  H3K27me3  CUT&RUN
  H3K9me3   CUT&RUN

Outputs:
  results/ccre_mark_fc_matrix.parquet   union cCREs x 20 mean-fc columns
  figs/05_epigenetic_<contrast>.png
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pybigtools

DIR = Path(__file__).resolve().parent

PEAKSET_PATH = DIR / "results" / "da_peak_set.parquet"
OUT_MAT = DIR / "results" / "ccre_mark_fc_matrix.parquet"
OUT_MAT.parent.mkdir(parents=True, exist_ok=True)
OUT_FIG = DIR / "figs"
OUT_FIG.mkdir(parents=True, exist_ok=True)

# the 4 stage transitions
CONTRASTS = ["DE_vs_ESC", "HB_vs_DE", "iHEP_vs_HB", "mHEP_vs_iHEP"]

ENCODE = DIR / "data" / "cutnrun"
H3K27AC = DIR / "data" / "chip"

STAGES = ["ESC", "DE", "HB", "iHEP", "mHEP"]
MARKS = ["H3K4me3", "H3K27ac", "H3K27me3", "H3K9me3"]
CUTNRUN_MARKS = {"H3K4me3", "H3K27me3", "H3K9me3"}

FLANK = 1000          # mean fc over cCRE center +/- FLANK
CAP = 0               # max rows per contrast (top-|log2FC|); 0 = no cap
SMOOTH = 15           # display-only rolling mean (rows) over the log2FC sort
N_THREADS = 16


# ---------------------------------------------------------------- track paths
def pick_fc(sample_dir: Path) -> Path | None:
    """Canonical fold-change bigwig: pooled-rep if present, else rep1."""
    pooled = sample_dir / "signal" / "pooled-rep" / "rep.pooled_x_ctl.pooled.fc.signal.bigwig"
    if pooled.exists():
        return pooled
    hits = sorted((sample_dir / "signal" / "rep1").glob("*.fc.signal.bigwig"))
    return hits[0] if hits else None


def resolve(mark: str, stage: str) -> Path:
    if mark in CUTNRUN_MARKS:
        ds = sorted(ENCODE.glob(f"??_{stage}_{mark}_*"))
        p = pick_fc(ds[0]) if ds else None
    elif mark == "H3K27ac":
        d = H3K27AC / f"{stage}_H3K27ac"
        p = pick_fc(d) if d.exists() else None
    else:
        raise ValueError(mark)
    if p is None:
        raise FileNotFoundError(f"no fc bigwig for {mark} {stage}")
    return p


def columns():
    """List of (col_idx, mark, stage, path) in mark-major, stage-minor order."""
    cols = []
    for m in MARKS:
        for s in STAGES:
            cols.append((len(cols), m, s, resolve(m, s)))
    return cols


# ---------------------------------------------------------------- matrix build
def fill_column(path: Path, chrom: np.ndarray, center: np.ndarray,
                out_col: np.ndarray) -> None:
    """Fill out_col with mean fc over center +/- FLANK for every cCRE."""
    bw = pybigtools.open(str(path))
    chromlen = bw.chroms()
    buf = np.zeros(1, dtype=np.float64)
    for i in range(len(center)):
        c = chrom[i]
        L = chromlen.get(c)
        if L is None:
            out_col[i] = np.nan
            continue
        s = max(0, int(center[i]) - FLANK)
        e = min(int(L), int(center[i]) + FLANK)
        if e <= s:
            out_col[i] = np.nan
            continue
        bw.values(c, s, e, bins=1, summary="mean", fillna=0.0, arr=buf)
        out_col[i] = buf[0]
    bw.close()


def build_matrix(union: pl.DataFrame) -> np.ndarray:
    chrom = union["chrom"].to_numpy()
    center = ((union["start"].to_numpy() + union["end"].to_numpy()) // 2)
    n = len(chrom)
    cols = columns()
    M = np.empty((n, len(cols)), dtype=np.float64)  # rows x 30
    print(f"Reading {len(cols)} bigwigs over {n:,} cCREs "
          f"({N_THREADS} threads, +/-{FLANK}bp mean fc)...")

    def worker(spec):
        j, mark, stage, path = spec
        fill_column(path, chrom, center, M[:, j])
        return j

    with ThreadPoolExecutor(max_workers=N_THREADS) as pool:
        for _ in pool.map(worker, cols):
            pass
    return M


# ---------------------------------------------------------------- main
def main():
    ps = pl.read_parquet(PEAKSET_PATH)
    print(f"DA peak set: {ps.height:,} rows over {ps['contrast'].n_unique()} contrasts")
    print(f"Using the 4 stage transitions: {CONTRASTS}")

    # per-contrast ordered cCRE lists from the DA peak set
    # (FDR<0.05 & |log2FC|>1 & max stage-mean log2-CPM>1), sorted by signed log2FC desc
    per_contrast: dict[str, pl.DataFrame] = {}
    for name in CONTRASTS:
        sub = (ps.filter(pl.col("contrast") == name)
                 .select(["chrom", "start", "end", "ccre_id", "ccre_class",
                          pl.col("log2FC").alias("log2fc"), "padj"]))
        n = sub.height
        if CAP and n > CAP:
            sub = (sub.with_columns(pl.col("log2fc").abs().alias("absfc"))
                      .sort("absfc", descending=True).head(CAP).drop("absfc"))
            print(f"  {name:16s} {n:>8,} DA peaks -> capped to {CAP:,} by |log2FC|")
        else:
            print(f"  {name:16s} {n:>8,} DA peaks (no cap)")
        per_contrast[name] = sub.sort("log2fc", descending=True)

    # union of all capped cCREs, deduped, with coords
    union = (pl.concat([df.select(["chrom", "start", "end", "ccre_id"])
                        for df in per_contrast.values()])
               .unique(subset="ccre_id", keep="first")
               .sort(["chrom", "start"]))
    print(f"Union cCREs across contrasts: {union.height:,}")

    col_names = [f"{m}.{s}" for m in MARKS for s in STAGES]
    # reuse the cached matrix if it already covers this union (plot-only iteration)
    cached = None
    if OUT_MAT.exists():
        c = pl.read_parquet(OUT_MAT)
        if set(union["ccre_id"]) <= set(c["ccre_id"]) and all(n in c.columns for n in col_names):
            cached = c
    if cached is not None:
        cc = cached.filter(pl.col("ccre_id").is_in(union["ccre_id"]))
        union = cc.select(["chrom", "start", "end", "ccre_id"])
        M = cc.select(col_names).to_numpy()
        print(f"(reused cached matrix {OUT_MAT.name})")
    else:
        M = build_matrix(union)
        (union.with_columns([pl.Series(col_names[j], M[:, j]) for j in range(M.shape[1])])
              .write_parquet(OUT_MAT))
        print(f"-> {OUT_MAT}")

    # log2 fc for display, floored
    logM = np.log2(np.clip(M, 2 ** -4, None))
    idx_of = {cid: i for i, cid in enumerate(union["ccre_id"].to_list())}

    # per-mark display range (p2,p98 across that mark's 5 stage columns, union rows)
    mark_vlim = {}
    for mi, m in enumerate(MARKS):
        block = logM[:, mi * 5:(mi + 1) * 5]
        finite = block[np.isfinite(block)]
        mark_vlim[m] = (float(np.percentile(finite, 2)),
                        float(np.percentile(finite, 98)))

    for name, sub in per_contrast.items():
        plot_contrast(name, sub, logM, idx_of, mark_vlim)


def smooth_rows(a: np.ndarray, w: int) -> np.ndarray:
    """Centered rolling mean down the rows (axis 0), NaN-aware. Display only."""
    if w <= 1 or len(a) <= w:
        return a
    a2 = np.where(np.isfinite(a), a, np.nan)
    col_med = np.nanmedian(a2, axis=0, keepdims=True)
    a2 = np.where(np.isfinite(a2), a2, col_med)
    k = np.ones(w) / w
    pad = w // 2
    out = np.empty_like(a2)
    for j in range(a2.shape[1]):
        padded = np.concatenate([np.full(pad, a2[0, j]), a2[:, j],
                                 np.full(w - pad - 1, a2[-1, j])])
        out[:, j] = np.convolve(padded, k, mode="valid")
    return out


def plot_contrast(name, sub, logM, idx_of, mark_vlim):
    rows = [idx_of[c] for c in sub["ccre_id"].to_list()]
    block = smooth_rows(logM[rows, :], SMOOTH)  # (n_rows, 30), display-smoothed
    l2 = sub["log2fc"].to_numpy()
    n = len(rows)
    n_up = int((l2 > 0).sum())

    # layout: ATAC log2FC strip | 6 mark blocks ; colorbars row underneath
    fig = plt.figure(figsize=(2 + 1.55 * len(MARKS), 9))
    gs = fig.add_gridspec(
        2, 1 + len(MARKS), height_ratios=[20, 1],
        width_ratios=[0.6] + [5] * len(MARKS), wspace=0.08, hspace=0.04,
    )
    cmap_marks = "magma"

    # ATAC log2FC sidebar (the ordering variable)
    ax0 = fig.add_subplot(gs[0, 0])
    vlim = float(np.nanpercentile(np.abs(l2), 99))
    ax0.imshow(l2[:, None], aspect="auto", cmap="RdBu_r", vmin=-vlim, vmax=vlim)
    ax0.set_xticks([0]); ax0.set_xticklabels(["ATAC\nlog2FC"], fontsize=7)
    ax0.set_yticks([])
    ax0.set_ylabel(f"{n:,} sig cCREs  (sorted by ATAC log2FC; {n_up:,} up / "
                   f"{n - n_up:,} down)", fontsize=9)
    if 0 < n_up < n:
        ax0.axhline(n_up - 0.5, color="black", lw=0.8)

    cb_axes = []
    for mi, m in enumerate(MARKS):
        ax = fig.add_subplot(gs[0, mi + 1])
        vmin, vmax = mark_vlim[m]
        im = ax.imshow(block[:, mi * 5:(mi + 1) * 5], aspect="auto",
                       cmap=cmap_marks, vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        ax.set_xticks(range(5)); ax.set_xticklabels(STAGES, fontsize=6.5, rotation=90)
        ax.set_yticks([])
        ax.set_title(m, fontsize=9)
        if 0 < n_up < n:
            ax.axhline(n_up - 0.5, color="white", lw=0.6)
        cax = fig.add_subplot(gs[1, mi + 1])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.ax.tick_params(labelsize=6)
        cb_axes.append(cb)
    cb_axes[0].set_label("log2 fold-change signal", fontsize=7)

    fig.suptitle(
        f"Epigenetic context of DA cCREs — {name}\n"
        f"mean fc over cCRE center +/-{FLANK}bp; rows = DA peak set "
        f"(FDR<0.05, |log2FC|>1, log2-CPM>1; rolling mean {SMOOTH} rows)",
        fontsize=11, y=0.98,
    )
    out = OUT_FIG / f"05_epigenetic_{name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
