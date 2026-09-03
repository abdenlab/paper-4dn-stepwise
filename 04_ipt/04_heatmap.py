# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas", "numpy", "matplotlib", "seaborn", "pyarrow",
#                 "cooltools"]   # clusterheatmap.py uses numutils.coarsen + runlength
# ///
"""IPT cluster heatmap.

Rows = 50-kb bins grouped by the canonical IPT trajectory (k=10 K-means -> 7:
A1/A2/A3/B1/B4/Inactive/Quies), B4 placed LAST (it breaks a pure GC sort but is
less distracting). Sorted within group by centel_abs. Tracks: per-stage jointly-
PCA score vectors, GC, centel_abs, and the five histone marks (H3K27ac, then
H3K27me3/H3K9me2/H3K9me3/H4K20me3), each z-scored across the genome.

Output: figs/ipt_heatmap.pdf / .png, figs/ipt_colorbars.pdf / .png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"

DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DIR))
from clusterheatmap import clustermap  # noqa: E402

DIR = Path(__file__).resolve().parent
PCA_PQ = DIR / "data" / "jointly_pca.parquet"
GC_PQ = DIR / "data" / "hg38.bins.gc.50000.pq"
RESULTS = DIR / "results"
FIGS = DIR / "figs"; FIGS.mkdir(parents=True, exist_ok=True)

LABELED = RESULTS / "hepdiff.jointly_pca.norm.kmeans_10_7.labeled.tsv"
MARKS_PQ = RESULTS / "marks.50kb.pq"

STAGES = ["ESC", "DE", "HB", "iHLC", "HLC"]
MARKS = ["H3K27ac", "H3K27me3", "H3K9me2", "H3K9me3", "H4K20me3"]
N_PC = 10
COORDS = ["chrom", "start", "end"]
CLUSTER_ORDER = ["A1", "A2", "A3", "B1", "Quies", "Inactive", "B4"]


def mark_columns(bins):
    """Add z-scored mark columns; return {mark: [column names]} for the multivecs."""
    bins = bins.merge(pd.read_parquet(MARKS_PQ), on=COORDS, how="left")
    mark_cols = {m: [f"{s}.{m}" for s in STAGES] for m in MARKS}
    # z-score each track (row) within itself, across the genome (independent per
    # mark x stage) -- i.e. scale(X, axis=1) on the heatmap matrix
    for cols in mark_cols.values():
        for c in cols:
            v = bins[c].to_numpy(float)
            bins[c] = (v - np.nanmean(v)) / np.nanstd(v)
    return bins, mark_cols


def make_colorbars(trackconfs):
    """Small standalone colorbar sheet (place manually in Illustrator). Min/max
    ticks, plus a center tick for divergent tracks."""
    specs = [
        ("GC", trackconfs["GC"], True),
        ("centel_abs", trackconfs["centel_abs"], False),
        ("PC score", trackconfs["ESC.PCA"], True),
        ("mark z-score", trackconfs[MARKS[0]], True),
    ]
    fig, axes = plt.subplots(1, len(specs), figsize=(7, 3))
    for ax, (label, conf, divergent) in zip(axes, specs):
        o = conf["options"]
        vmin, vmax, cmap = o["vmin"], o["vmax"], o.get("cmap", "Reds")
        cb = fig.colorbar(ScalarMappable(Normalize(vmin, vmax), cmap), cax=ax)
        ticks = [vmin, (vmin + vmax) / 2, vmax] if divergent else [vmin, vmax]
        cb.set_ticks(ticks)
        cb.set_ticklabels([f"{t/1e6:g}M" if abs(t) >= 1e4 else f"{t:g}" for t in ticks])
        cb.ax.tick_params(labelsize=8)
        cb.set_label(label, fontsize=9)
    fig.tight_layout()
    return fig


def build_heatmap():
    bins = pd.read_parquet(PCA_PQ)
    lab = pd.read_csv(LABELED, sep="\t")[COORDS + ["name", "color"]]
    gc = pd.read_parquet(GC_PQ)[COORDS + ["GC", "centel_abs"]]
    bins = (bins.merge(lab, on=COORDS, how="left")
                .merge(gc, on=COORDS, how="left")
                .rename(columns={"name": "cluster"}))

    color_dict = (lab.dropna(subset=["name"]).drop_duplicates("name")
                  .set_index("name")["color"].to_dict())

    pc_cols = [f"PCA{i}_{s}" for s in STAGES for i in range(1, N_PC + 1)]
    mask = bins[pc_cols].notnull().all(axis=1)
    overall = np.linalg.norm(bins.loc[mask, pc_cols].values)
    for s in STAGES:
        cols = [f"PCA{i}_{s}" for i in range(1, N_PC + 1)]
        bins.loc[:, cols] = bins.loc[:, cols] / np.linalg.norm(bins.loc[mask, cols].values) * overall

    bins, mark_cols = mark_columns(bins)

    bins = bins[bins["cluster"].notnull()].reset_index(drop=True)
    bins["cluster_rank"] = bins["cluster"].map({c: i for i, c in enumerate(CLUSTER_ORDER)})
    print(f"bins={len(bins)} | order={CLUSTER_ORDER}")

    trackconfs = {
        "GC": {"type": "divergent", "options": {"cmap": "RdYlBu_r", "vmin": 0.35, "vmax": 0.65}},
        "centel_abs": {"type": "scalar", "options": {"cmap": "Greys", "vmin": 0,
                       "vmax": float(bins["centel_abs"].max())}},
        "cluster": {"type": "category", "options": {"color_dict": color_dict}},
        **{f"{s}.PCA": {"type": "divergent", "height": 0.3,
                        "multivec": [f"PCA{i}_{s}" for i in range(1, N_PC + 1)],
                        "options": {"cmap": "RdBu_r", "vmin": -0.0025, "vmax": 0.0025}}
           for s in STAGES},
        **{m: {"type": "divergent", "height": 1, "multivec": mark_cols[m],
               "options": {"cmap": "PuOr_r", "vmin": -2.5, "vmax": 2.5}} for m in mark_cols},
    }
    layout = {
        "clust": ["cluster"],
        "genomic": ["GC", "centel_abs"],
        "pc": [f"{s}.PCA" for s in STAGES],
        "marks": list(mark_cols),   # H3K27ac first, then the heterochromatin marks
    }

    fig = clustermap(bins, group_by="cluster_rank", sort_by=["centel_abs"],
                     layout=layout, trackconfs=trackconfs, coarse_factor=10,
                     figsize=(22, 16), block_gap=0.8)
    return fig, trackconfs


def main():
    for pth in (PCA_PQ, GC_PQ, LABELED, MARKS_PQ):
        if not pth.exists():
            raise SystemExit(f"missing {pth} — run 01-03 first")

    fig, trackconfs = build_heatmap()
    fig.savefig(FIGS / "ipt_heatmap.pdf")
    fig.savefig(FIGS / "ipt_heatmap.png", dpi=150)
    print(f"-> {FIGS / 'ipt_heatmap.pdf'}")

    cbfig = make_colorbars(trackconfs)
    cbfig.savefig(FIGS / "ipt_colorbars.pdf")
    cbfig.savefig(FIGS / "ipt_colorbars.png", dpi=200)
    print(f"-> {FIGS / 'ipt_colorbars.pdf'}")


if __name__ == "__main__":
    main()
