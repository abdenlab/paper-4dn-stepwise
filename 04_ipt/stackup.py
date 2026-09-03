"""Shared machinery for the T.pcg joint H3K27me3 + H3K27ac stackups.

Used by 10_promoter_stackup.py (gene promoters) and 11_ccre_stackup.py (cCREs).
Both build the same thing over a different element set: a per-element +/-W signal
matrix for two marks x five stages, a joint k-means on the centre signal, and a
cluster-ordered dump that 20_plot_stackups.ipynb renders.

Bigwigs are the GEO-deposit fold-change tracks, resolved the same way
03_bin_marks.py resolves them: one flat directory per assay, files named
<NN>-<stage>_<assay>-<mark>[_<rep>].fc.signal.bigwig, pooled preferred.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pybigtools
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DIR = Path(__file__).resolve().parent
CNR_DIR = DIR / "data" / "cutnrun"
CHIP_DIR = DIR / "data" / "chip"
RESULTS = DIR / "results"

STAGES = ["ESC", "DE", "HB", "iHLC", "HLC"]
STAGE_PREFIX = {s: f"{i:02d}-{s}" for i, s in enumerate(STAGES, start=1)}
MARKS = ["H3K27me3", "H3K27ac"]
ASSAY = {"H3K27me3": ("CnR", CNR_DIR), "H3K27ac": ("ChIP", CHIP_DIR)}
MARK_CMAP = {"H3K27me3": "magma", "H3K27ac": "viridis"}

W, NB = 5000, 50
CEN = slice(NB // 2 - 1, NB // 2 + 2)
BINSIZE = 50_000
K, SEED = 3, 0

# official SCREEN cCRE colours (CA-TF/TF de-emphasised to grey; CA-TF before TF)
CLASS_COLOR = {"PLS": "#FF0000", "pELS": "#FFA700", "dELS": "#FFCD00",
               "CA-H3K4me3": "#FFAAAA", "CA-CTCF": "#00B0F0", "CA": "#06DA93",
               "CA-TF": "#9E9E9E", "TF": "#6B6B6B"}

# The canonical palette is keyed on the older iHEP/mHEP sample names.
_PALETTE_RENAME = {"iHEP": "iHLC", "mHEP": "HLC"}


def stage_colors() -> dict[str, str]:
    """Canonical stage palette (vendored in data/), re-keyed onto this module's
    stage names."""
    raw = json.loads((DIR / "data" / "stage_colors.json").read_text())
    return {_PALETTE_RENAME.get(k, k): v for k, v in raw.items()}


def resolve(mark: str, stage: str) -> Path:
    """Pooled fold-change bigwig if present, else the first replicate."""
    assay, root = ASSAY[mark]
    stem = f"{STAGE_PREFIX[stage]}_{assay}-{mark}"
    pooled = root / f"{stem}.fc.signal.bigwig"
    if pooled.exists():
        return pooled
    reps = sorted(root.glob(f"{stem}_*.fc.signal.bigwig"))
    if not reps:
        raise FileNotFoundError(f"no bigwig for {stem} under {root}")
    return reps[0]


def stack(chrom, centre, bw, have, strand=None) -> np.ndarray:
    """(n_elements, NB) of mean signal over centre +/- W.

    If `strand` is given, minus-strand rows are reversed so the profile is
    oriented 5'->3' (promoters). Elements whose window runs off the chromosome,
    or whose chromosome is absent from the bigwig, stay NaN.
    """
    out = np.full((len(chrom), NB), np.nan)
    for i, (c, t) in enumerate(zip(chrom, centre)):
        L = have.get(c)
        if L is None or t - W < 0 or t + W > L:
            continue
        v = bw.values(c, t - W, t + W, bins=NB, summary="mean", exact=True, fillna=0.0)
        out[i] = v[::-1] if (strand is not None and strand[i] == "-") else v
    return out


def build_matrices(chrom, centre, strand=None) -> dict[str, dict[str, np.ndarray]]:
    """{mark: {stage: (n, NB)}} over all MARKS x STAGES."""
    mats = {m: {} for m in MARKS}
    for m in MARKS:
        for stage in STAGES:
            bw = pybigtools.open(str(resolve(m, stage)))
            mats[m][stage] = stack(chrom, centre, bw, bw.chroms(), strand)
            bw.close()
            print(f"  {m} {stage}")
    return mats


def cluster_and_order(mats, clip=False):
    """Joint k-means on the centre signal of both marks x all stages.

    Feature vector = mean signal over the central 3 bins, log1p'd and z-scored.
    Returns (keep_mask, lab_k, order, bounds, seq, krank):
      keep_mask  elements with a finite feature vector (callers must subset too)
      order      row order: clusters by descending mean H3K27me3 at HB, then
                 within a cluster by descending mean H3K27me3 across stages
      bounds     row indices of the cluster boundaries
      seq        cluster ids in display order; krank maps id -> display rank
    """
    feat = np.column_stack([np.nanmean(mats[m][s][:, CEN], 1) for m in MARKS for s in STAGES])
    keep = np.all(np.isfinite(feat), axis=1)
    feat = feat[keep]
    mats = {m: {s: mats[m][s][keep] for s in STAGES} for m in MARKS}

    Z = StandardScaler().fit_transform(np.log1p(np.clip(feat, 0, None) if clip else feat))
    lab_k = KMeans(n_clusters=K, random_state=SEED, n_init=10).fit(Z).labels_

    me3_cen = np.nanmean(mats["H3K27me3"]["HB"][:, CEN], 1)
    me3_all = np.nanmean(np.column_stack(
        [np.nanmean(mats["H3K27me3"][s][:, CEN], 1) for s in STAGES]), 1)
    seq = sorted(range(K), key=lambda k: -np.nanmean(me3_cen[lab_k == k]))
    order = np.concatenate([np.where(lab_k == k)[0][np.argsort(-me3_all[lab_k == k])]
                            for k in seq])
    bounds = np.cumsum([np.sum(lab_k == k) for k in seq])[:-1]
    krank = {k: i for i, k in enumerate(seq)}
    return keep, mats, lab_k, order, bounds, seq, krank


def dump(tag, mats, lab_k, order, bounds, seq, me3_vmax, **strips):
    """Write the two npz caches 20_plot_stackups.ipynb reads.

    `strips` carries the left-annotation payload, already in row order:
    `cls_ord=` for the cCRE class strip, `lfc_ord=` for the promoter LFC raincloud.
    """
    RESULTS.mkdir(parents=True, exist_ok=True)
    xs = (np.arange(NB) + 0.5) / NB * (2 * W) - W
    sizes = np.array([int((lab_k == k).sum()) for k in seq])

    prof = np.array([[[np.nanmean(mats[m][s][lab_k == k], 0) for s in STAGES]
                      for m in MARKS] for k in seq])
    sem = np.array([[[np.nanstd(mats[m][s][lab_k == k], 0) / np.sqrt(max(int((lab_k == k).sum()), 1))
                      for s in STAGES] for m in MARKS] for k in seq])
    np.savez(RESULTS / f"{tag}_metaprofiles.npz", profiles=prof, sems=sem,
             sizes=sizes, x=xs, marks=np.array(MARKS), stages=np.array(STAGES))

    heat = np.stack([[mats[m][s][order] for s in STAGES] for m in MARKS])
    ac_vmax = float(np.nanpercentile(np.concatenate([mats["H3K27ac"][s] for s in STAGES]), 99))
    np.savez(RESULTS / f"{tag}_stackup.npz", heat=heat, bounds=np.asarray(bounds),
             sizes=sizes, vmax=np.array([me3_vmax, ac_vmax]),
             marks=np.array(MARKS), stages=np.array(STAGES), **strips)
    print(f"-> {RESULTS / f'{tag}_stackup.npz'}  heat={heat.shape}")
    print(f"-> {RESULTS / f'{tag}_metaprofiles.npz'}  sizes={sizes.tolist()}")


def summarize(mats, lab_k, seq, krank, extra=None):
    """Console cluster table: centre signal at ESC/HB/HLC per mark."""
    head = f"\n{'clust':6s}{'n':>6}  me3 ESC/HB/HLC   ac ESC/HB/HLC"
    print(head + ("   " + extra[0] if extra else ""))
    for k in seq:
        msk = lab_k == k
        me3 = [np.nanmean(np.nanmean(mats["H3K27me3"][s][msk][:, CEN], 1)) for s in ("ESC", "HB", "HLC")]
        ac = [np.nanmean(np.nanmean(mats["H3K27ac"][s][msk][:, CEN], 1)) for s in ("ESC", "HB", "HLC")]
        line = (f"  k{krank[k]:<3d}{int(msk.sum()):>6}  "
                f"{me3[0]:.1f}/{me3[1]:.1f}/{me3[2]:.1f}   {ac[0]:.1f}/{ac[1]:.1f}/{ac[2]:.1f}")
        print(line + ("   " + extra[1](msk) if extra else ""))
