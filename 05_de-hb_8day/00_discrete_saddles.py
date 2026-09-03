# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "cooler", "cooltools", "bioframe", "numpy", "pandas", "scipy",
#     "multiprocess", "matplotlib",
# ]
# ///
"""Discrete saddles for the DE->HB 8-day Hi-C time course (H1 and H9 series).

Four contact scopes per sample:
  cis       whole-chromosome expected (intra-chromosomal)
  cisarm    per-arm expected (intra-arm)
  trans     inter-chromosomal


Reads (all external):
  data/hic/<sample>.hg38.mapq_30.1000.mcool             contact matrices
  data/hic/cooltools/results/expected/<sample>.50000.expected.*.tsv
  data/hic/cooltools/results/annotation/hg38.chromarms.50000.bed   arm definitions
  ../05_ipt/results/hepdiff.jointly_pca.norm.kmeans_10_7.labeled.tsv  category labels

Writes:
  data/saddles.<sample>.kmeans_10_7.npz   mean obs/exp (sum/count) per scope;
                                          the `cis` array is read by
                                          10_plot_timecourse_metrics.py
  figs/saddles.<series>.kmeans_10_7.pdf   4 scopes x 9 timepoints panel grid
"""
from __future__ import annotations

from functools import partial
from itertools import combinations
from pathlib import Path

import bioframe
import cooler
import matplotlib as mpl
import matplotlib.pyplot as plt
import multiprocess as mp
import numpy as np
import pandas as pd
from cooltools.lib import numutils
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from scipy.linalg import toeplitz

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"

DIR = Path(__file__).resolve().parent
RESULTS = DIR / "results"; RESULTS.mkdir(parents=True, exist_ok=True)
FIGS = DIR / "figs"; FIGS.mkdir(parents=True, exist_ok=True)

HIC_DIR = DIR / "data" / "hic"
EXP_DIR = HIC_DIR / "cooltools" / "results" / "expected"
ANNOT_DIR = HIC_DIR / "cooltools" / "results" / "annotation"
KLUST_TSV = DIR.parent / "05_ipt" / "results" / "hepdiff.jointly_pca.norm.kmeans_10_7.labeled.tsv"

BINSIZE = 50_000
NPROC = 20
OVERWRITE = False

# Discrete categories (k-means 10->7 labels), in display order
LABELS = ["A1", "A2", "B1", "A3", "Quies", "Inactive", "B4"]

# 8-day time course: two independent series, 9 timepoints each (DE -> HB)
TIMEPOINTS = ["DE", "DEp1", "DEp2", "DEp3", "DEp4", "DEp5", "DEp6", "DEp7", "HB"]
TP_LABELS = ["DE", "+1", "+2", "+3", "+4", "+5", "+6", "+7", "HB"]  # day after DE
SERIES = {
    "H1": [f"H1_{t}_HiC3_20240517" for t in TIMEPOINTS],
    "H9": [f"H9ESCNup155_{t}_20241021" for t in TIMEPOINTS],
}


# ------------------------------------------------------------------ saddle machinery
def make_cis_obsexp_fetcher(clr, expected, weight_name="weight"):
    expected, expected_name = expected
    expected = {k: x.values for k, x in expected.groupby("region1")[expected_name]}

    def _fetch_cis_oe(region_dict, regname1, regname2):
        reg1 = region_dict[regname1]
        obs_mat = clr.matrix(balance=weight_name).fetch(reg1)
        exp_mat = toeplitz(expected[regname1][: obs_mat.shape[0]])
        with np.errstate(divide="ignore", invalid="ignore"):
            return obs_mat / exp_mat

    return _fetch_cis_oe


def make_cis_offdiag_obsexp_fetcher(clr, expected, weight_name="weight"):
    expected, expected_name = expected
    expected = {k: x.values for k, x in expected.groupby(["region1", "region2"])[expected_name]}

    def _fetch_cis_oe(region_dict, regname1, regname2):
        reg1 = region_dict[regname1]
        reg2 = region_dict[regname2]
        obs_mat = clr.matrix(balance=weight_name).fetch(reg1, reg2)
        exp_mat = toeplitz(
            expected[regname1, regname2][: obs_mat.shape[0]],
            expected[regname1, regname2][: obs_mat.shape[1]],
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            return obs_mat / exp_mat

    return _fetch_cis_oe


def make_trans_obsexp_fetcher(clr, expected, weight_name="weight"):
    if np.isscalar(expected):
        return lambda reg1, reg2: (
            clr.matrix(balance=weight_name).fetch(reg1, reg2) / expected
        )

    elif isinstance(expected, (tuple, list)):
        expected, expected_name = expected

        if not expected_name:
            raise ValueError("Name of data column not provided.")

        expected = {
            k: x.values for k, x in expected.groupby(["region1", "region2"])[expected_name]
        }

        def _fetch_trans_exp(region1, region2):
            if (region1, region2) in expected.keys():
                return expected[region1, region2]
            elif (region2, region1) in expected.keys():
                return expected[region2, region1]
            else:
                raise KeyError(
                    "trans-exp index is missing a pair of chromosomes: "
                    "{}, {}".format(region1, region2)
                )

        def _fetch_trans_oe(region_dict, regname1, regname2):
            reg1 = region_dict[regname1]
            reg2 = region_dict[regname2]

            return (
                clr.matrix(balance=weight_name).fetch(reg1, reg2) /
                _fetch_trans_exp(regname1, regname2)
            )

        return _fetch_trans_oe

    else:
        raise ValueError("Unknown type of expected")


def _accumulate(getmatrix, region_dict, digitized, min_diag, max_diag, n_bins, arg):
    regname1, regname2 = arg

    S = np.zeros((n_bins, n_bins))
    C = np.zeros((n_bins, n_bins))

    n_bins = S.shape[0]
    matrix = getmatrix(region_dict, regname1, regname2)

    if regname1[0] == regname2[0]:
        for d in np.arange(-min_diag + 1, min_diag):
            numutils.set_diag(matrix, np.nan, d)
        if max_diag >= 0:
            for d in np.append(
                np.arange(-matrix.shape[0], -max_diag),
                np.arange(max_diag + 1, matrix.shape[0]),
            ):
                numutils.set_diag(matrix, np.nan, d)

    for i in range(n_bins):
        row_mask = digitized[regname1] == i
        for j in range(n_bins):
            col_mask = digitized[regname2] == j
            data = matrix[row_mask, :][:, col_mask]
            data = data[np.isfinite(data)]
            S[i, j] += np.sum(data)
            C[i, j] += float(len(data))

    return S, C


def make_saddle(
    getmatrix,
    digitized,
    codes,
    contact_type,
    regions,
    min_diag=3,
    max_diag=-1,
    trim_outliers=False,
    verbose=False,
    support_pairs=None,
    nproc=NPROC,
):
    digitized_df, name = digitized
    n_bins = len(codes)

    region_names = regions["name"].tolist()
    region_dict = dict(zip(
        regions["name"].tolist(),
        regions[["chrom", "start", "end"]].values.tolist()
    ))

    digitized_tracks = {
        regname: bioframe.select(digitized_df, region_dict[regname])[name]
        for regname in region_names
    }

    if contact_type == "cis":
        supports = list(zip(region_names, region_names))
    elif contact_type == "trans":
        supports = list(combinations(region_names, 2))
    elif contact_type == "custom":
        supports = support_pairs
    else:
        raise ValueError(
            "The allowed values for the contact_type argument are 'cis' or 'trans'."
        )

    interaction_sum = np.zeros((n_bins, n_bins))
    interaction_count = np.zeros((n_bins, n_bins))

    job = partial(
        _accumulate,
        getmatrix,
        region_dict,
        digitized_tracks,
        min_diag,
        max_diag,
        n_bins
    )
    with mp.Pool(nproc) as pool:
        for S, C in pool.imap_unordered(job, supports):
            interaction_sum += (S + S.T)
            interaction_count += (C + C.T)

    if trim_outliers:
        interaction_sum = interaction_sum[1:-1, 1:-1]
        interaction_count = interaction_count[1:-1, 1:-1]

    return interaction_sum, interaction_count


# ------------------------------------------------------------------ plotting
def plot_saddle(ax, M, labels, vmin, vmax, title=""):
    """Lower-triangle (+ main diagonal) log2(obs/exp) saddle drawn as individual vector
    Rectangles (NOT rasterized -> each cell is a selectable object in Illustrator).
    No cell text, despined."""
    cmap = mpl.colormaps["RdBu_r"]
    norm = Normalize(vmin=vmin, vmax=vmax)
    L = np.log2(M)
    n = len(labels)
    for i in range(n):
        for j in range(i + 1):                       # lower triangle + main diagonal
            v = L[i, j]
            if np.isfinite(v):
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       facecolor=cmap(norm(v)), edgecolor="none"))
    ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(n - 0.5, -0.5)   # row 0 at top, like imshow
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(n)); ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticks(np.arange(n)); ax.set_yticklabels(labels, fontsize=6)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=8)
    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    return sm


def plot_series(series_name, samples, tp_labels, vmin=np.log2(1 / 4), vmax=np.log2(4)):
    scopes = [("cis", "cis"), ("cisarm", "intra-arm"),
              ("interarm", "inter-arm"), ("trans", "trans")]
    nrow, ncol = len(scopes), len(samples)
    fig, axs = plt.subplots(nrow, ncol, figsize=(2.4 * ncol, 2.6 * nrow), squeeze=False)
    im = None
    for c, (sample, tpl) in enumerate(zip(samples, tp_labels)):
        arrs = np.load(RESULTS / f"saddles.{sample}.kmeans_10_7.npz")
        for r, (key, scope_label) in enumerate(scopes):
            ax = axs[r, c]
            im = plot_saddle(ax, arrs[key], LABELS, vmin, vmax)
            if r == 0:
                ax.set_title(f"{tpl}\n{scope_label}", fontsize=8)
            else:
                ax.set_title(scope_label, fontsize=8)
            if c == 0:
                ax.set_ylabel(scope_label, fontsize=9)
    fig.suptitle(f"{series_name} series — discrete saddles (log2 mean obs/exp)",
                 fontsize=12, y=1.005)
    fig.subplots_adjust(right=0.92, hspace=0.4, wspace=0.3)
    cax = fig.add_axes([0.94, 0.3, 0.012, 0.4])
    fig.colorbar(im, cax=cax, label="log2(obs/exp)")   # colorbar can stay rasterized
    out = FIGS / f"saddles.{series_name}.kmeans_10_7.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}")


# ------------------------------------------------------------------ main
def digitized_labels(chromosomes):
    """Map the k-means label track onto integer codes; NaN (unassigned) -> -1,
    which is excluded from every category mask."""
    klust = pd.read_table(KLUST_TSV)
    klust = klust[klust["chrom"].isin(chromosomes)].reset_index(drop=True)
    df = klust[["chrom", "start", "end", "name"]].copy()
    dct = {lab: i for i, lab in enumerate(LABELS)}
    df["K"] = df["name"].map(dct).fillna(-1).astype(int)
    print(df["K"].value_counts().sort_index().to_string())
    return df


def main() -> None:
    chromsizes = bioframe.fetch_chromsizes("hg38", filter_chroms=True)
    chromosomes = list(chromsizes[:"chr22"].index)

    arms = bioframe.read_table(ANNOT_DIR / f"hg38.chromarms.{BINSIZE}.bed", schema="bed4")
    arms = arms[arms["chrom"].isin(chromosomes)]

    regions_chrom = bioframe.fetch_chromsizes("hg38", filter_chroms=True, as_bed=True)
    regions_chrom["name"] = regions_chrom["chrom"]
    regions_chrom = regions_chrom[regions_chrom["chrom"].isin(chromosomes)]

    df = digitized_labels(chromosomes)
    interarm_pairs = [(chrom + "p", chrom + "q") for chrom in chromosomes]
    codes = np.arange(len(LABELS))

    for sample in SERIES["H1"] + SERIES["H9"]:
        out = RESULTS / f"saddles.{sample}.kmeans_10_7.npz"
        if out.exists() and not OVERWRITE:
            print(f"skip  {sample} (exists)")
            continue
        print(sample)
        clr = cooler.Cooler(
            f"{HIC_DIR / f'{sample}.hg38.mapq_30.1000.mcool'}::resolutions/{BINSIZE}")

        print("  cis")
        expected = pd.read_table(EXP_DIR / f"{sample}.{BINSIZE}.expected.cis.tsv")
        getmatrix = make_cis_obsexp_fetcher(clr, (expected, "balanced.avg"))
        Sc, Cc = make_saddle(getmatrix, (df, "K"), codes, "cis", regions_chrom,
                             min_diag=3, max_diag=-1)

        print("  cisarm (intra-arm)")
        expected = pd.read_table(EXP_DIR / f"{sample}.{BINSIZE}.expected.cisarm.tsv")
        getmatrix = make_cis_obsexp_fetcher(clr, (expected, "balanced.avg"))
        Sa, Ca = make_saddle(getmatrix, (df, "K"), codes, "cis", arms,
                             min_diag=3, max_diag=-1)

        # print("  interarm")
        # expected = pd.read_table(EXP_DIR / f"{sample}.{BINSIZE}.expected.interarm.tsv")
        # getmatrix = make_cis_offdiag_obsexp_fetcher(clr, (expected, "balanced.avg"))
        # Si, Ci = make_saddle(getmatrix, (df, "K"), codes, "custom", arms,
        #                      min_diag=3, max_diag=-1, support_pairs=interarm_pairs)

        print("  trans")
        expected = pd.read_table(EXP_DIR / f"{sample}.{BINSIZE}.expected.trans.tsv")
        getmatrix = make_trans_obsexp_fetcher(clr, (expected, "balanced.avg"))
        St, Ct = make_saddle(getmatrix, (df, "K"), codes, "trans", regions_chrom,
                             min_diag=3, max_diag=-1)

        np.savez(out, cis=Sc / Cc, cisarm=Sa / Ca, interarm=Si / Ci, trans=St / Ct)
        print(f"  saved {out}")

    for series_name, samples in SERIES.items():
        plot_series(series_name, samples, TP_LABELS)


if __name__ == "__main__":
    main()
