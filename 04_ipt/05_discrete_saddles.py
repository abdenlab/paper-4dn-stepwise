# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "cooler", "cooltools", "bioframe", "numpy", "pandas", "scipy",
#     "multiprocess", "matplotlib", "seaborn",
# ]
# ///
"""Discrete saddles over the 7 IPT classes, for the 5 differentiation stages.

Three contact scopes per stage:
  cis       whole-chromosome expected (intra-chromosomal)
  cisarm    per-arm expected (intra-arm)
  trans     inter-chromosomal

Reads : results/hepdiff.jointly_pca.norm.kmeans_10_7.labeled.tsv
        data/hic/<stem>.hg38.mapq_30.1000.mcool
        data/expected/<stem>.50000.expected.{cis,cisarm,trans}.tsv
        data/hg38.chromarms.50000.bed
Writes: results/saddles.<stage>.kmeans_10_7.npz   mean obs/exp (sum/count) per scope
        figs/saddles.kmeans_10_7.pdf              3 scopes x 5 stages
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
import seaborn as sns
from cooltools.lib import numutils
from scipy.linalg import toeplitz

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"

DIR = Path(__file__).resolve().parent
DATA = DIR / "data"
RESULTS = DIR / "results"
FIGS = DIR / "figs"

HIC = DATA / "hic"
EXPECTED = DATA / "expected"
ARMS_BED = DATA / "hg38.chromarms.50000.bed"
LABELED = RESULTS / "hepdiff.jointly_pca.norm.kmeans_10_7.labeled.tsv"

BINSIZE = 50_000
NPROC = 20
OVERWRITE = False

LABELS = ["A1", "A2", "B1", "A3", "Quies", "Inactive", "B4"]
STAGES = ["ESC", "DE", "HB", "iHLC", "HLC"]
STAGE_STEM = {
    "ESC": "ESC-FA-DSG-DdeI-DpnII-P1P2",
    "DE": "DE-FA-DSG-DdeI-DpnII-P1P2",
    "HB": "HB-FA-DSG-DdeI-DpnII-P1P2",
    "iHLC": "iHEP-FA-DSG-DdeI-DpnII-P1P2",
    "HLC": "mHEP-FA-DSG-DdeI-DpnII-P1P2",
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


# ------------------------------------------------------------------ inputs
def digitized_labels(chromosomes):
    """The IPT class track as integer codes; unassigned (NaN) -> -1."""
    klust = pd.read_table(LABELED)
    klust = klust[klust["chrom"].isin(chromosomes)].reset_index(drop=True)
    df = klust[["chrom", "start", "end", "name"]].copy()
    dct = {lab: i for i, lab in enumerate(LABELS)}
    df["K"] = df["name"].map(dct).fillna(-1).astype(int)
    print(df["K"].value_counts().sort_index().to_string())
    return df


def expected_table(stage, scope):
    return pd.read_table(EXPECTED / f"{STAGE_STEM[stage]}.{BINSIZE}.expected.{scope}.tsv")


# ------------------------------------------------------------------ plotting
def plot_all():
    """3 scopes x 5 stages of log2(mean obs/exp), annotated."""
    scopes = [("cis", "cis"), ("cisarm", "intra-arm"),
              ("trans", "trans")]
    vmin, vmax = np.log2(1 / 4), np.log2(4)
    n = len(LABELS)
    plt.figure(figsize=(9 * len(STAGES), 24))
    gs = plt.GridSpec(nrows=len(scopes), ncols=2 * len(STAGES),
                      width_ratios=[20, 1] * len(STAGES), hspace=0.5)

    for i, stage in enumerate(STAGES):
        arrs = np.load(RESULTS / f"saddles.{stage}.kmeans_10_7.npz")
        for r, (key, scope_label) in enumerate(scopes):
            ax = plt.subplot(gs[r, 2 * i])
            sns.heatmap(
                np.log2(arrs[key]),
                cmap="RdBu_r",
                annot=True,
                annot_kws={"size": 6},
                cbar=True,
                vmin=vmin,
                vmax=vmax,
                cbar_kws={"label": "log2(mean obs/exp contact freq.)"},
            )
            ax.set_aspect(1)
            ax.set_xticks(np.arange(n) + 0.5)
            ax.set_xticklabels(LABELS, rotation=90)
            ax.set_yticks(np.arange(n) + 0.5)
            ax.set_yticklabels(LABELS, rotation="horizontal")
            ax.set_ylim([n, 0])
            ax.set_title(f"{stage}\n{scope_label}" if r == 0 else scope_label)

    out = FIGS / "saddles.kmeans_10_7.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"-> {out}")


# ------------------------------------------------------------------ main
def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    if not LABELED.exists():
        raise SystemExit(f"missing {LABELED} — run 02_ipt_tracks.py first")

    chromsizes = bioframe.fetch_chromsizes("hg38", filter_chroms=True)
    chromosomes = list(chromsizes[:"chr22"].index)

    arms = bioframe.read_table(ARMS_BED, schema="bed4")
    arms = arms[arms["chrom"].isin(chromosomes)]

    regions = bioframe.fetch_chromsizes("hg38", filter_chroms=True, as_bed=True)
    regions["name"] = regions["chrom"]
    regions = regions[regions["chrom"].isin(chromosomes)]

    df = digitized_labels(chromosomes)
    interarm_pairs = [(chrom + "p", chrom + "q") for chrom in chromosomes]
    codes = np.arange(len(LABELS))

    for stage in STAGES:
        out = RESULTS / f"saddles.{stage}.kmeans_10_7.npz"
        if out.exists() and not OVERWRITE:
            print(f"skip  {stage} (exists)")
            continue
        stem = STAGE_STEM[stage]
        print(stage)
        clr = cooler.Cooler(
            f"{HIC / f'{stem}.hg38.mapq_30.1000.mcool'}::resolutions/{BINSIZE}")

        print("  cis")
        getmatrix = make_cis_obsexp_fetcher(clr, (expected_table(stage, "cis"), "balanced.avg"))
        Sc, Cc = make_saddle(getmatrix, (df, "K"), codes, "cis", regions,
                             min_diag=3, max_diag=-1)

        print("  cisarm (intra-arm)")
        getmatrix = make_cis_obsexp_fetcher(clr, (expected_table(stage, "cisarm"), "balanced.avg"))
        Sa, Ca = make_saddle(getmatrix, (df, "K"), codes, "cis", arms,
                             min_diag=3, max_diag=-1)

        # print("  interarm")
        # getmatrix = make_cis_offdiag_obsexp_fetcher(
        #     clr, (expected_table(stage, "interarm"), "balanced.avg"))
        # Si, Ci = make_saddle(getmatrix, (df, "K"), codes, "custom", arms,
        #                      min_diag=3, max_diag=-1, support_pairs=interarm_pairs)

        print("  trans")
        getmatrix = make_trans_obsexp_fetcher(
            clr, (expected_table(stage, "trans"), "balanced.avg"))
        St, Ct = make_saddle(getmatrix, (df, "K"), codes, "trans", regions,
                             min_diag=3, max_diag=-1)

        np.savez(out, cis=Sc / Cc, cisarm=Sa / Ca, trans=St / Ct)
        print(f"  saved {out}")

    plot_all()


if __name__ == "__main__":
    main()
