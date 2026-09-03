# /// script
# requires-python = ">=3.10"
# dependencies = ["polars", "numpy", "matplotlib"]
# ///
"""T.pcg lineage restriction.

Reads : results/gene_ipt_expression.tsv
        results/gsea_lineage_<IPT>__PanglaoDB…__HB_vs_DE.tsv
Writes: figs/ipt_lineage.{png,pdf}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"

DIR = Path(__file__).resolve().parent
RESULTS = DIR / "results"
FIGS = DIR / "figs"
STAGES = ["ESC", "DE", "HB", "iHEP", "mHEP"]
X = np.arange(len(STAGES))
COLS = [("A2", "#f78200", "Active · euchromatin"),
        ("B1", "#009cdf", "Polycomb · H3K27me3 (T.pcg)"),
        ("Inactive", "#555555", "LAD · H3K9me2")]
ALPHA, LFC_CUT = 0.05, -1.0
LIB = "PanglaoDB_Augmented_2021"


def clean_term(t):
    return (t[:34] + "…") if len(t) > 35 else t


def signed_top(cluster, n_each=8):
    res = pl.read_csv(RESULTS / f"gsea_lineage_{cluster}__{LIB}__HB_vs_DE.tsv", separator="\t")
    up = res.filter(pl.col("NES") > 0).sort("FDR q-val").head(n_each)
    dn = res.filter(pl.col("NES") < 0).sort("FDR q-val").head(n_each)
    return pl.concat([up, dn]).sort("NES").select("Term", "NES", "FDR q-val")


def mean_sem(mat):
    n = np.sum(~np.isnan(mat), axis=0)
    return np.nanmean(mat, axis=0), np.nanstd(mat, axis=0) / np.sqrt(np.maximum(n, 1))


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    expr = RESULTS / "gene_ipt_expression.tsv"
    if not expr.exists():
        raise SystemExit(f"missing {expr} — run 07_gene_ipt.py first")
    for cl, *_ in COLS:
        g = RESULTS / f"gsea_lineage_{cl}__{LIB}__HB_vs_DE.tsv"
        if not g.exists():
            raise SystemExit(f"missing {g} — run 08_gsea_lineage.py first")

    df = pl.read_csv(expr, separator="\t")
    down = {cl: df.filter((pl.col("cluster") == cl) & (pl.col("lfc_HB_vs_DE") < LFC_CUT)
                          & (pl.col("lrt_padj") < ALPHA)) for cl, *_ in COLS}
    up = {cl: df.filter((pl.col("cluster") == cl) & (pl.col("lfc_HB_vs_DE") > -LFC_CUT)
                        & (pl.col("lrt_padj") < ALPHA)) for cl, *_ in COLS}

    fig, axes = plt.subplots(2, len(COLS), figsize=(4.8 * len(COLS) + 0.5, 7.4),
                             gridspec_kw=dict(height_ratios=[1, 1.6], hspace=0.42, wspace=0.4))
    for ax in axes.ravel():
        ax.spines[["top", "right"]].set_visible(False)

    for c, (cl, color, subtitle) in enumerate(COLS):
        # row A: up/down gene-set expression trajectory (mean VST +/- SEM)
        ax = axes[0, c]
        for st, ls, mfc, lab in [(down[cl], "-", color, "down (—)"),
                                 (up[cl], "--", "white", "up (- -)")]:
            if st.height == 0:
                continue
            vst = np.array([st[f"vst_{s}"].to_numpy() for s in STAGES]).T
            mu, se = mean_sem(vst)
            ax.errorbar(X, mu, yerr=se, color=color, ls=ls, marker="o", ms=5, lw=2.2,
                        mfc=mfc, mec=color, capsize=3, zorder=3,
                        label=f"{lab}  n={st.height}")
        ax.axvspan(1, 2, color="0.93", zorder=0)
        ax.set_xticks(X); ax.set_xticklabels(STAGES)
        ax.set_ylabel("gene expression\n(mean VST ± SEM)", fontsize=9)
        ax.legend(fontsize=7.5, frameon=False, loc="best")
        ax.set_ylim(4.5, 10)
        ax.set_title(f"{cl}  ·  {subtitle}", color=color, fontweight="bold", fontsize=11)

        # row B: within-IPT PanglaoDB lineage GSEA, signed NES diverging bars
        ax = axes[1, c]
        tt = signed_top(cl)
        y = np.arange(tt.height)
        nes = tt["NES"].to_numpy(); fdr = tt["FDR q-val"].to_numpy()
        ax.barh(y, nes, color=[mpl.colors.to_rgba(color, 0.95 if q < 0.05 else 0.30)
                               for q in fdr])
        ax.axvline(0, color="0.4", lw=0.8, zorder=1)
        ax.set_yticks(y)
        ax.set_yticklabels([clean_term(t) + (" *" if q < 0.05 else "")
                            for t, q in zip(tt["Term"], fdr)], fontsize=7.5)
        ax.set_xlim(-2.5, 2.5); ax.set_xticks([-2, -1, 0, 1, 2])
        ax.set_xlabel("GSEA NES  (← down · up →; * q<0.05)", fontsize=9)
        ax.set_title("lineages (PanglaoDB)", fontsize=10)

    fig.suptitle("ED-Fig 3b: T.pcg lineage restriction\n"
                 "up/down gene-set trajectories + within-IPT PanglaoDB GSEA",
                 fontsize=12, y=0.99, fontweight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"ipt_lineage.{ext}", dpi=200, bbox_inches="tight")
    print(f"-> {FIGS / 'ipt_lineage.pdf'}")
    # report down-gene-set sizes
    for cl, *_ in COLS:
        print(f"  {cl:9s} down n={down[cl].height:4d}  up n={up[cl].height:4d}")


if __name__ == "__main__":
    main()
