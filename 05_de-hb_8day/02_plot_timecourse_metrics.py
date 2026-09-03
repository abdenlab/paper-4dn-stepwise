# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "polars", "matplotlib", "pyarrow"]
# ///
"""Combined 8-day DE->HB time-course figure (H9), gene-level only:

  1. Compartment saddle strength (H9 + H1)        -- 01_compartment_strength.py
  2. T.a1 discrete-saddle interactions (H9, cis)  -- A1-self vs A1-Inactive obs/exp
  3. Cell-cycle scores (S, G2/M, proliferation)   -- 01_rna VST
  4. Lamin A/C (LMNA, gene)                        -- 01_rna GeTMM log2
  5. Lamin B1/B2                                   -- 01_rna GeTMM log2

Reads : results/timecourse_metrics.parquet, data/saddles.*.npz,
        ../01_rna/results/gene_{vst,getmm_log2}_8day.parquet
Output: figs/combined_timecourse.{pdf,png}
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import polars as pl
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"

DIR = Path(__file__).resolve().parent
ANALYSIS = DIR.parent
(DIR / "figs").mkdir(exist_ok=True)

METRICS = DIR / "results" / "timecourse_metrics.parquet"   # <- 01_compartment_strength.py
SADDLE = DIR / "results"                                   # <- 00_discrete_saddles.py
RNA_VST = ANALYSIS / "01_rna" / "results" / "gene_vst_8day.parquet"          # cell-cycle (same-gene z)
RNA_GETMM = ANALYSIS / "01_rna" / "results" / "gene_getmm_log2_8day.parquet"  # lamins (cross-gene)

for _p in (METRICS, SADDLE, RNA_VST, RNA_GETMM):
    if not _p.exists():
        raise SystemExit(f"missing {_p} — run 00_discrete_saddles.py, "
                         f"01_compartment_strength.py and 01_rna first")

LAB = ["DE", "+1", "+2", "+3", "+4", "+5", "+6", "+7", "HB"]
SIX = LAB.index("+6")
# RNA timepoint -> column in rna_joint (no DE-day-0 RNA)
RNA_COL = {"+1": "H9_DE-p1", "+2": "H9_DE-p2", "+3": "H9_DE-p3", "+4": "H9_DE-p4",
           "+5": "H9_DE-p5", "+6": "H9_DE-p6", "+7": "H9_DE-p7", "HB": "H9_HB"}
# discrete-saddle sample per timepoint (H9 series)
SAD_TP = ["DE", "DEp1", "DEp2", "DEp3", "DEp4", "DEp5", "DEp6", "DEp7", "HB"]
LABELS7 = ["A1", "A2", "B1", "A3", "Quies", "Inactive", "B4"]      # discrete-saddle order
iA1, iIN = LABELS7.index("A1"), LABELS7.index("Inactive")

S_GENES = ("MCM5 PCNA TYMS FEN1 MCM2 MCM4 RRM1 UNG GINS2 MCM6 CDCA7 DTL PRIM1 UHRF1 "
           "HELLS RFC2 RPA2 NASP RAD51AP1 GMNN WDR76 SLBP CCNE2 UBR7 POLD3 MSH2 ATAD2 "
           "RAD51 RRM2 CDC45 CDC6 EXO1 TIPIN DSCC1 BLM CASP8AP2 USP1 CLSPN POLA1 CHAF1B "
           "BRIP1 E2F8").split()
G2M_GENES = ("HMGB2 CDK1 NUSAP1 UBE2C BIRC5 TPX2 TOP2A NDC80 CKS2 NUF2 CKS1B MKI67 TMPO "
             "CENPF TACC3 PIMREG SMC4 CCNB2 CKAP2L CKAP2 AURKB BUB1 KIF11 ANP32E TUBB4B "
             "GTSE1 KIF20B HJURP CDCA3 JPT1 CDC20 TTK CDC25C KIF2C RANGAP1 NCAPD2 DLGAP5 "
             "CDCA2 CDCA8 ECT2 KIF23 HMMR AURKA PSRC1 ANLN LBR CKAP5 CENPE NEK2 G2E3 "
             "GAS2L3 CBX5 CENPA").split()
PROLIF = ["MKI67", "PCNA", "TOP2A"]


def rna_panel():
    """Cell-cycle scores from VST (same-gene z); lamins from GeTMM (cross-gene comparable).
    Aligned to LAB (NaN at DE — no DE-day-0 RNA)."""
    v = pl.read_parquet(RNA_VST).unique(subset="gene_name", keep="first")
    g = pl.read_parquet(RNA_GETMM).unique(subset="gene_name", keep="first")
    cols = [RNA_COL[l] for l in LAB if l in RNA_COL]

    def gene_set_mean(df, genes):
        sub = df.filter(pl.col("gene_name").is_in(genes))
        m = sub.select(cols).to_numpy().mean(axis=0)          # mean across genes
        return {l: m[cols.index(RNA_COL[l])] for l in LAB if l in RNA_COL}

    def zscore(genes):
        d = gene_set_mean(v, genes)
        full = np.array([d.get(l, np.nan) for l in LAB])
        return (full - np.nanmean(full)) / np.nanstd(full)

    def getmm_track(gene):
        d = gene_set_mean(g, [gene])
        return np.array([d.get(l, np.nan) for l in LAB])

    return dict(
        S=zscore(S_GENES), G2M=zscore(G2M_GENES), prolif=zscore(PROLIF),
        LMNA=getmm_track("LMNA"), LMNB1=getmm_track("LMNB1"), LMNB2=getmm_track("LMNB2"),
    )


SAD_SAMPLE = {"H9": lambda tp: f"H9ESCNup155_{tp}_20241021",
              "H1": lambda tp: f"H1_{tp}_HiC3_20240517"}


def saddle_track(series, cell):
    out = []
    for tp in SAD_TP:
        M = np.load(SADDLE / f"saddles.{SAD_SAMPLE[series](tp)}.kmeans_10_7.npz")["cis"]
        out.append(float(M[cell]))
    return np.array(out)


def main():
    x = np.arange(len(LAB))
    comp = pl.read_parquet(METRICS)
    rna = rna_panel()
    a1_self = saddle_track("H9", (iA1, iA1))
    a1_inact = saddle_track("H9", (iA1, iIN))
    a1_self_h1 = saddle_track("H1", (iA1, iA1))
    a1_inact_h1 = saddle_track("H1", (iA1, iIN))

    fig, axes = plt.subplots(4, 1, figsize=(7.2, 10.5), sharex=True,
                             gridspec_kw=dict(hspace=0.18))
    for ax in axes:
        ax.axvspan(SIX - 0.5, SIX + 0.5, color="0.92", zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

    # 1. compartment strength
    ax = axes[0]
    ax.plot(x, comp["comp_H9"], "o-", color="#2A4D8F", label="H9")
    ax.plot(x, comp["comp_H1"], "s--", color="#C0504D", label="H1")
    ax.set_ylabel("compartment strength\nlog2((AA+BB)/(AB+BA))", fontsize=8)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    ax.set_title("DE→HB 8-day: structure (Hi-C) and RNA, on the latest rna_joint quant",
                 fontsize=10, loc="left")

    # 2. T.a1 discrete-saddle interactions, cis — log2(obs/exp), centred at 0
    #    H9 solid, H1 dashed
    ax = axes[1]
    ax.axhline(0, color="0.7", lw=0.8, ls=":")
    ax.plot(x, np.log2(a1_self), "o-", color="#e23838", label="T.a1 ↔ T.a1 (self)")
    ax.plot(x, np.log2(a1_inact), "s-", color="0.35", label="T.a1 ↔ T.inactive")
    ax.plot(x, np.log2(a1_self_h1), "o--", color="#e23838", alpha=0.65)
    ax.plot(x, np.log2(a1_inact_h1), "s--", color="0.35", alpha=0.65)
    ax.plot([], [], color="0.5", ls="--", label="H1 (dashed); H9 solid")
    ax.set_ylabel("discrete-saddle\nlog2(obs/exp), cis", fontsize=8)
    ax.legend(fontsize=8, frameon=False)

    # 3. cell-cycle scores (latest RNA)
    ax = axes[2]
    ax.axhline(0, color="0.85", lw=0.6)
    ax.plot(x, rna["S"], "o-", color="#2E8B57", label="S-phase")
    ax.plot(x, rna["G2M"], "^-", color="#C8990C", label="G2/M")
    ax.plot(x, rna["prolif"], "d-", color="0.45", label="proliferation")
    ax.set_ylabel("cell-cycle score\n(VST, z across timepoints)", fontsize=8)
    ax.legend(fontsize=8, frameon=False, ncol=3)

    # 4. lamins — GeTMM log2, common axis (cross-gene comparable -> real stoichiometry)
    ax = axes[3]
    ax.plot(x, rna["LMNA"], "o-", color="#1f77b4", label="LMNA (A/C)")
    ax.plot(x, rna["LMNB1"], "s-", color="#2ca02c", label="LMNB1")
    ax.plot(x, rna["LMNB2"], "^-", color="#bcbd22", label="LMNB2")
    ax.set_ylabel("lamin GeTMM log2", fontsize=8)
    ax.legend(fontsize=8, frameon=False, ncol=3)
    ax.set_xticks(x); ax.set_xticklabels(LAB)
    ax.set_xlabel("day of DE→HB (DMSO/HGF);  DE = day 0  (no RNA at DE)", fontsize=9)

    for ext in ("pdf", "png"):
        fig.savefig(DIR / "figs" / f"combined_timecourse.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("A1-self:   ", " ".join(f"{t}={v:.2f}" for t, v in zip(LAB, a1_self)))
    print("A1-Inactive:", " ".join(f"{t}={v:.2f}" for t, v in zip(LAB, a1_inact)))
    print(f"-> {DIR/'figs'/'combined_timecourse.png'}")


if __name__ == "__main__":
    main()
