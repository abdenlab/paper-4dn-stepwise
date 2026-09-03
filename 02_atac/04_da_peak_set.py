# /// script
# requires-python = ">=3.10"
# dependencies = ["polars", "numpy", "matplotlib", "pyarrow"]
# ///
"""Define the 'DA peak set' per contrast.

Per-contrast filter combines:
  - FDR < 0.05
  - |log2FC| > 1
  - max(stage_mean_A, stage_mean_B) > 1 in normalized log2-CPM
    (i.e., at least one of the two compared stage groups is biologically
    accessible, CPM > 2 — catches stage-specific accessibility while
    excluding background-level cCREs)

Reads:
  results/ccre_da_results.parquet
  results/ccre_insertion_matrix.parquet
  results/loess_offsets.parquet (02)

Outputs:
  results/da_peak_set.parquet           long format: ccre x contrast x stats,
                                        only DA-significant rows.
  results/da_peak_class_summary.parquet counts per cCRE class per contrast
  figs/04_volcano_panel_da.png          volcano panels, DA hits highlighted
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

DIR = Path(__file__).resolve().parent
RESULTS = DIR / "results"; RESULTS.mkdir(parents=True, exist_ok=True)
FIGS = DIR / "figs"; FIGS.mkdir(parents=True, exist_ok=True)
DA_PATH = RESULTS / "ccre_da_results.parquet"
COUNT_PATH = RESULTS / "ccre_insertion_matrix.parquet"
OFFSET_PATH = RESULTS / "loess_offsets.parquet"

SAMPLES = [
    "01_ESC.R1", "01_ESC.R2", "02_DE.R1", "02_DE.R2",
    "03_HB.R1",  "03_HB.R2",  "04_iHEP.R1", "04_iHEP.R2",
    "05_mHEP.R1", "05_mHEP.R2",
]
STAGES = ["01_ESC", "02_DE", "03_HB", "04_iHEP", "05_mHEP"]

# Each entry: contrast name -> (group_A_stages_with_weights, group_B_stages_with_weights)
# Convention: log2FC = mean(group_B) - mean(group_A) in stage_means
CONTRAST_GROUPS: dict[str, tuple[dict[str, float], dict[str, float]]] = {
    "DE_vs_ESC":         ({"01_ESC": 1.0}, {"02_DE": 1.0}),
    "HB_vs_DE":          ({"02_DE": 1.0}, {"03_HB": 1.0}),
    "iHEP_vs_HB":        ({"03_HB": 1.0}, {"04_iHEP": 1.0}),
    "mHEP_vs_iHEP":      ({"04_iHEP": 1.0}, {"05_mHEP": 1.0}),
    "hepatic_vs_pre":    ({"01_ESC": 0.5, "02_DE": 0.5},
                          {"03_HB": 1/3, "04_iHEP": 1/3, "05_mHEP": 1/3}),
    "mature_vs_immature": ({"03_HB": 0.5, "04_iHEP": 0.5}, {"05_mHEP": 1.0}),
}

FDR = 0.05
LFC = 1.0
ACC_CUTOFF = 1.0  # log2-CPM threshold for "biologically accessible"


def compute_stage_means(counts: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Returns (n_genes, 5) matrix of per-stage mean normalized log2-CPM."""
    lib = counts.sum(axis=0)
    log_cpm = np.log2((counts + 1) / lib[None, :] * 1e6) - offsets
    out = np.empty((counts.shape[0], len(STAGES)))
    for j, stg in enumerate(STAGES):
        cols = [SAMPLES.index(f"{stg}.R1"), SAMPLES.index(f"{stg}.R2")]
        out[:, j] = log_cpm[:, cols].mean(axis=1)
    return out


def main() -> None:
    da = pl.read_parquet(DA_PATH)
    print(f"loaded DA results: {da.height:,} cCREs")

    # Need stage means in normalized log-CPM. Recompute from counts + offsets,
    # filtering to the same set DA was run on.
    counts_df = pl.read_parquet(COUNT_PATH)
    offsets_df = pl.read_parquet(OFFSET_PATH)
    # Align by ccre_id ordering (DA rows -> source rows)
    # `is_in` against a Series is deprecated in polars (it now wants an imploded
    # column); a plain set of the ids is unambiguous and avoids the copy.
    keep = counts_df["ccre_id"].is_in(set(da["ccre_id"].to_list()))
    counts_kept = counts_df.filter(keep).sort("ccre_id")
    offsets_kept = offsets_df.filter(keep).sort("ccre_id")
    da_sorted = da.sort("ccre_id")

    counts_arr = counts_kept.select(SAMPLES).to_numpy().astype(np.float64)
    offsets_arr = offsets_kept.select(SAMPLES).to_numpy()
    stage_means = compute_stage_means(counts_arr, offsets_arr)

    # Per-contrast filtering and DA peak set assembly
    da_rows = []
    print(f"\n{'contrast':<22} {'pass_fdr':>10} {'+lfc':>10} {'+access':>10}")
    print("-" * 56)
    for name, (grp_a, grp_b) in CONTRAST_GROUPS.items():
        # Group means via stage mean weights
        wa = np.zeros(len(STAGES)); wb = np.zeros(len(STAGES))
        for stg, w in grp_a.items(): wa[STAGES.index(stg)] = w
        for stg, w in grp_b.items(): wb[STAGES.index(stg)] = w
        mean_a = stage_means @ wa
        mean_b = stage_means @ wb
        mean_max = np.maximum(mean_a, mean_b)

        padj = da_sorted[f"padj_{name}"].to_numpy()
        lfc  = da_sorted[f"log2FC_{name}"].to_numpy()
        f_fdr = padj < FDR
        f_lfc = np.abs(lfc) > LFC
        f_acc = mean_max > ACC_CUTOFF
        keep_da = f_fdr & f_lfc & f_acc
        print(f"{name:<22} {int(f_fdr.sum()):>10,} "
              f"{int((f_fdr & f_lfc).sum()):>10,} {int(keep_da.sum()):>10,}")

        sub = da_sorted.filter(pl.Series(keep_da)).with_columns(
            contrast=pl.lit(name),
            stage_mean_A=pl.Series(mean_a[keep_da]),
            stage_mean_B=pl.Series(mean_b[keep_da]),
        ).select([
            "contrast", "chrom", "start", "end", "ccre_id", "ccre_class",
            "mean_lcpm",
            pl.col(f"log2FC_{name}").alias("log2FC"),
            pl.col(f"p_{name}").alias("p"),
            pl.col(f"padj_{name}").alias("padj"),
            "stage_mean_A", "stage_mean_B",
        ])
        da_rows.append(sub)

    da_set = pl.concat(da_rows)
    da_set.write_parquet(RESULTS / "da_peak_set.parquet")
    print(f"\n-> {RESULTS / 'da_peak_set.parquet'}  ({da_set.height:,} rows)")

    # Volcano panels with DA peaks highlighted
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
    for ax, name in zip(axes.flat, CONTRAST_GROUPS):
        lfc = da_sorted[f"log2FC_{name}"].to_numpy()
        padj = da_sorted[f"padj_{name}"].to_numpy()
        nlp = -np.log10(np.clip(padj, 1e-300, 1))
        # recompute mean_max for this contrast
        wa = np.zeros(len(STAGES)); wb = np.zeros(len(STAGES))
        for stg, w in CONTRAST_GROUPS[name][0].items(): wa[STAGES.index(stg)] = w
        for stg, w in CONTRAST_GROUPS[name][1].items(): wb[STAGES.index(stg)] = w
        mean_max = np.maximum(stage_means @ wa, stage_means @ wb)
        is_da = (padj < FDR) & (np.abs(lfc) > LFC) & (mean_max > ACC_CUTOFF)
        ax.hexbin(lfc[~is_da], nlp[~is_da],
                  gridsize=120, cmap="Greys", bins="log", mincnt=1)
        ax.scatter(lfc[is_da], nlp[is_da], s=2, c="tab:red", alpha=0.45,
                   label=f"DA: {is_da.sum():,}")
        ax.axhline(-np.log10(FDR), color="k", lw=0.5, ls="--", alpha=0.5)
        ax.axvline(+LFC, color="k", lw=0.5, ls="--", alpha=0.5); ax.axvline(-LFC, color="k", lw=0.5, ls="--", alpha=0.5)
        ax.set_title(name, fontsize=11)
        ax.legend(loc="upper left", fontsize=9, framealpha=0.7)
    for ax in axes[-1]: ax.set_xlabel("log2 fold change")
    for ax in axes[:, 0]: ax.set_ylabel("-log10(padj)")
    fig.suptitle(f"DA peak set: FDR<{FDR}, |log2FC|>{LFC}, max(stage means)>{ACC_CUTOFF} in normalized log2-CPM",
                 fontsize=12)
    fig.tight_layout()
    out = FIGS / "04_volcano_panel_da.png"
    fig.savefig(out, dpi=120)
    print(f"-> {out}")

    # Per-class breakdown of the DA peak set
    cls = (da_set.group_by(["contrast", "ccre_class"]).len()
           .pivot(values="len", on="contrast", index="ccre_class")
           .fill_null(0))
    # also add the n_total per class (filtered universe)
    totals = da.group_by("ccre_class").len().rename({"len": "n_universe"})
    cls = cls.join(totals, on="ccre_class").sort("n_universe", descending=True)
    with pl.Config(tbl_rows=20, tbl_width_chars=200):
        print("\nDA peak set counts per cCRE class:")
        print(cls)
    cls.write_parquet(RESULTS / "da_peak_class_summary.parquet")
    print(f"-> {RESULTS / 'da_peak_class_summary.parquet'}")


if __name__ == "__main__":
    main()
