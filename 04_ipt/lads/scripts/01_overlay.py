# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas", "numpy", "bioframe", "matplotlib"]
# ///
"""Overlay human LAD maps + SPIN states (+ Shah2023 LAD atlas) onto the 50-kb IPT bins.

(1) per-IPT mean LAD fraction across 7 LMNB1-DamID cell types (incl. H1-hESC = ESC match);
(2) IPT × SPIN-state enrichment for H1 (ESC match) and HFFc6 (differentiated), local hicatlas calls;
(3) per-IPT cLAD/fLAD/ciLAD/fiLAD composition from the Shah 2023 ChIP-based LAD atlas (interpret with care).

IPT order: T.a1, T.a2, T.a3, T.pcg, T.quies, T.inactive, T.b4 (active → Polycomb → quiescent → lamina → het).
Run: source ../.venv/bin/activate; python scripts/01_overlay.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import bioframe as bf
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"

DIR = Path(__file__).resolve().parent.parent
HG38, EXT = DIR / "hg38", DIR / "external"
LABELS = Path("/net/users/nezar/projects/4dn2hepdiff/hardening/pca/output/"
              "hepdiff.jointly_pca.norm.kmeans_10_7.labeled.tsv")
(DIR / "figs").mkdir(exist_ok=True); (DIR / "output").mkdir(exist_ok=True)

IPT_ORDER = ["A1", "A2", "A3", "B1", "Quies", "Inactive", "B4"]   # T.pcg before T.quies before T.inactive
IPT_LABEL = {"A1": "T.a1", "A2": "T.a2", "A3": "T.a3", "B1": "T.pcg",
             "Quies": "T.quies", "Inactive": "T.inactive", "B4": "T.b4"}
LAD_CELLS = ["H1hESC", "HFFc6", "K562", "HCT116", "HAP1", "RPEhTERT", "U2OS"]
SPIN_ORDER = ["Speckle", "Interior_Act1", "Interior_Act2", "Interior_Act3",
              "Interior_Repr1", "Interior_Repr2", "Near_Lm1", "Near_Lm2", "Lamina"]
SPIN_ORDER_K562 = SPIN_ORDER[:-1] + ["Lamina_Like", "Lamina"]   # K562 has an extra Lamina_Like state
ATLAS_ORDER = ["cLAD", "fLAD", "fiLAD", "ciLAD"]
ATLAS_COLOR = {"cLAD": "#08306b", "fLAD": "#4292c6", "fiLAD": "#fdae6b", "ciLAD": "#fee0b2"}
ylab = [IPT_LABEL[k] for k in IPT_ORDER]

# ---- IPT bins ----
ipt = pd.read_csv(LABELS, sep="\t", dtype={"chrom": str})[["chrom", "start", "end", "name", "itemRgb"]]
ipt = ipt[ipt["name"].isin(IPT_ORDER)].reset_index(drop=True)
ipt["binlen"] = ipt["end"] - ipt["start"]
ipt["bid"] = np.arange(len(ipt))
ipt_color = {r["name"]: "#%02x%02x%02x" % tuple(int(x) for x in r["itemRgb"].split(","))
             for _, r in ipt.drop_duplicates("name").iterrows()}


def enrich_heatmap(ax, ipt_df, state_col, states, title):
    ct = pd.crosstab(ipt_df["name"], ipt_df[state_col]).reindex(index=IPT_ORDER, columns=states).fillna(0)
    row = ct.div(ct.sum(axis=1), axis=0)
    exp = ct.sum(axis=0) / ct.to_numpy().sum()
    enr = np.log2((row + 1e-6).div(exp + 1e-6, axis=1)).to_numpy()
    vlim = 4.0
    im = ax.imshow(np.clip(enr, -vlim, vlim), cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="auto")
    for i in range(enr.shape[0]):
        for j in range(enr.shape[1]):
            v = enr[i, j]
            ax.text(j, i, ("≤−4" if v <= -vlim else f"{v:+.1f}"), ha="center", va="center",
                    fontsize=6, color="white" if abs(np.clip(v, -vlim, vlim)) > 2.4 else "0.3")
    ax.set_xticks(range(len(states))); ax.set_xticklabels(states, rotation=45, ha="right", fontsize=7)
    ax.set_title(title, fontsize=10)
    return im, pd.DataFrame(enr, index=IPT_ORDER, columns=states)


# ---- (1) per-IPT LAD fraction per DamID cell type ----
for cell in LAD_CELLS:
    lad = bf.read_table(HG38 / f"LAD_{cell}_LMNB1_50kb.hg38.bed", schema="bed3")
    ipt[f"lad_{cell}"] = bf.coverage(ipt, lad)["coverage"].to_numpy() / ipt["binlen"].to_numpy()
lad_frac = ipt.groupby("name")[[f"lad_{c}" for c in LAD_CELLS]].mean().loc[IPT_ORDER]
lad_frac.columns = LAD_CELLS
lad_frac.to_csv(DIR / "output" / "ipt_lad_fraction.tsv", sep="\t")
print("per-IPT LAD fraction:\n", lad_frac.round(2).to_string())

# ---- (1b) H1 NAD fraction per IPT (nucleolus-associated, DamID 4xAP3 — covers 73% genome; skeptical) ----
nad = bf.read_table(HG38 / "NAD_H1hESC_50kb.hg38.bed", schema="bed3")
ipt["nad_H1"] = bf.coverage(ipt, nad)["coverage"].to_numpy() / ipt["binlen"].to_numpy()
nad_frac = ipt.groupby("name")["nad_H1"].mean().loc[IPT_ORDER]
nad_frac.to_csv(DIR / "output" / "ipt_nad_fraction_H1.tsv", sep="\t")
print("\nper-IPT H1 NAD fraction:\n", nad_frac.round(2).to_string())

# ---- (2) SPIN H1 + HFFc6 (local hicatlas 50-kb 'mode' calls) ----
for cell in ["H1", "HFFc6"]:
    cov = pd.read_csv(EXT / f"SPIN_{cell}_cov50kb.bed", sep="\t", dtype={"chrom": str})[["chrom", "start", "mode"]]
    ipt = ipt.merge(cov.rename(columns={"mode": f"spin_{cell}"}), on=["chrom", "start"], how="left")

# ---- (2b) SPIN K562 (hg38 25-kb states; dominant state per 50-kb bin via bioframe) ----
spin_k = bf.read_table(HG38 / "SPIN_K562_states.hg38.bed", schema="bed4")
ovk = bf.overlap(ipt, spin_k, how="inner", suffixes=("", "_s"), return_overlap=True)
ovk["olen"] = ovk["overlap_end"] - ovk["overlap_start"]
ipt["spin_K562"] = ipt["bid"].map(ovk.sort_values("olen").groupby("bid").tail(1).set_index("bid")["name_s"])

# ---- (3) Shah 2023 LAD atlas (ChIP-based) cLAD/fLAD/ciLAD/fiLAD mode ----
atl = pd.read_csv(EXT / "LADatlas_Shah2023_cov50kb.bed", sep="\t", dtype={"chrom": str})[["chrom", "start", "mode"]]
ipt = ipt.merge(atl.rename(columns={"mode": "atlas"}), on=["chrom", "start"], how="left")

# ===================== FIGURES =====================
# Fig 1: DamID LAD fraction heatmap (IPT × cell type)
fig, ax = plt.subplots(figsize=(6.2, 4.2))
M = lad_frac.to_numpy()
im = ax.imshow(M, cmap="magma", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(LAD_CELLS))); ax.set_xticklabels(LAD_CELLS, rotation=45, ha="right", fontsize=8)
ax.get_xticklabels()[0].set_fontweight("bold")
ax.set_yticks(range(len(IPT_ORDER))); ax.set_yticklabels(ylab, fontsize=9)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=7,
                color="white" if M[i, j] < 0.55 else "black")
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("mean LAD fraction", fontsize=8)
ax.set_title("Per-IPT LAD coverage (LaminB1 DamID, 50 kb, hg38)", fontsize=10)
fig.tight_layout(); fig.savefig(DIR / "figs" / "ipt_lad_fraction.pdf"); fig.savefig(DIR / "figs" / "ipt_lad_fraction.png", dpi=200); plt.close(fig)

# Fig 2: H1-hESC LAD fraction barplot
fig, ax = plt.subplots(figsize=(4.6, 3.4))
vals = lad_frac["H1hESC"].to_numpy()
ax.bar(range(len(IPT_ORDER)), vals, color=[ipt_color[k] for k in IPT_ORDER], edgecolor="0.2", lw=0.5)
ax.set_xticks(range(len(IPT_ORDER))); ax.set_xticklabels(ylab, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("LAD fraction (H1-hESC)", fontsize=9); ax.set_ylim(0, 1)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.set_title("H1-hESC LADs by IPT (ESC-stage match)", fontsize=10)
fig.tight_layout(); fig.savefig(DIR / "figs" / "ipt_lad_H1.pdf"); fig.savefig(DIR / "figs" / "ipt_lad_H1.png", dpi=200); plt.close(fig)

# Fig 2b: H1 LAD vs NAD by IPT — is T.b4 nucleolar (NAD) rather than laminar (LAD)?
fig, ax = plt.subplots(figsize=(5.8, 3.6))
x = np.arange(len(IPT_ORDER)); w = 0.38
ax.bar(x - w / 2, lad_frac["H1hESC"].to_numpy(), w, label="LAD (LMNB1 DamID)", color="#6a51a3", edgecolor="0.2", lw=0.4)
ax.bar(x + w / 2, nad_frac.to_numpy(), w, label="NAD (nucleolar DamID)", color="#e6550d", edgecolor="0.2", lw=0.4)
ax.set_xticks(x); ax.set_xticklabels(ylab, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("fraction of IPT bins (H1-hESC)", fontsize=9); ax.set_ylim(0, 1)
ax.axhline(nad_frac.mean(), color="#e6550d", ls=":", lw=0.8, alpha=0.7)  # genome-wide NAD baseline (~0.73)
ax.legend(fontsize=7, frameon=False, loc="upper left")
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.set_title("H1-hESC: LAD vs NAD by IPT  (NAD DamID covers 73% genome — interpret with care)", fontsize=8.5)
fig.tight_layout(); fig.savefig(DIR / "figs" / "ipt_lad_vs_nad_H1.pdf"); fig.savefig(DIR / "figs" / "ipt_lad_vs_nad_H1.png", dpi=200); plt.close(fig)

# Fig 3: SPIN enrichment, H1 + HFFc6 + K562 panels
SPIN_PANELS = [("H1", SPIN_ORDER), ("HFFc6", SPIN_ORDER), ("K562", SPIN_ORDER_K562)]
fig, axes = plt.subplots(1, 3, figsize=(19.5, 4.3), sharey=True)
for ax, (cell, states) in zip(axes, SPIN_PANELS):
    sub = ipt[ipt[f"spin_{cell}"].notna()]
    im, etab = enrich_heatmap(ax, sub, f"spin_{cell}", states, f"IPT × SPIN ({cell})")
    etab.to_csv(DIR / "output" / f"ipt_spin_enrichment_{cell}.tsv", sep="\t")
axes[0].set_yticks(range(len(IPT_ORDER))); axes[0].set_yticklabels(ylab, fontsize=9)
cb = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.02); cb.set_label("log2 obs/exp", fontsize=8)
fig.suptitle("IPT × SPIN nuclear-compartment state — H1-hESC (ESC match), HFFc6 (differentiated), K562", fontsize=11)
fig.savefig(DIR / "figs" / "ipt_spin_enrichment.pdf"); fig.savefig(DIR / "figs" / "ipt_spin_enrichment.png", dpi=200); plt.close(fig)

# Fig 3b: SPIN composition — row fraction P(state | IPT), NO log transform
def spin_fraction(ipt_df, state_col, states):
    ct = pd.crosstab(ipt_df["name"], ipt_df[state_col]).reindex(index=IPT_ORDER, columns=states).fillna(0)
    return ct.div(ct.sum(axis=1), axis=0)

fracs = {cell: spin_fraction(ipt[ipt[f"spin_{cell}"].notna()], f"spin_{cell}", states) for cell, states in SPIN_PANELS}
fvmax = max(f.to_numpy().max() for f in fracs.values())
fig, axes = plt.subplots(1, 3, figsize=(19.5, 4.3), sharey=True)
for ax, (cell, states) in zip(axes, SPIN_PANELS):
    F = fracs[cell].to_numpy()
    im = ax.imshow(F, cmap="Blues", vmin=0, vmax=fvmax, aspect="auto")
    for i in range(F.shape[0]):
        for j in range(F.shape[1]):
            ax.text(j, i, f"{F[i,j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if F[i, j] < 0.5 * fvmax else "black")
    ax.set_xticks(range(len(states))); ax.set_xticklabels(states, rotation=45, ha="right", fontsize=7)
    ax.set_title(f"IPT × SPIN ({cell})", fontsize=10)
    fracs[cell].to_csv(DIR / "output" / f"ipt_spin_fraction_{cell}.tsv", sep="\t")
axes[0].set_yticks(range(len(IPT_ORDER))); axes[0].set_yticklabels(ylab, fontsize=9)
cb = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.02); cb.set_label("fraction of IPT bins  P(state | IPT)", fontsize=8)
fig.suptitle("IPT × SPIN composition (row fraction, no log) — H1-hESC, HFFc6, K562", fontsize=11)
fig.savefig(DIR / "figs" / "ipt_spin_fraction.pdf"); fig.savefig(DIR / "figs" / "ipt_spin_fraction.png", dpi=200); plt.close(fig)

# Fig 4: Shah2023 LAD-atlas composition per IPT (cLAD/fLAD/fiLAD/ciLAD)
comp = (ipt[ipt["atlas"].isin(ATLAS_ORDER)].groupby("name")["atlas"].value_counts(normalize=True)
        .unstack().reindex(index=IPT_ORDER, columns=ATLAS_ORDER).fillna(0))
comp.to_csv(DIR / "output" / "ipt_ladatlas_composition.tsv", sep="\t")
fig, ax = plt.subplots(figsize=(6.4, 4.0))
left = np.zeros(len(IPT_ORDER))
for cls in ATLAS_ORDER:
    ax.barh(range(len(IPT_ORDER)), comp[cls].to_numpy(), left=left, color=ATLAS_COLOR[cls],
            edgecolor="white", lw=0.4, label=cls)
    left += comp[cls].to_numpy()
ax.set_yticks(range(len(IPT_ORDER))); ax.set_yticklabels(ylab, fontsize=9); ax.invert_yaxis()
ax.set_xlim(0, 1); ax.set_xlabel("fraction of IPT bins", fontsize=9)
ax.legend(ncol=4, fontsize=7, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02))
ax.set_title("IPT composition by Shah 2023 LAD atlas (ChIP-based — interpret with care)", fontsize=9.5, pad=22)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(DIR / "figs" / "ipt_ladatlas_composition.pdf"); fig.savefig(DIR / "figs" / "ipt_ladatlas_composition.png", dpi=200); plt.close(fig)

print("\nLAD-atlas composition (mode):\n", comp.round(2).to_string())
print("\n-> figs: ipt_lad_fraction, ipt_lad_H1, ipt_spin_enrichment (H1+HFFc6), ipt_ladatlas_composition")
