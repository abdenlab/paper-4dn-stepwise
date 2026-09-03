# `lads/` — human LAD maps + SPIN states for the IPT ↔ lamina comparison

Goal: test whether **T.inactive ≈ LADs** and **T.b4 ≈ constitutive/peripheral heterochromatin**,
beyond "they're B-compartment." See the reference frame + caveats in
[`../memory/nuclear_organization.md`](../memory/nuclear_organization.md) ("Is T.inactive/T.b4 really LAD?").

**Everything here is hg38 at 50 kb (LADs) or 25 kb (SPIN) → no liftover needed.** liftOver + chains are
installed for any future hg18/hg19 beds. Reproduce with `bash scripts/00_fetch.sh`.

## Inventory (`hg38/`, ready to overlay on 50-kb IPT bins)

### Human LMNB1 LAD domain calls — 4DN, hg38, 50 kb, "associated domains (LADs), mean of replicates"
DamID-seq / pA-DamID (van Steensel-group protocol) processed by 4DN. Downloaded from the AWS Open Data
S3 mirror (the `/@@download/` REST path is 403 without an access key; `open_data_url` is public).

| file | cell type | 4DN acc | n LADs | genome cov |
|---|---|---|---|---|
| `LAD_H1hESC_LMNB1_50kb.hg38.bed` | **H1-hESC (Tier 1)** — our ESC match | 4DNFIJXADI29 | 563 | **46%** |
| `LAD_HFFc6_LMNB1_50kb.hg38.bed` | HFFc6 fibroblast (Tier 1) | 4DNFIT9W77EE | 1235 | 56% |
| `LAD_K562_LMNB1_50kb.hg38.bed` | K562 | 4DNFIJHD22QE | 759 | 52% |
| `LAD_HCT116_LMNB1_50kb.hg38.bed` | HCT116 | 4DNFIA2LBQCD | 808 | 58% |
| `LAD_HAP1_LMNB1_50kb.hg38.bed` | HAP-1 | 4DNFIDFCY3JN | 826 | 54% |
| `LAD_RPEhTERT_LMNB1_50kb.hg38.bed` | RPE-hTERT | 4DNFIUVTO2H3 | 893 | 62% |
| `LAD_U2OS_LMNB1_50kb.hg38.bed` | U2OS | 4DNFIZ3JHKWC | 838 | 53% |

Note the gradient: **H1-hESC has the fewest LADs (563) / lowest coverage (46%)**; differentiated +
cancer lines cover 52–62%. Consistent with the in-system observation that hESC heterochromatin is minimal
and LADs expand on differentiation — and with Nezar's point that fibroblast/cancer lines accrue more
peripheral heterochromatin (HFFc6 ≈ 2.2× the H1 LAD count).

### SPIN multi-structure states — hg38, 25 kb
| file | cell type | source | states |
|---|---|---|---|
| `SPIN_K562_states.hg38.bed` | K562 | Wang…van Steensel, Belmont, Ma 2021, Genome Biol (10.1186/s13059-020-02253-3) | Speckle, Interior_Act1–3, Interior_Repr1–2, Near_Lm1–2, Lamina, Lamina_Like (10 states, 4510 domains) |

The single most useful annotation: partitions the genome relative to **speckle / interior / lamina**
simultaneously → overlay directly to test T.a1≈Speckle, T.inactive≈Lamina, T.b4≈(peripheral/nucleolar).

### Nucleolus-associated domains (NAD) — H1, hg38, 50 kb (for T.b4)
| file | cell type | 4DN acc | n | genome cov | note |
|---|---|---|---|---|---|
| `NAD_H1hESC_50kb.hg38.bed` | H1-hESC | 4DNFI8K9OYU3 | 548 | **73%** | DamID (4xAP3 nucleolar bait); the right handle for T.b4 |
⚠ very broad calls (73% coverage) + Nezar distrusts the NAD assay → read the *gradient*, not absolutes.

## Reference only (NOT hg38)
- `raw/mm9_cLAD_regions.bed`, `raw/mm9_ciLAD_regions.bed` — **constitutive LAD / inter-LAD** partition,
  Meuleman 2013 (10.1101/gr.141028.112), via GSE17051 (Peric-Hupkes mouse ESC/NPC/AC/MEF panel).
  **MOUSE mm9.** No direct mm9→hg38 chain exists (cross-species), so these are a *conceptual* reference,
  not lifted onto our bins. cLAD = LAD in all 4 mouse types; defined by **A/T content / isochores**, NOT a
  histone mark (so cLAD≠H3K9me3, fLAD≠H3K9me2 — see the memory note).

## What was deliberately skipped / not auto-fetchable
- **NADs**: H1-hESC NAD (50 kb) now pulled (see above) — 4DN also has 25/100 kb if needed.
- **GC**: skipped per request (cLAD = A/T-rich → relevant cross-check for later).
- Classic per-cell-type LAD beds from paper supplements (Guelen 2008 Tig3 = GEO GSE8854 has only RAW array
  data; Kind 2013 KBM7) — superseded here by the 4DN hg38 LAD calls.

## Layout
- `scripts/00_fetch.sh` — reproducible download of everything above.
- `raw/` — original downloads (gz), chains (`hg18ToHg38`, `hg19ToHg38`), mouse cLAD/ciLAD.
- `hg38/` — analysis-ready hg38 beds (7 LMNB1-DamID LAD maps + SPIN K562).
- `external/` — symlinks to local hicatlas 50-kb `mode` calls: SPIN **H1** + **HFFc6**
  (`/net/users/nezar/projects/hicatlas/states/spin/`) and the **Shah 2023 ChIP LAD atlas**
  (`…/states/lad_atlas/`, Shah et al. 2023, 10.1186/s13059-023-02849-5 — ChIP, not DamID).

## Results — overlay on the 50-kb IPT bins (`scripts/01_overlay.py`)
51,844 labelled IPT bins. Sources: 7 LMNB1-DamID LAD maps (`hg38/`); **SPIN H1 + HFFc6** (local hicatlas
50-kb `mode` calls — H1 = ESC match); **Shah 2023 ChIP LAD atlas** (cLAD/ciLAD/fLAD/fiLAD per bin).
IPT order: T.a1, T.a2, T.a3, **T.pcg, T.quies, T.inactive**, T.b4.
Figs: `ipt_lad_fraction`, `ipt_lad_H1`, **`ipt_lad_vs_nad_H1`** (T.b4-focused LAD vs NAD), `ipt_spin_enrichment`
(log2 obs/exp) + `ipt_spin_fraction` (row composition P(state|IPT), no log) — each H1|HFFc6|K562, `ipt_ladatlas_composition`.
Tables: `output/ipt_lad_fraction.tsv`, `ipt_spin_{enrichment,fraction}_{H1,HFFc6,K562}.tsv`,
`ipt_ladatlas_composition.tsv`. (K562 keeps its extra `Lamina_Like` state; H1/HFFc6 have 9 states.)

**The central claim holds: T.inactive ≈ LADs, robustly and constitutively.**

| IPT | H1 LAD frac | all-7 range | SPIN H1 (top) | Shah atlas (mode) | reading |
|---|---|---|---|---|---|
| **T.inactive** | **0.96** | 0.82–0.98 | **Lamina (+1.5)** | **63% cLAD** + 28% fLAD | **bona fide, constitutive LADs** |
| T.quies | 0.62 | 0.59–0.71 | Interior_Act3/Near_Lm | 51% fiLAD, 32% fLAD, 14% cLAD | mixed peripheral, flanks T.inactive |
| T.pcg | 0.13 | 0.27–0.72 (diff.) | interior (Act2 +2.0, Repr2 +1.4) | **83% fiLAD** | **interior Polycomb, NOT lamina**; facultative |
| T.b4 | 0.16 | 0.33–0.64 (diff.) | **Near_Lm1 (+3.8)**, Lamina ≤−4 | 65% fiLAD, 9% cLAD | **near-lamina/het, NOT clean Lamina**; low in H1 |
| T.a3 | 0.11 | 0.15–0.35 | Interior_Act1/Repr1 | 65% fiLAD, 30% ciLAD | weakly active interior |
| T.a2 | 0.00 | 0.02–0.18 | Interior_Act1/2 | 61% ciLAD | active interior |
| **T.a1** | **0.00** | 0.00–0.02 | **Speckle (+3.0)** | **90% ciLAD** | speckle/A1, **never LAD** |

Key points:
- **T.inactive is THE constitutive LAD trajectory** — DamID ≈0.9 in every line incl. H1, SPIN-H1 **Lamina**-
  enriched, and **63% cLAD** by the ChIP atlas. The external lamin evidence the narrative needed.
- **T.a1 = nuclear speckle** (SPIN-H1 +3.0; 90% ciLAD; 0% LAD) — the speckle/A1 anchor, confirmed.
- **T.pcg = interior Polycomb, not lamina** (SPIN-H1 interior; 83% fiLAD); LAD fraction **facultative** — low
  in H1 (0.13), rises in fibroblast/cancer (HFFc6 0.66, RPE 0.72) = lamina-invasion on differentiation.
- **T.b4 ≠ bona fide LAD in H1**: low-LAD (0.16), maps to **Near_Lm** (not Lamina) in SPIN-H1, and the atlas
  calls it mostly **fiLAD not cLAD**. Consistent with hESC centromeres being interior (Wiblin 2005) and
  LAD/NAD overlap. (This is also where the Shah-atlas ChIP interpretation looks *off* — pericentromeric
  constitutive het arguably should be cLAD, not fiLAD; Nezar's skepticism noted.)
- **T.b4 = nucleolar (NAD), not laminar — `ipt_lad_vs_nad_H1`.** H1 NAD overlay (`hg38/NAD_H1hESC_50kb.hg38.bed`,
  4DN 4xAP3 DamID): **T.b4 NAD 0.95 ≫ LAD 0.16** — the one trajectory strongly nucleolar and *not* laminar.
  Conversely **T.inactive is the only IPT with LAD (0.96) > NAD (0.84)** = laminar. NAD gradient T.b4 (0.95) >
  T.inactive (0.84) > T.pcg/T.quies (0.75) > active (0.49–0.57). So **T.b4 = perinucleolar/pericentromeric het,
  T.inactive = laminar** — the clean split, *without* SPIN. ⚠ **NAD caveat**: the 4DN H1 NAD calls cover **73%
  of the genome** (very permissive; baseline dotted line at 0.73) and Nezar distrusts the NAD assay — read the
  *gradient/contrast*, not absolute values. (Discrete labels = boulder of salt.)
- **Constitutive vs facultative falls out empirically** (T.inactive/T.quies constitutive; T.pcg/T.b4 H1-low →
  differentiated-high) without relying on the mouse-defined Meuleman cLAD/fLAD calls — and is mirrored by the
  H1→HFFc6 SPIN shift.

CAVEATS:
- SPIN/atlas are non-hepatic cell types; H1 is the only primed-hESC LAD match (no hepatic lamin map).
- **Shah 2023 LAD atlas is ChIP-based (LaminB1 ChIP, not DamID) — interpret its cLAD/fLAD/fiLAD calls with
  care** (esp. pericentromeric T.b4, flagged above).
- **SPIN provenance/quality**: H1 + HFFc6 are official 4DN releases (Jian Ma lab) but use a **9-state**
  scheme **lacking `Lamina_Like`**; K562 is the **original-paper 10-state** flagship (Google Drive). The
  binning of the coverage files is verified correct (99.1% vs a fresh re-derivation), so the H1/HFFc6 "look"
  is *not* a binning bug — it's the coarser 9-state model: the 9-state H1/HFFc6 lack `Lamina_Like`, so T.b4
  lands in `Near_Lm1` (in K562 it lands in `Lamina_Like`), and T.quies/T.pcg are mushier. The robust calls
  (T.a1→Speckle, T.inactive→Lamina) are **identical across all three**. T.pcg→Interior_Act in H1 may be
  *real* poised/bivalent-ESC biology, not error. **For the ESC-stage lamina call, prefer the H1 LMNB1-DamID
  LADs** (direct measurement) over the H1 SPIN-Lamina state; use K562 (10-state) as the better-resolved SPIN
  reference.
- **Do NOT read T.b4 off SPIN** (Nezar, resolved): `Lamina_Like`/`Lamina` = B2/B3, which are **cen vs telo
  Rabl-split versions of the same IPG**, NOT distinct chromatin states — and Ma's GM12878-trained subcompartment
  caller **excludes B4 from training**, so it conflates B2/B3 with B4. Constitutive het = **B4** (pericentric/
  telo/chr19/H3K9me3) = our **T.b4**, which SPIN does not model; in some cell types (HCT116) LAD regions fill
  with H3K9me3 (B4 pervasive — Spracklin, Abdennur et al. 2023, NSMB, 10.1038/s41594-022-00892-7). So T.b4 →
  Lamina_Like is a **cen-Rabl positional coincidence**. Characterize T.b4 with **H3K9me3 / NADs**, not SPIN.
  (See `../memory/nuclear_organization.md`.)
