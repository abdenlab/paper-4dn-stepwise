# /// script
# requires-python = ">=3.10"
# dependencies = ["cooler", "cooltools", "bioframe", "numpy", "pandas", "pyarrow",
#                 "pysam"]   # bioframe.load_fasta needs it (engine="pysam")
# ///
"""DE->HB 8-day time course: cis compartment saddle strength (Hi-C).

Per-timepoint compartment strength log2((AA+BB)/(AB+BA)) at 250 kb, from a
GC-phased E1 saddle over the whole-autosome view, for both Hi-C series:

Reads : *9 x 2 mcools from the de-to-hb Hi-C freeze*, *hg38 FASTA* (GC phasing)
Writes: results/timecourse_metrics.parquet   (timepoint, comp_H9, comp_H1)
"""
from __future__ import annotations

import warnings
from pathlib import Path

import bioframe
import cooler
import cooltools
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DIR = Path(__file__).resolve().parent
OUT = DIR / "results" / "timecourse_metrics.parquet"
OUT.parent.mkdir(parents=True, exist_ok=True)

HIC = DIR / "data" / "hic"
FASTA = str(DIR / "data" / "hg38.fa")
RES = 250_000
CHROMS = [f"chr{i}" for i in range(1, 23)]

# timepoint axis (DE = day 0 of the HGF switch; +1..+7 days; HB)
TP = ["DE", "DEp1", "DEp2", "DEp3", "DEp4", "DEp5", "DEp6", "DEp7", "HB"]
LAB = ["DE", "+1", "+2", "+3", "+4", "+5", "+6", "+7", "HB"]
COOL = {
    "H9": {tp: HIC / f"H9ESCNup155_{tp}_20241021.hg38.mapq_30.1000.mcool" for tp in TP},
    "H1": {tp: HIC / f"H1_{tp}_HiC3_20240517.hg38.mapq_30.1000.mcool" for tp in TP},
}


def saddle_strength(clr, gc, view, frac=0.2, n_bins=38):
    """cis compartment strength log2((AA+BB)/(AB+BA)) from a GC-phased E1 saddle."""
    _, ev = cooltools.eigs_cis(clr, gc, view_df=view, n_eigs=1, clr_weight_name="weight")
    cvd = cooltools.expected_cis(clr, view_df=view, clr_weight_name="weight", nproc=1)
    isum, icount = cooltools.saddle(clr, cvd, ev[["chrom", "start", "end", "E1"]], "cis",
                                    n_bins=n_bins, qrange=(0.02, 0.98), view_df=view,
                                    clr_weight_name="weight", expected_value_col="balanced.avg")
    s = (isum / icount)[1:-1, 1:-1]
    k = max(1, int(s.shape[0] * frac))
    AA, BB = np.nanmean(s[-k:, -k:]), np.nanmean(s[:k, :k])
    AB, BA = np.nanmean(s[:k, -k:]), np.nanmean(s[-k:, :k])
    return float(np.log2((AA + BB) / (AB + BA)))


def compute_table() -> pd.DataFrame:
    clr0 = cooler.Cooler(f"{COOL['H9']['DEp6']}::/resolutions/{RES}")
    cs = clr0.chromsizes
    view = pd.DataFrame({"chrom": CHROMS, "start": 0,
                         "end": [cs[c] for c in CHROMS], "name": CHROMS})
    bins = clr0.bins()[:][["chrom", "start", "end"]]
    bins = bins[bins.chrom.isin(CHROMS)].reset_index(drop=True)
    gc = bioframe.frac_gc(bins, bioframe.load_fasta(FASTA))

    comp: dict[str, dict[str, float]] = {"H9": {}, "H1": {}}
    for line in ("H9", "H1"):
        for tp, lab in zip(TP, LAB):
            try:
                clr = cooler.Cooler(f"{COOL[line][tp]}::/resolutions/{RES}")
                comp[line][lab] = saddle_strength(clr, gc, view)
                print(f"  {line} {lab}: {comp[line][lab]:.3f}")
            except Exception as e:
                comp[line][lab] = np.nan
                print(f"  {line} {lab}: FAILED ({type(e).__name__}: {e})")

    tab = pd.DataFrame([dict(timepoint=lab,
                             comp_H9=comp["H9"].get(lab, np.nan),
                             comp_H1=comp["H1"].get(lab, np.nan)) for lab in LAB])
    tab.to_parquet(OUT)
    return tab


def main() -> None:
    if OUT.exists():
        tab = pd.read_parquet(OUT)
        print(f"(reusing {OUT.name}; delete it to recompute)")
    else:
        tab = compute_table()
        print(f"\n-> {OUT}")
    print("\n" + tab.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
