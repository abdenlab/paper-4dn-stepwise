# /// script
# requires-python = ">=3.10"
# dependencies = ["polars", "numpy", "pybigtools", "pyarrow", "scikit-learn"]
# ///
"""Joint H3K27me3 + H3K27ac stackup at T.pcg cCREs, clustered jointly.

The cCRE counterpart of 10_promoter_stackup.py: same +/-5 kb two-mark x 5-stage
matrix and the same joint k-means, over a random sample of the T.pcg (B1) cCREs
rather than gene promoters. Elements are centred on the cCRE midpoint and are NOT
strand-oriented (cCREs have no strand).xxxw

Reads : ../03_tfbs/data/ccre_universe.parquet
        data/{cutnrun,chip}/
        results/hepdiff.jointly_pca.norm.kmeans_10_7.labeled.tsv
Writes: results/ccre_joint_cluster_{stackup,metaprofiles}.npz
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

import stackup as S

DIR = Path(__file__).resolve().parent
CCRE = DIR.parent / "03_tfbs" / "data" / "ccre_universe.parquet"
LABELS = DIR / "results" / "hepdiff.jointly_pca.norm.kmeans_10_7.labeled.tsv"
TAG = "ccre_joint_cluster"
ME3_VMAX = 10.0          # bulk cCREs top out well below promoter H3K27me3
N_TOTAL = 5000


def main() -> None:
    for pth in (CCRE, LABELS):
        if not pth.exists():
            raise SystemExit(f"missing {pth} — run 02_ipt_tracks.py first "
                             f"(and stage the cCRE universe into data/)")

    cc = pl.read_parquet(CCRE).select("chrom", "start", "end", "ccre_class")
    cc = cc.with_columns(((pl.col("start") + pl.col("end")) // 2).alias("mid"))
    cc = cc.with_columns((pl.col("mid") // S.BINSIZE * S.BINSIZE).alias("bin_start"))
    lab = (pl.read_csv(LABELS, separator="\t")
             .select("chrom", pl.col("start").alias("bin_start"), "name"))
    cc = cc.join(lab, on=["chrom", "bin_start"], how="left").filter(pl.col("name") == "B1")

    rng = np.random.default_rng(S.SEED)
    if cc.height > N_TOTAL:
        cc = cc[rng.choice(cc.height, N_TOTAL, replace=False)]
    chrom = cc["chrom"].to_list(); mid = cc["mid"].to_list()
    cls = cc["ccre_class"].to_numpy()
    print(f"{len(chrom)} T.pcg cCREs (random sample of {N_TOTAL})")

    mats = S.build_matrices(chrom, mid)
    keep, mats, lab_k, order, bounds, seq, krank = S.cluster_and_order(mats, clip=False)
    cls = cls[keep]

    S.dump(TAG, mats, lab_k, order, bounds, seq, ME3_VMAX, cls_ord=cls[order])

    present = [c for c in S.CLASS_COLOR if c in set(cls)]
    S.summarize(mats, lab_k, seq, krank, extra=(
        "class top",
        lambda m: ", ".join(f"{c}:{int((cls[m] == c).sum())}"
                            for c in present if (cls[m] == c).sum() > 0)[:32]))


if __name__ == "__main__":
    main()
