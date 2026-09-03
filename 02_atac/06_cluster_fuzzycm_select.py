# /// script
# requires-python = ">=3.10"
# dependencies = ["polars", "numpy", "matplotlib", "scikit-learn"]
# ///
"""Choose the fuzzy c-means cluster count for the ATAC trajectories.

Clusters the 5-stage ATAC accessibility trajectory of the DA cCRE union.

Sweeps c and reports validity to pick a principled c:
  - PC  partition coefficient (->1 crisp; higher better)
  - XB  Xie-Beni (compactness/separation; lower better)
  - stability = mean pairwise adjusted Rand index of hard labels across seeds
    (1 = same partition regardless of init).

Output: figs/06_fcm_select.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.metrics import adjusted_rand_score

DIR = Path(__file__).resolve().parent
MAT_PATH = DIR / "results" / "ccre_mark_fc_matrix.parquet"
LCPM_PATH = DIR / "results" / "ccre_norm_lcpm.parquet"
FIGS = DIR / "figs"; FIGS.mkdir(parents=True, exist_ok=True)
STAGES = ["ESC", "DE", "HB", "iHEP", "mHEP"]
CS = range(4, 11)
SEEDS = range(5)
SUB = 40_000


def fcm(X, c, m=2.0, n_iter=150, tol=1e-5, seed=0):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    U = rng.random((c, n)); U /= U.sum(0, keepdims=True)
    cent = None
    for _ in range(n_iter):
        Um = U ** m
        cent = (Um @ X) / Um.sum(1, keepdims=True)
        d2 = np.maximum(((X[None] - cent[:, None]) ** 2).sum(2), 1e-12)
        p = 1.0 / (m - 1)
        Un = np.empty_like(U)
        for i in range(c):
            Un[i] = 1.0 / ((d2[i][None] / d2) ** p).sum(0)
        if np.abs(Un - U).max() < tol:
            U = Un; break
        U = Un
    return cent, U


def xie_beni(X, cent, U, m=2.0):
    d2 = ((X[None] - cent[:, None]) ** 2).sum(2)
    num = ((U ** m) * d2).sum()
    cc = ((cent[None] - cent[:, None]) ** 2).sum(2)
    np.fill_diagonal(cc, np.inf)
    return float(num / (X.shape[0] * cc.min()))


def main():
    mat = pl.read_parquet(MAT_PATH).select("ccre_id")
    lcpm = pl.read_parquet(LCPM_PATH)
    lc = mat.join(lcpm, on="ccre_id", how="left")
    atac = np.column_stack([
        (lc[f"{SAMP}.R1"].to_numpy() + lc[f"{SAMP}.R2"].to_numpy()) / 2
        for SAMP in ["01_ESC", "02_DE", "03_HB", "04_iHEP", "05_mHEP"]])
    atac = np.where(np.isnan(atac), np.nanmin(atac, axis=1, keepdims=True), atac)
    X = (atac - atac.mean(0)) / atac.std(0)           # column z (magnitude-preserving)
    rng = np.random.default_rng(0)
    Xs = X[rng.choice(len(X), SUB, replace=False)]

    print(f"{'c':>3} {'PC':>7} {'XB':>8} {'stability(ARI)':>15}")
    pcs, xbs, stabs = [], [], []
    for c in CS:
        labs, cent0, U0 = [], None, None
        pc_acc, xb_acc = [], []
        for sd in SEEDS:
            cent, U = fcm(Xs, c, seed=sd)
            labs.append(U.argmax(0))
            pc_acc.append(float((U ** 2).sum() / U.shape[1]))
            xb_acc.append(xie_beni(Xs, cent, U))
        aris = [adjusted_rand_score(labs[i], labs[j])
                for i in range(len(labs)) for j in range(i + 1, len(labs))]
        pc, xb, st = np.mean(pc_acc), np.mean(xb_acc), np.mean(aris)
        pcs.append(pc); xbs.append(xb); stabs.append(st)
        print(f"{c:>3} {pc:>7.3f} {xb:>8.3f} {st:>15.3f}")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, y, t in zip(ax, [pcs, xbs, stabs],
                       ["partition coeff (higher=crisper)",
                        "Xie-Beni (lower=better)",
                        "stability ARI (higher=robust)"]):
        a.plot(list(CS), y, "o-"); a.set_xlabel("c"); a.set_title(t, fontsize=10); a.grid(alpha=0.3)
    fig.suptitle("Fuzzy c-means validity — ATAC trajectory of the DA cCRE union", fontsize=12)
    fig.tight_layout(); fig.savefig(FIGS / "06_fcm_select.png", dpi=130); plt.close(fig)
    print(f"-> {FIGS / '06_fcm_select.png'}")


if __name__ == "__main__":
    main()
