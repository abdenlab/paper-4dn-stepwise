#!/usr/bin/env Rscript
# Differential expression, PER EXPERIMENT.
#   5-stage : ~stage with replicates; apeglm-shrunken sequential LFCs + Wald padj;
#             combined numeric contrasts (hepatic-vs-prehepatic, maturation); and
#             an across-stage omnibus LRT (~stage vs ~1, df = 4).
#   8-day   : single-rep time course -> ~timepoint_day continuous; LRT vs ~1 for the
#             temporal trend; apeglm-shrunken per-day slope.
suppressPackageStartupMessages({ library(data.table); library(DESeq2); library(arrow); library(apeglm) })
.args <- commandArgs(trailingOnly = FALSE)
.fa   <- sub("^--file=", "", grep("^--file=", .args, value = TRUE))
HERE  <- normalizePath(file.path(dirname(.fa)))
OUT  <- file.path(HERE, "results")
DEDIR <- file.path(OUT, "de")

dir.create(DEDIR, showWarnings = FALSE, recursive = TRUE)

# Carry gene_name through: the downstream GSEA steps rank on gene symbols.
genes   <- as.data.table(read_parquet(file.path(HERE, "metadata", "gene_annotation.parquet")))
name_of <- setNames(genes$gene_name, genes$gene_id)

tidy <- function(res, contrast, lfc = NULL) {
    dt <- as.data.table(as.data.frame(res), keep.rownames = "gene_id")
    if (!is.null(lfc)) dt[, log2FoldChange := as.data.frame(lfc)$log2FoldChange]
    dt[, gene_name := name_of[gene_id]]
    dt[, contrast := contrast]
    setcolorder(dt, c("gene_id", "gene_name"))[]
}

## ===== 5-stage =====
d5 <- readRDS(file.path(OUT, "dds_5stage.rds"))
d5$stage <- factor(as.character(d5$stage), levels = c("ESC","DE","HB","iHEP","mHEP"))
design(d5) <- ~ stage

# -- Wald: sequential + combined contrasts
dw <- DESeq(d5)
seq_pairs <- list(c("DE","ESC"), c("HB","DE"), c("iHEP","HB"), c("mHEP","iHEP"))
res5 <- list()
for (p in seq_pairs) {
    to <- p[1]; from <- p[2]
    dw$stage <- relevel(dw$stage, ref = from); dw <- nbinomWaldTest(dw)
    cf <- paste0("stage_", to, "_vs_", from)
    wald <- results(dw, name = cf)
    shr  <- lfcShrink(dw, coef = cf, type = "apeglm")
    res5[[paste0(to,"_vs_",from)]] <- tidy(wald, paste0(to,"_vs_",from), lfc = shr)
}
# combined numeric contrasts (unshrunken)
dw$stage <- relevel(dw$stage, ref = "ESC"); dw <- nbinomWaldTest(dw)
lv <- resultsNames(dw)
w <- function(...) { v <- setNames(rep(0, length(lv)), lv); a <- list(...); for(n in names(a)) v[n] <- a[[n]]; v }
# hepatic (HB,iHEP,mHEP)/3 - (ESC,DE)/2 ; maturation (iHEP,mHEP)/2 - HB
hep <- w(stage_HB_vs_ESC=1/3, stage_iHEP_vs_ESC=1/3, stage_mHEP_vs_ESC=1/3, stage_DE_vs_ESC=-1/2)
mat <- w(stage_iHEP_vs_ESC=1/2, stage_mHEP_vs_ESC=1/2, stage_HB_vs_ESC=-1)
res5[["hepatic_vs_prehepatic"]]    <- tidy(results(dw, contrast = hep), "hepatic_vs_prehepatic")
res5[["maturation_vs_prematuration"]] <- tidy(results(dw, contrast = mat), "maturation_vs_prematuration")
de5 <- rbindlist(res5, fill = TRUE)
write_parquet(as.data.frame(de5), file.path(DEDIR, "de_5stage.parquet"))

# -- LRT: omnibus across stages (full ~stage vs reduced ~1, df = 4). Refit from
#    the unfitted dds so the Wald releveling above cannot leak in.
dl   <- DESeq(d5, test = "LRT", reduced = ~ 1)
lrt5 <- results(dl)
lrt5_out <- data.table(gene_id  = rownames(lrt5),
                       gene_name = name_of[rownames(lrt5)],
                       lrt_stat = lrt5$stat,
                       pvalue   = lrt5$pvalue,
                       padj     = lrt5$padj)
write_parquet(as.data.frame(lrt5_out), file.path(DEDIR, "lrt_5stage.parquet"))

## ===== 8-day (trajectory only; exclude p3C minus-4sU control) =====
d8 <- readRDS(file.path(OUT, "dds_8day.rds"))
d8 <- d8[, d8$in_trajectory]
d8$timepoint_day <- as.numeric(d8$timepoint_day)
design(d8) <- ~ timepoint_day
d8 <- DESeq(d8, test = "LRT", reduced = ~ 1)
lrt  <- results(d8)                                    # LRT padj for the temporal trend
shr8 <- lfcShrink(d8, coef = "timepoint_day", type = "apeglm")
de8  <- tidy(lrt, "trend_per_day", lfc = shr8)
write_parquet(as.data.frame(de8), file.path(DEDIR, "de_8day_trend.parquet"))

cat("DE done.\n5-stage contrasts:\n"); print(de5[, .N, by = contrast])
cat(sprintf("5-stage HB_vs_DE padj<0.05: %d\n",
            de5[contrast=="HB_vs_DE" & padj < 0.05, .N]))
cat(sprintf("5-stage across-stage LRT padj<0.05: %d of %d\n",
            lrt5_out[padj < 0.05, .N], nrow(lrt5_out)))
cat(sprintf("8-day temporal-trend genes (LRT padj<0.05): %d\n", de8[padj < 0.05, .N]))
