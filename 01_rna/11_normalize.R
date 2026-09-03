#!/usr/bin/env Rscript
# Import + normalization, PER EXPERIMENT (each on its own scale).
#   - tximport (salmon, tx2gene) with countsFromAbundance="no"  ->  raw counts +
#     per-gene-per-sample avgTxLength, so DESeq2 uses length as a GLM *offset*.
#   - drop chrM / Mt_* genes BEFORE normalization (MT mass skews the closure).
#   - then, WITHIN each experiment: expression filter, median-of-ratios size
#     factors, blind VST, and a TPM re-closure over that experiment's genes.
#
# Outputs (all keyed on ENSG gene_id, + gene_name). The 5-stage series keeps the
# unsuffixed names; the 8-day series is suffixed `_8day`:
#   gene_counts, gene_norm_counts, gene_vst, gene_tpm, size_factors, dds_5stage.rds
#   gene_counts_8day, ... , size_factors_8day, dds_8day.rds
suppressPackageStartupMessages({
    library(data.table); library(tximport); library(DESeq2)
    library(arrow); library(SummarizedExperiment)
})

.args <- commandArgs(trailingOnly = FALSE)
.fa   <- sub("^--file=", "", grep("^--file=", .args, value = TRUE))
HERE  <- normalizePath(file.path(dirname(.fa)))
MD   <- file.path(HERE, "metadata")
OUT  <- file.path(HERE, "results"); dir.create(OUT, showWarnings = FALSE)

## ---- samples + files --------------------------------------------------------
meta <- fread(file.path(MD, "samples.tsv"))
files <- file.path(meta$quant_dir, "quant.sf"); names(files) <- meta$sample
stopifnot(all(file.exists(files)))

tx2gene <- fread(file.path(MD, "tx2gene.tsv"))
genes   <- as.data.table(read_parquet(file.path(MD, "gene_annotation.parquet")))

## ---- import (gene level, length offset) -------------------------------------
# Quantification is per-library, so importing everything at once is only an I/O
# convenience -- it introduces no cross-experiment coupling.
txi <- tximport(files, type = "salmon", tx2gene = tx2gene,
                ignoreTxVersion = FALSE, countsFromAbundance = "no",
                dropInfReps = TRUE)

## ---- drop mitochondrial genes BEFORE any normalization ----------------------
mito <- genes[is_mito == TRUE, gene_id]
keep_nuc <- !(rownames(txi$counts) %in% mito)
cat(sprintf("dropping %d mito genes present in matrix\n", sum(!keep_nuc)))
for (m in c("counts","abundance","length")) txi[[m]] <- txi[[m]][keep_nuc, , drop = FALSE]

meta_df <- as.data.frame(meta); rownames(meta_df) <- meta_df$sample

## ---- per-experiment filter + normalization ----------------------------------
# suffix: "" for the 5-stage series (canonical names), "_8day" for the time course
SUFFIX <- list("5stage" = "", "8day" = "_8day")

for (expname in names(SUFFIX)) {
    sfx  <- SUFFIX[[expname]]
    cols <- meta[experiment == expname, sample]
    cat(sprintf("\n=== %s: %d libraries ===\n", expname, length(cols)))

    txe <- list(countsFromAbundance = txi$countsFromAbundance)
    for (m in c("counts","abundance","length")) txe[[m]] <- txi[[m]][, cols, drop = FALSE]

    keep <- rowSums(txe$counts >= 10) >= 2
    cat(sprintf("genes: %d -> %d after filter (>=10 counts in >=2 samples)\n",
                length(keep), sum(keep)))
    for (m in c("counts","abundance","length")) txe[[m]] <- txe[[m]][keep, , drop = FALSE]

    # design ~1: normalization only. 13_de_contrasts.R sets the modelling design.
    md  <- meta_df[cols, ]
    dds <- DESeqDataSetFromTximport(txe, colData = md, design = ~ 1)
    dds <- estimateSizeFactors(dds)     # normalizationFactors matrix (length offset)
    vsd <- vst(dds, blind = TRUE)       # shared-scale transform for clustering/PCA
    # With the avgTxLength offset, sizeFactors(dds) is NULL (DESeq2 uses
    # normalizationFactors, per gene x sample). Report a per-sample scalar
    # median-of-ratios size factor separately:
    sf  <- estimateSizeFactorsForMatrix(counts(dds))

    # nuclear-reclosed TPM: sums to 1e6 per sample over THIS experiment's genes
    tpm <- txe$abundance
    tpm <- sweep(tpm, 2, colSums(tpm), "/") * 1e6

    gid <- rownames(txe$counts)
    name_of <- genes[match(gid, gene_id), gene_name]
    pre <- function(mat) data.frame(gene_id = gid, gene_name = name_of, mat,
                                    check.names = FALSE)
    p <- function(stem) file.path(OUT, sprintf("%s%s.parquet", stem, sfx))

    write_parquet(pre(counts(dds)),                    p("gene_counts"))
    write_parquet(pre(counts(dds, normalized = TRUE)), p("gene_norm_counts"))
    write_parquet(pre(assay(vsd)),                     p("gene_vst"))
    write_parquet(pre(tpm),                            p("gene_tpm"))
    write_parquet(data.frame(sample = colnames(dds), size_factor = sf),
                  p("size_factors"))
    saveRDS(dds, file.path(OUT, sprintf("dds_%s.rds", expname)))

    cat(sprintf("wrote gene_{counts,norm_counts,vst,tpm}%s, size_factors%s; dds_%s.rds\n",
                sfx, sfx, expname))
    print(round(setNames(sf, colnames(dds)), 3))
}
