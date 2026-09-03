#!/usr/bin/env Rscript
# GeTMM (Smid et al. 2018): length- AND composition-normalized expression that is
# comparable across genes AND across samples. RPK = counts / (union-exon kb), then
# edgeR TMM factors on the RPK matrix; GeTMM = CPM of the TMM-normalized RPK.
suppressPackageStartupMessages({ library(data.table); library(edgeR); library(arrow) })

.args <- commandArgs(trailingOnly = FALSE)
.fa   <- sub("^--file=", "", grep("^--file=", .args, value = TRUE))
HERE  <- normalizePath(file.path(dirname(.fa)))
OUT  <- file.path(HERE, "results")

genes <- as.data.table(read_parquet(file.path(HERE, "metadata", "gene_annotation.parquet")))

for (sfx in c("", "_8day")) {
    src <- file.path(OUT, sprintf("gene_counts%s.parquet", sfx))
    if (!file.exists(src)) stop("missing ", src, " -- run 11_normalize.R first")
    cnt <- as.data.table(read_parquet(src))
    cat(sprintf("\n=== gene_counts%s: %d genes x %d samples ===\n",
                sfx, nrow(cnt), ncol(cnt) - 2L))

    samples <- setdiff(names(cnt), c("gene_id", "gene_name"))
    m   <- as.matrix(cnt[, ..samples]); rownames(m) <- cnt$gene_id
    len_kb <- genes[match(cnt$gene_id, gene_id), union_exon_len] / 1000
    stopifnot(!anyNA(len_kb), all(len_kb > 0))

    rpk <- m / len_kb                                   # reads per kb (per sample)
    dge <- DGEList(counts = rpk)
    dge <- calcNormFactors(dge, method = "TMM")         # TMM on the RPK matrix = GeTMM
    getmm <- cpm(dge, normalized.lib.sizes = TRUE)      # comparable across genes & samples

    pre <- function(mat) data.frame(gene_id = cnt$gene_id, gene_name = cnt$gene_name,
                                    mat, check.names = FALSE)
    p <- function(stem) file.path(OUT, sprintf("%s%s.parquet", stem, sfx))
    write_parquet(pre(getmm),           p("gene_getmm"))
    write_parquet(pre(log2(getmm + 1)), p("gene_getmm_log2"))
    write_parquet(data.frame(sample = colnames(dge),
                             tmm_factor = dge$samples$norm.factors,
                             rpk_lib_size = dge$samples$lib.size),
                  p("getmm_factors"))
    cat(sprintf("wrote gene_getmm%s(+log2) and getmm_factors%s\n", sfx, sfx))
    print(round(setNames(dge$samples$norm.factors, colnames(dge)), 3))
}
