#!/usr/bin/env Rscript
# Parse the GENCODE v50 GTF into (a) tx2gene for tximport/tximeta and
# (b) a gene-level annotation table (gene_id, gene_name, gene_type, chrom, mito flag,
# union-exon length for GeTMM). IDs keep their version suffix to match salmon's
# --gencode output (ENST/ENSG with .N). Run via the pixi env.
suppressPackageStartupMessages({
    library(data.table)
    library(GenomicFeatures)
    library(txdbmaker)
    library(arrow)
})

# $REF / $GTF must match the annotation indexed by 01_build_index.sh.
REF  <- Sys.getenv("REF", "data/gencode_v50")
GTF  <- Sys.getenv("GTF", file.path(REF, "gencode.v50.annotation.gtf.gz"))
if (!file.exists(GTF)) stop("annotation GTF not found: ", GTF)

.args <- commandArgs(trailingOnly = FALSE)
.fa   <- sub("^--file=", "", grep("^--file=", .args, value = TRUE))
HERE  <- normalizePath(file.path(dirname(.fa)))

OUT  <- file.path(HERE, "metadata")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

## ---- fast attribute parse with data.table -----------------------------------
gtf <- fread(cmd = sprintf("zcat %s | grep -v '^#'", GTF), sep = "\t", header = FALSE,
             quote = "", col.names = c("chrom","src","feature","start","end",
                                       "score","strand","frame","attr"))

getattr <- function(attr, key) {
    m <- regmatches(attr, regexpr(sprintf('%s "[^"]*"', key), attr))
    val <- rep(NA_character_, length(attr))
    val[lengths(m) > 0 | nzchar(m)] <- sub(sprintf('%s "([^"]*)"', key), "\\1",
                                           m[nzchar(m)])
    val
}

## ---- tx2gene (transcript rows) ----------------------------------------------
tx <- gtf[feature == "transcript"]
tx2gene <- data.table(transcript_id = getattr(tx$attr, "transcript_id"),
                      gene_id       = getattr(tx$attr, "gene_id"))
stopifnot(!anyNA(tx2gene$transcript_id), !anyNA(tx2gene$gene_id))
fwrite(tx2gene, file.path(OUT, "tx2gene.tsv"), sep = "\t")

## ---- gene annotation (gene rows) --------------------------------------------
g <- gtf[feature == "gene"]
genes <- data.table(
    gene_id   = getattr(g$attr, "gene_id"),
    gene_name = getattr(g$attr, "gene_name"),
    gene_type = getattr(g$attr, "gene_type"),
    chrom     = g$chrom
)
genes[, is_mito := chrom == "chrM" | gene_type %in% c("Mt_rRNA", "Mt_tRNA")]

## ---- union-exon length per gene (for GeTMM RPK) -----------------------------
txdb  <- suppressWarnings(makeTxDbFromGFF(GTF, format = "gtf"))
exbg  <- exonsBy(txdb, by = "gene")
ulen  <- sum(width(reduce(exbg)))                      # named by gene_id (no version?)
# makeTxDbFromGFF keeps GENCODE gene_id WITH version -> should match; guard anyway
genes[, union_exon_len := ulen[gene_id]]
if (anyNA(genes$union_exon_len)) {
    # fall back: strip version on both sides
    names(ulen) <- sub("\\..*$", "", names(ulen))
    genes[is.na(union_exon_len),
          union_exon_len := ulen[sub("\\..*$", "", gene_id)]]
}

write_parquet(as.data.frame(genes), file.path(OUT, "gene_annotation.parquet"))

cat(sprintf("transcripts: %d\ngenes: %d\nmito genes: %d\ngenes missing union_exon_len: %d\n",
            nrow(tx2gene), nrow(genes), sum(genes$is_mito), sum(is.na(genes$union_exon_len))))
cat("gene_type counts (top):\n"); print(head(sort(table(genes$gene_type), decreasing = TRUE), 8))
cat("mito genes:\n"); print(genes[is_mito == TRUE, .(gene_id, gene_name, gene_type, chrom)])
