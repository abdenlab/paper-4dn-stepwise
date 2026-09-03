#!/usr/bin/env bash
# Build a decoy-aware Salmon index (whole genome as decoy). GENCODE v50 by default.
# Run via the pixi env:  cd paper/01_rna && pixi run bash 01_build_index.sh [threads]

set -euxo pipefail

REF=${REF:-data/gencode_v50}
IDX=${IDX:-$REF/salmon_index}
TXFA=${TXFA:-gencode.v50.transcripts.fa.gz}          # transcript FASTA, relative to $REF
GENOMEFA=${GENOMEFA:-GRCh38.primary_assembly.genome.fa.gz}
# --gencode strips the pipe-delimited GENCODE header to the bare ENST id.
# Set GENCODE_FLAG= (empty) for a non-GENCODE transcript FASTA.
GENCODE_FLAG=${GENCODE_FLAG---gencode}
THREADS=${1:-32}

cd "$REF"

# decoys = all genome contig names (first token of each FASTA header)
if [ ! -s decoys.txt ]; then
    zcat "$GENOMEFA" | grep '^>' | sed 's/^>//; s/[[:space:]].*//' > decoys.txt
fi
echo "decoys: $(wc -l < decoys.txt)"

# gentrome = transcripts followed by genome (order matters: targets first, decoys last)
if [ ! -s gentrome.fa.gz ]; then
    cat "$TXFA" "$GENOMEFA" > gentrome.fa.gz
fi

salmon index \
    -t gentrome.fa.gz \
    -d decoys.txt \
    -i "$IDX" \
    -k 31 \
    ${GENCODE_FLAG:+$GENCODE_FLAG} \
    --keepDuplicates \
    -p "$THREADS"

echo "INDEX BUILT: $IDX"
ls -la "$IDX"
