#!/usr/bin/env bash
# fastp trim -> salmon quant (selective alignment, decoy-aware) for all libraries.
# Library type auto-detected per sample (-l A).
#
# Driver:  pixi run bash 02_quant.sh [jobs] [threads_per_job]
# Worker (internal):  02_quant.sh --one <sample> <r1> <r2>
#
# Reads  metadata/quant_manifest.tsv  (written by 00_build_metadata.py)
# Writes results/quant/<sample>/     (one Salmon output dir per library)
#
# $REF/$IDX must match the index built by 01_build_index.sh.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REF=${REF:-data/gencode_v50}
IDX=${IDX:-$REF/salmon_index}
OUTROOT="$HERE/results/quant"

quant_one() {
    local sample=$1 r1=$2 r2=$3
    local out="$OUTROOT/$sample"
    if [ -f "$out/quant.sf" ]; then echo "[skip] $sample"; return 0; fi
    echo "[$(date +%H:%M:%S)] start $sample"
    mkdir -p "$out"
    local trim; trim=$(mktemp -d "$OUTROOT/_trim.$sample.XXXX")
    trap 'rm -rf "$trim"' RETURN

    fastp -i "$r1" -I "$r2" \
        -o "$trim/r1.fq.gz" -O "$trim/r2.fq.gz" \
        --detect_adapter_for_pe --thread 8 \
        --json "$out/fastp.json" --html "$out/fastp.html" \
        2> "$out/fastp.log"

    salmon quant -i "$IDX" -l A \
        -1 "$trim/r1.fq.gz" -2 "$trim/r2.fq.gz" \
        -p "${THREADS:-24}" \
        --gcBias --seqBias --posBias \
        -o "$out" > "$out/salmon.log" 2>&1

    local lt; lt=$(grep -o '"expected_format": "[^"]*"' "$out/lib_format_counts.json" 2>/dev/null | head -1)
    echo "[$(date +%H:%M:%S)] done  $sample  $lt"
}

if [ "${1:-}" = "--one" ]; then
    shift; quant_one "$@"; exit 0
fi

# ---- driver ----
JOBS=${1:-4}
export THREADS=${2:-24}
export -f quant_one
export OUTROOT IDX THREADS
[ -d "$IDX" ] || { echo "ERROR: index not found at $IDX (run 01_build_index.sh)" >&2; exit 1; }
MANIFEST="$HERE/metadata/quant_manifest.tsv"
[ -s "$MANIFEST" ] || { echo "ERROR: no manifest at $MANIFEST (run 00_build_metadata.py)" >&2; exit 1; }
mkdir -p "$OUTROOT"

tail -n +2 "$MANIFEST" \
  | xargs -P "$JOBS" -d '\n' -I{} bash -c '
        IFS=$'"'"'\t'"'"' read -r s r1 r2 <<< "{}"
        "'"$HERE"'/02_quant.sh" --one "$s" "$r1" "$r2"'

echo "ALL QUANT DONE"
