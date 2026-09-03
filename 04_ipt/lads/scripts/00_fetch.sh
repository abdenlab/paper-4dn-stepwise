#!/usr/bin/env bash
# Fetch human LAD maps + SPIN states + constitutive-LAD reference for the IPT↔LAD comparison.
# All primary LAD/SPIN products are hg38 → NO liftover needed. Chains + liftOver are installed
# anyway (for any future hg18/hg19 beds). Run from analysis/lads/.
#
# Tooling: liftOver + bedtools via `pixi global install -c bioconda -c conda-forge ucsc-liftover bedtools`
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p raw hg38

echo "## 1. liftOver chain files (UCSC) — for any future hg18/hg19 beds"
for c in hg18ToHg38 hg19ToHg38; do
  curl -s -o raw/${c}.over.chain.gz "https://hgdownload.soe.ucsc.edu/goldenPath/${c%To*}/liftOver/${c}.over.chain.gz"
done

echo "## 2. SPIN K562 states (hg38, 25kb) — lamina/speckle/nucleolus 10-state annotation"
# Wang...van Steensel, Belmont, Ma 2021, Genome Biol (10.1186/s13059-020-02253-3); Google Drive link from SPIN README
curl -sL "https://drive.google.com/uc?export=download&id=1gdwtrhTctddO9TCBXBaZpZFOAHWCUTli" -o hg38/SPIN_K562_states.hg38.bed

echo "## 3. Human LMNB1 LAD domain calls, 50kb, hg38 (4DN, AWS Open Data S3)"
# Meuleman/van Steensel + 4DN pA-DamID/DamID-seq; "<line>_LMNB1 50kb bin associated domains (LADs), mean of replicates".
# accession -> resolve open_data_url (public S3, no auth) -> download. NB the /@@download/ API path is 403 (needs key); use open_data_url.
declare -A LAD=( [H1hESC]=4DNFIJXADI29 [HFFc6]=4DNFIT9W77EE [K562]=4DNFIJHD22QE \
                 [HCT116]=4DNFIA2LBQCD [HAP1]=4DNFIDFCY3JN [RPEhTERT]=4DNFIUVTO2H3 [U2OS]=4DNFIZ3JHKWC )
for name in "${!LAD[@]}"; do
  acc=${LAD[$name]}
  url=$(curl -s -H "Accept: application/json" "https://data.4dnucleome.org/files-processed/${acc}/?format=json" \
        | python3 -c "import sys,json;print(json.load(sys.stdin)['open_data_url'])")
  curl -sL "$url" -o raw/LAD_${name}_LMNB1_50kb_${acc}.bed.gz
  zcat raw/LAD_${name}_LMNB1_50kb_${acc}.bed.gz | cut -f1-3 > hg38/LAD_${name}_LMNB1_50kb.hg38.bed
done
