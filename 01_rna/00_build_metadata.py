"""Build the RNA-seq sample metadata table.

Two experiments:
  - 5stage : H1-derived ESC->DE->HB->iHEP->mHEP, 2 biological reps (Rick1/Rick2).
             DNBSEQ strand-specific mRNA.
  - 8day   : H9-derived DE->HB daily time course (DE+1 .. DE+7, HB). Libraries were
             prepped for SLAM-seq with 4sU but isolated as ordinary RNA (no alkylation,
             no pulldown) -> quantify as standard poly-A RNA. p3C is the minus-4sU
             control twin of p3 (both DE+3)

External input : the two FASTQ directories, symlinked under data/ --
                 data/rna_5stage (H1 5-stage) and data/rna_8day (H9 time course).
Writes         : metadata/samples.tsv        one row per library
                 metadata/quant_manifest.tsv sample + FASTQ pair
"""

from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
RNA5 = HERE / "data" / "rna_5stage"
RNA8 = HERE / "data" / "rna_8day"
QUANT = HERE / "results" / "quant"
MD = HERE / "metadata"
MD.mkdir(parents=True, exist_ok=True)

# ---- 5-stage (H1) -------------------------------------------------------------
stages5 = ["ESC", "DE", "HB", "iHEP", "mHEP"]
cultures = {"REP1": "Rick1", "REP2": "Rick2"}
rows = []
for st in stages5:
    for rep in ("REP1", "REP2"):
        s = f"{st}_{rep}"
        rows.append(
            dict(
                sample=s,
                experiment="5stage",
                cell_line="H1",
                stage=st,
                timepoint_day=None,
                biorep=rep.replace("REP", "R"),
                culture=cultures[rep],
                library_kit="DNBSEQ_stranded_mRNA",
                adapter="Nextera",
                read_len=100,
                tresU=False,
                in_trajectory=True,
                fastq_1=str(RNA5 / f"{s}_1.fq.gz"),
                fastq_2=str(RNA5 / f"{s}_2.fq.gz"),
            )
        )

# ---- 8-day (H9) ---------------------------------------------------------------
# file stem -> (sample, day, in_trajectory, tresU)
h9 = [
    ("DE-1_S11", "H9_DE-p1", 1, True, True),
    ("DE-2_S12", "H9_DE-p2", 2, True, True),
    ("DE-3_S13", "H9_DE-p3", 3, True, True),
    ("DE-3C_S14", "H9_DE-p3C", 3, False, False),  # minus-4sU control twin of p3
    ("DE-4_S15", "H9_DE-p4", 4, True, True),
    ("DE-5_S16", "H9_DE-p5", 5, True, True),
    ("DE-6_S17", "H9_DE-p6", 6, True, True),
    ("DE-7_S18", "H9_DE-p7", 7, True, True),
    ("HB_S19", "H9_HB", 8, True, True),  # endpoint (HB)
]
for stem, s, day, intraj, t4su in h9:
    stage = "HB" if s == "H9_HB" else "DE_to_HB"
    rows.append(
        dict(
            sample=s,
            experiment="8day",
            cell_line="H9",
            stage=stage,
            timepoint_day=day,
            biorep="R1",
            culture=None,
            library_kit="SLAMprep_TruSeq",
            adapter="TruSeq",
            read_len=150,
            tresU=t4su,
            in_trajectory=intraj,
            fastq_1=str(RNA8 / f"{stem}_L001_R1_001.fastq.gz"),
            fastq_2=str(RNA8 / f"{stem}_L001_R2_001.fastq.gz"),
        )
    )

df = pl.DataFrame(rows).with_columns(
    strand_inferred=pl.lit(None, dtype=pl.Utf8),  # filled after RSeQC
    quant_dir=pl.col("sample").map_elements(
        lambda s: str(QUANT / s), return_dtype=pl.Utf8
    ),
)


df.write_csv(MD / "samples.tsv", separator="\t")
df.select("sample", "fastq_1", "fastq_2").write_csv(
    MD / "quant_manifest.tsv", separator="\t"
)

missing = [f for f in (*df["fastq_1"], *df["fastq_2"]) if not Path(f).exists()]
print(f"wrote metadata/samples.tsv + metadata/quant_manifest.tsv ({df.height} libraries)")
if missing:
    print(f"WARNING: {len(missing)} FASTQ(s) not found, e.g. {missing[0]}")
