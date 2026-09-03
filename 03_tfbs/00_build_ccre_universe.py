"""Build the cCRE universe: peak-supported ENCODE cCREs across the 5 stages.

For each ENCODE cCRE (v4, GRCh38), take the maximum fold-change of the MACS2
IDR-optimal peaks intersecting it in each of the 5 differentiation stages, and
keep every cCRE hit in at least one stage.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import oxbow as ox
from giql import transpile

DIR = Path(__file__).resolve().parent
OUT_PATH = DIR / "data" / "ccre_universe.parquet"

# ENCODE cCRE v4 BED, from https://downloads.wenglab.org/V3/GRCh38-cCREs.bed
CCRE_PATH = str(DIR / "data" / "GRCh38-cCREs.bed")
# ENCODE ATAC pipeline output tree, one subdirectory per stage
PEAKS = DIR / "data" / "atac_peaks"
PEAK_FILES = {
    stage: str(PEAKS / stage / "peak" / "idr_reproducibility" / "idr.optimal_peak.narrowPeak.gz")
    for stage in ("01_ESC", "02_DE", "03_HB", "04_iHEP", "05_mHEP")
}


def main() -> None:
    conn = duckdb.connect()

    # cCRE BED is bed3 + (DHS accession, cCRE accession, classification).
    ccres = ox.from_bed(
        CCRE_PATH,
        bed_schema=("bed3", {"dhs_id": "string", "ccre_id": "string", "ccre_class": "string"}),
    ).to_duckdb(conn)
    conn.register("ccres", ccres)

    # narrowPeak: bed6 + signalValue/pValue/qValue/summit-offset. Only
    # fold_change is aggregated.
    narrowpeak_schema = (
        "bed6",
        {"fold_change": "f64", "neglog10p": "f64", "neglog10q": "f64", "summit": "i32"},
    )

    for stage, path in PEAK_FILES.items():
        peaks_name = f"peaks_{stage}"
        peaks_rel = ox.from_bed(
            path, compression="gzip", bed_schema=narrowpeak_schema
        ).to_duckdb(conn)
        conn.register(peaks_name, peaks_rel)
        sql = transpile(
            f"""
            SELECT c.ccre_id, MAX(p.fold_change) AS max_fc
            FROM ccres c
            JOIN {peaks_name} p ON c.interval INTERSECTS p.interval
            GROUP BY c.ccre_id
            """,
            tables=["ccres", peaks_name],
            dialect="duckdb",   # per-chromosome IE-join; see 02_atac/01 for why
        )
        conn.execute(f'CREATE TEMP TABLE "hits_{stage}" AS {sql}')
        n = conn.sql(f'SELECT COUNT(*) FROM "hits_{stage}"').fetchone()[0]
        print(f"{stage}: {n:,} cCREs intersect a peak")

    # Left-join the per-stage max-fc tables, keep any-hit cCREs, sort by
    # coordinates (preserves the cCRE file's lexicographic chrom order:
    # chr1, chr10, ..., chr2, ..., chrX, chrY).
    fc_cols = ",\n          ".join(f'h_{s}.max_fc AS "{s}"' for s in PEAK_FILES)
    join_clauses = "\n        ".join(
        f'LEFT JOIN "hits_{s}" h_{s} USING (ccre_id)' for s in PEAK_FILES
    )
    where_any = " OR ".join(f'"{s}" IS NOT NULL' for s in PEAK_FILES)

    final_sql = f"""
        WITH flagged AS (
          SELECT
            c.chrom, c.start, c."end", c.dhs_id, c.ccre_id, c.ccre_class,
          {fc_cols}
          FROM ccres c
          {join_clauses}
        )
        SELECT *
        FROM flagged
        WHERE {where_any}
        ORDER BY chrom, start, "end"
    """

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn.sql(final_sql).write_parquet(str(OUT_PATH))

    n_total = conn.sql(f"SELECT COUNT(*) FROM '{OUT_PATH}'").fetchone()[0]
    print(f"Wrote {n_total:,} cCREs -> {OUT_PATH}")


if __name__ == "__main__":
    main()
