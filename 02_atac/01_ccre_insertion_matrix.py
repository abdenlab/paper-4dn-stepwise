# /// script
# requires-python = ">=3.11"   # giql requires 3.11+
# dependencies = ["duckdb", "oxbow", "pyarrow", "giql"]
# ///
"""Build a cCRE x sample Tn5 INSERTION-SITE count matrix for ATAC-seq.

ATAC-native read-overlap counting: each fragment is two independent Tn5 cut
sites (one per read end), with the +4/-5 strand-aware shift correcting for
Tn5's 9bp staggered cut. We count cCREs whose interval contains each shifted
insertion site.

5'-end insertion site (1-bp interval) in BED 0-based half-open:
    + strand (flag & 16 == 0): [pos + 3, pos + 4)
    - strand (flag & 16 != 0): [end - 6, end - 5)

Derivation:
* BED start = pos - 1; +4 shift on + 5' end -> pos + 3.
* BED last base = end - 1; -5 shift on - 5' end -> end - 6.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import duckdb
import oxbow as ox
import pyarrow as pa
from giql import Table, transpile

DIR = Path(__file__).resolve().parent
DATA = DIR / "data"
CCRE_PATH = str(DATA / "GRCh38-cCREs.bed")

# One symlink per library in data/bams/, named <NN>_<STAGE>.R<n>.bam.
SAMPLES = [
    "01_ESC.R1",  "01_ESC.R2",
    "02_DE.R1",   "02_DE.R2",
    "03_HB.R1",   "03_HB.R2",
    "04_iHEP.R1", "04_iHEP.R2",
    "05_mHEP.R1", "05_mHEP.R2",
]
ALIGNMENTS = {s: str(DATA / "bams" / f"{s}.bam") for s in SAMPLES}

OUT_PATH = DIR / "results" / "ccre_insertion_matrix.parquet"


def _safe(name: str) -> str:
    return name.replace(".", "_")


def compute_sample_insertions(
    conn: duckdb.DuckDBPyConnection, sample: str, bam_path: str
) -> tuple[str, pa.Table]:
    """Pure worker: open a cursor, derive Tn5 insertion sites from BAM,
    spatial-join with cCREs, return (ccre_id, n) PyArrow Table."""
    s = _safe(sample)
    cur = conn.cursor()
    raw = ox.from_bam(
        bam_path, fields=["rname", "pos", "end", "flag"]
    ).to_duckdb(cur)
    cur.register(f"reads_raw_{s}", raw)
    # 1-bp interval at the shifted 5' end.
    cur.execute(
        f"""
        CREATE TEMP VIEW insertions_{s} AS
        SELECT
            rname,
            CASE WHEN (flag & 16) = 0 THEN pos + 3 ELSE "end" - 6 END AS start,
            CASE WHEN (flag & 16) = 0 THEN pos + 4 ELSE "end" - 5 END AS end
        FROM reads_raw_{s}
        """
    )
    rt = Table(
        f"insertions_{s}",
        chrom_col="rname", start_col="start", end_col="end", strand_col=None,
    )
    sql = transpile(
        f"""
        SELECT c.ccre_id, COUNT(r.start) AS n
        FROM ccres c
        LEFT JOIN insertions_{s} r ON c.interval INTERSECTS r.interval
        GROUP BY c.ccre_id
        """,
        tables=["ccres", rt],
        dialect="duckdb",
    )
    return sample, cur.sql(sql).to_arrow_table()


def main() -> None:
    conn = duckdb.connect()

    ccres_oxbow = ox.from_bed(
        CCRE_PATH,
        bed_schema=("bed3", {"dhs_id": "string", "ccre_id": "string", "ccre_class": "string"}),
    ).to_duckdb(conn)
    conn.register("ccres_oxbow", ccres_oxbow)
    conn.execute("CREATE TABLE ccres AS SELECT * FROM ccres_oxbow")

    t_par = time.time()
    with ThreadPoolExecutor(max_workers=len(ALIGNMENTS)) as ex:
        futures = [
            ex.submit(compute_sample_insertions, conn, s, bam)
            for s, bam in ALIGNMENTS.items()
        ]
        sample_tables: dict[str, pa.Table] = {}
        for fut in futures:
            sample, tbl = fut.result()
            sample_tables[sample] = tbl
    print(f"All per-sample joins done in {time.time() - t_par:.1f}s wall")

    for sample, tbl in sample_tables.items():
        conn.register(f"counts_{_safe(sample)}", tbl)

    sample_cols = ",\n          ".join(
        f'"counts_{_safe(s)}".n AS "{s}"' for s in ALIGNMENTS
    )
    join_clauses = "\n        ".join(
        f'JOIN "counts_{_safe(s)}" USING (ccre_id)' for s in ALIGNMENTS
    )
    final_sql = f"""
        SELECT
          c.chrom, c.start, c."end",
          c.dhs_id, c.ccre_id, c.ccre_class,
          {sample_cols}
        FROM ccres c
        {join_clauses}
        ORDER BY c.chrom, c.start, c."end"
    """

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    conn.sql(final_sql).write_parquet(str(OUT_PATH))
    print(f"Wrote matrix in {time.time() - t0:.1f}s")

    n_total = conn.sql(f"SELECT COUNT(*) FROM '{OUT_PATH}'").fetchone()[0]
    print(f"Wrote {n_total:,} cCREs x {len(ALIGNMENTS)} samples -> {OUT_PATH}")


if __name__ == "__main__":
    main()
