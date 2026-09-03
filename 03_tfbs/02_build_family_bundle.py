"""Build the family-level PWM bundle from data/motif_families.txt.

For each TF listed under each family, extract the rank-0 motif from HOCOMOCO
H14CORE (via the UniProt-entry-name alias table). The composite dimer entry
(POU5F1::SOX2) is pulled from JASPAR.

Writes:
  data/pwms/family_bundle.meme  MEME-format bundle (140 motifs)
  data/pwms/family_manifest.tsv TF → family → motif_id → tier table for the
                                downstream "best driver per family" aggregation
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import polars as pl

DIR = Path(__file__).resolve().parent
FAM_FILE = DIR / "data" / "motif_families.txt"
H14_BUNDLE = DIR / "data" / "pwms" / "H14CORE_meme_format.meme"
OUT_BUNDLE = DIR / "data" / "pwms" / "family_bundle.meme"
MANIFEST = DIR / "data" / "pwms" / "family_manifest.tsv"

JASPAR_URL = "https://jaspar.elixir.no/api/v1/matrix/{}.meme"

ALIASES = {
    "POU5F1": "PO5F1", "ONECUT1": "HNF6", "ONECUT2": "ONEC2", "ONECUT3": "ONEC3",
    "NR3C1": "GCR", "NR3C2": "MCR", "AR": "ANDR", "PGR": "PRGR",
    "THRA": "THA", "THRB": "THB", "NR2F1": "COT1", "NR2F2": "COT2",
    "RAX": "RX", "RORC": "RORG",
}

HEADER = (
    "MEME version 4\n\n"
    "ALPHABET= ACGT\n\n"
    "strands: + -\n\n"
    "Background letter frequencies\n"
    "A 0.25 C 0.25 G 0.25 T 0.25\n\n"
)


def parse_family_file(path: Path) -> dict[str, list[str]]:
    families: dict[str, list[str]] = {}
    current = None
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].rstrip()
        s = line.strip()
        if not s:
            current = None; continue
        if not line.startswith("  "):
            current = s; families[current] = []
        else:
            families[current].append(s)
    return families


def index_h14(text: str) -> dict[str, list[tuple[str, str]]]:
    """Return short_name -> list of (motif_id, body) tuples for H14CORE."""
    idx: dict[str, list[tuple[str, str]]] = {}
    cur_id = None
    buf: list[str] = []
    for ln in text.splitlines():
        if ln.startswith("MOTIF "):
            if cur_id is not None:
                idx.setdefault(cur_id.split(".")[0], []).append((cur_id, "\n".join(buf).rstrip() + "\n"))
            cur_id = ln.split()[1]
            buf = [ln]
        elif cur_id is not None:
            buf.append(ln)
    if cur_id is not None:
        idx.setdefault(cur_id.split(".")[0], []).append((cur_id, "\n".join(buf).rstrip() + "\n"))
    return idx


def fetch_jaspar_body(matrix_id: str) -> str:
    url = JASPAR_URL.format(matrix_id)
    with urllib.request.urlopen(url, timeout=20) as resp:
        text = resp.read().decode()
    start = text.find(f"MOTIF {matrix_id}")
    nxt = text.find("\nMOTIF ", start + 1)
    end = nxt + 1 if nxt >= 0 else len(text)
    return text[start:end].rstrip() + "\n"


def relabel_motif_line(body: str, alt_id: str) -> str:
    """Rewrite the second token of the MOTIF line to alt_id (so AME prints it)."""
    lines = body.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith("MOTIF "):
            parts = ln.split()
            lines[i] = f"MOTIF {parts[1]} {alt_id}"
            break
    return "\n".join(lines) + "\n"


def main():
    families = parse_family_file(FAM_FILE)
    print(f"Parsed {len(families)} families")

    h14_idx = index_h14(H14_BUNDLE.read_text())
    print(f"H14CORE: {sum(len(v) for v in h14_idx.values())} motifs across {len(h14_idx)} TFs")

    out = [HEADER]
    rows = []
    n_ok = 0
    n_miss = 0
    seen_motif_ids: set[str] = set()  # AME requires unique MOTIF ids in the bundle

    for fam, items in families.items():
        for item in items:
            if "composite" in item.lower():
                # Dimer — JASPAR MA0142.1
                mid = "MA0142.1"
                alt = f"POU5F1__SOX2__dimer"
                body = fetch_jaspar_body(mid)
                body = relabel_motif_line(body, alt)
                out.append(body + "\n")
                rows.append({"family": fam, "tf": "POU5F1::SOX2", "h14_short": "(JASPAR)",
                             "motif_id": mid, "alt_id": alt, "tier": "—",
                             "source": "JASPAR_CORE_2024"})
                n_ok += 1
                continue
            tf = item.strip()
            short = ALIASES.get(tf, tf)
            mids = h14_idx.get(short, [])
            if not mids:
                print(f"  [MISSING] {fam} / {tf} -> {short}")
                n_miss += 1
                continue
            # Prefer rank-0 motif
            rank0 = [(m, b) for m, b in mids if ".0." in m]
            mid, body = (rank0[0] if rank0 else mids[0])
            if mid in seen_motif_ids:
                # A TF appearing in multiple families would re-add the same motif —
                # we skip the duplicate body but still record the family→tf mapping.
                rows.append({"family": fam, "tf": tf, "h14_short": short,
                             "motif_id": mid, "alt_id": mid.split('.')[0],
                             "tier": ".".join(mid.rsplit(".", 2)[1:]),
                             "source": "HOCOMOCO_H14CORE"})
                continue
            # Use the H14CORE short name as alt-id so AME prints it
            alt = short
            body = relabel_motif_line(body, alt)
            out.append(body + "\n")
            seen_motif_ids.add(mid)
            tier = ".".join(mid.rsplit(".", 2)[1:])
            rows.append({"family": fam, "tf": tf, "h14_short": short,
                         "motif_id": mid, "alt_id": alt, "tier": tier,
                         "source": "HOCOMOCO_H14CORE"})
            n_ok += 1

    OUT_BUNDLE.parent.mkdir(parents=True, exist_ok=True)
    OUT_BUNDLE.write_text("".join(out))
    pl.DataFrame(rows).write_csv(MANIFEST, separator="\t")
    print(f"\n-> {OUT_BUNDLE}  ({OUT_BUNDLE.stat().st_size//1024} KB, {n_ok} motifs)")
    print(f"-> {MANIFEST}  ({len(rows)} (family, TF) pairs)")
    print(f"   {n_miss} unresolved (should be 0)")


if __name__ == "__main__":
    main()
