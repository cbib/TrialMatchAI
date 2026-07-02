"""Turn publicly downloadable open ontologies into concept-dictionary rows
(``<id>||<name>|<synonym>|...``) for the entity-linking concept store.

Two source kinds: ``obo`` (OBO ontologies, filtered by id prefix) and ``gene_info``
(NCBI gene_info TSV). Licensed OMOP vocabularies (SNOMED/LOINC/RxNorm) are not here —
they come from an OHDSI Athena ``CONCEPT.csv``.
"""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Iterator

import requests

from trialmatchai.utils.logging_config import setup_logging

logger = setup_logging(__name__)

_SYN_RE = re.compile(r'synonym:\s*"((?:[^"\\]|\\.)*)"')
_WS_RE = re.compile(r"\s+")
_DOWNLOAD_TIMEOUT = 1800
_CHUNK = 1 << 20


@dataclass(frozen=True)
class ConceptSource:
    """A downloadable open vocabulary and how to map it to concept rows."""

    name: str
    url: str
    kind: str  # "obo" | "gene_info"
    vocabulary_id: str
    domain_id: str
    id_prefix: str | None = None  # OBO: keep only ids with this prefix

    @property
    def filename(self) -> str:
        return self.url.rsplit("/", 1)[-1]

    @property
    def dict_filename(self) -> str:
        return f"dict_{self.domain_id}.txt"


# The default "open" source set (no auth, no licence acceptance required).
OPEN_SOURCES: dict[str, ConceptSource] = {
    "genes": ConceptSource(
        "genes",
        "https://ftp.ncbi.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz",
        "gene_info", "NCBIGene", "Gene",
    ),
    "diseases": ConceptSource(
        "diseases", "http://purl.obolibrary.org/obo/doid.obo",
        "obo", "DOID", "Disease", "DOID:",
    ),
    "chemicals": ConceptSource(
        "chemicals", "https://ftp.ebi.ac.uk/pub/databases/chebi/ontology/chebi_lite.obo",
        "obo", "ChEBI", "Chemical", "CHEBI:",
    ),
    "cell_lines": ConceptSource(
        "cell_lines", "https://ftp.expasy.org/databases/cellosaurus/cellosaurus.obo",
        "obo", "Cellosaurus", "CellLine", "CVCL_",
    ),
    "cell_types": ConceptSource(
        "cell_types", "http://purl.obolibrary.org/obo/cl.obo",
        "obo", "CL", "CellType", "CL:",
    ),
    "phenotypes": ConceptSource(
        "phenotypes", "http://purl.obolibrary.org/obo/hp.obo",
        "obo", "HPO", "Phenotype", "HP:",
    ),
}


def _open(path: Path) -> IO[str]:
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _clean_names(names: Iterator[str] | list[str]) -> list[str]:
    """Normalise whitespace, drop blanks, dedupe case-insensitively (order-preserving)."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in names:
        name = _WS_RE.sub(" ", (raw or "").replace("\\", "")).strip()
        key = name.casefold()
        if name and key not in seen:
            seen.add(key)
            out.append(name)
    return out


def parse_obo(path: Path, prefix: str | None) -> Iterator[tuple[str, list[str]]]:
    """Yield ``(id, [name, *synonyms])`` for each non-obsolete OBO term."""
    cid = name = None
    synonyms: list[str] = []
    obsolete = False
    in_term = False

    def emit() -> tuple[str, list[str]] | None:
        if cid and name and not obsolete and (not prefix or cid.startswith(prefix)):
            return cid, [name, *synonyms]
        return None

    with _open(path) as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if line.startswith("[") and line.endswith("]"):
                # Any stanza header closes the pending term; only [Term] fields are captured.
                rec = emit()
                if rec:
                    yield rec
                cid = name = None
                synonyms = []
                obsolete = False
                in_term = line == "[Term]"
            elif in_term and line.startswith("id:"):
                cid = line[3:].strip()
            elif in_term and line.startswith("name:"):
                name = line[5:].strip()
            elif in_term and line.startswith("synonym:"):
                m = _SYN_RE.search(line)
                if m:
                    synonyms.append(m.group(1))
            elif in_term and line.startswith("is_obsolete:") and "true" in line:
                obsolete = True
        rec = emit()
        if rec:
            yield rec


def parse_gene_info(path: Path) -> Iterator[tuple[str, list[str]]]:
    """Yield ``(gene_id, [symbol, full name, *synonyms, *other designations])``."""
    with _open(path) as fh:
        for raw in fh:
            if raw.startswith("#"):
                continue
            cols = raw.rstrip("\n").split("\t")
            if len(cols) < 14:
                continue
            gene_id = cols[1].strip()
            symbol = cols[2].strip()
            syns = [] if cols[4] == "-" else cols[4].split("|")
            full_name = "" if cols[8] == "-" else cols[8].strip()
            auth_full = "" if cols[11] == "-" else cols[11].strip()
            other = [] if cols[13] == "-" else cols[13].split("|")
            names = [n for n in [symbol, auth_full, full_name, *syns, *other] if n]
            if gene_id and names:
                yield gene_id, names


def _iter_source(source: ConceptSource, raw_path: Path) -> Iterator[tuple[str, list[str]]]:
    if source.kind == "obo":
        return parse_obo(raw_path, source.id_prefix)
    if source.kind == "gene_info":
        return parse_gene_info(raw_path)
    raise ValueError(f"Unknown source kind: {source.kind!r}")


def write_dictionary(source: ConceptSource, raw_path: Path, dict_path: Path) -> int:
    """Convert a downloaded source into a concept-dictionary file; returns row count.

    Temp-then-rename so a crash can't leave a truncated dict that a later run reuses.
    """
    n = 0
    tmp_path = dict_path.with_name(dict_path.name + ".part")
    with open(tmp_path, "w", encoding="utf-8") as out:
        for cid, names in _iter_source(source, raw_path):
            clean = _clean_names(names)
            if cid and clean:
                out.write(f"{cid}||{'|'.join(clean)}\n")
                n += 1
    tmp_path.replace(dict_path)
    return n


def download_source(
    source: ConceptSource, dest_dir: Path, *, force: bool = False
) -> Path | None:
    """Download a source (idempotent, atomic). Returns the path, or None on failure."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / source.filename
    if dest.exists() and dest.stat().st_size > 0 and not force:
        logger.info("cached download: %s", dest)
        return dest
    logger.info("downloading %s -> %s", source.url, dest)
    tmp = dest.with_name(dest.name + ".tmp")
    try:
        with requests.get(source.url, stream=True, timeout=_DOWNLOAD_TIMEOUT) as resp:
            resp.raise_for_status()
            with open(tmp, "wb") as handle:
                for chunk in resp.iter_content(_CHUNK):
                    handle.write(chunk)
        tmp.replace(dest)
        return dest
    except Exception:
        logger.exception("download failed: %s", source.url)
        tmp.unlink(missing_ok=True)
        return None


def build_open_dictionaries(
    work_dir: Path,
    *,
    sources: list[str] | None = None,
    force: bool = False,
) -> list[tuple[ConceptSource, Path]]:
    """Download + convert the requested open sources into concept dictionaries.

    Idempotent (reuses existing downloads/dicts unless ``force``); returns
    ``(source, dict_path)`` per non-empty dictionary, skipping failed downloads.
    """
    work_dir = Path(work_dir)
    raw_dir = work_dir / "raw"
    work_dir.mkdir(parents=True, exist_ok=True)
    names = sources or list(OPEN_SOURCES)
    results: list[tuple[ConceptSource, Path]] = []

    for key in names:
        source = OPEN_SOURCES[key]
        dict_path = work_dir / source.dict_filename
        if dict_path.exists() and dict_path.stat().st_size > 0 and not force:
            logger.info("cached dictionary: %s", dict_path)
            results.append((source, dict_path))
            continue
        raw_path = download_source(source, raw_dir, force=force)
        if raw_path is None:
            logger.warning("skipping %s (download failed)", key)
            continue
        count = write_dictionary(source, raw_path, dict_path)
        logger.info("converted %s: %s concepts -> %s", key, count, dict_path)
        if count > 0:
            results.append((source, dict_path))
        else:
            logger.warning("skipping %s (0 concepts after conversion)", key)
    return results
