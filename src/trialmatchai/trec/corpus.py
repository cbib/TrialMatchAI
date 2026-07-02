"""TREC track definitions and per-track corpus resolution.

The per-track corpus (the set of trials the index is restricted to) is derived
at run time from the official qrels — see ``trialmatchai.trec.qrels`` — not from
any checked-in NCT list.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# "sigir" has no official direct source (CSIRO portal), so it is not in the default run set.
TRACK_KEYS = ("21", "22", "sigir")
DEFAULT_TRACKS = ("21", "22")

# Topic N becomes "<prefix>N", matching the qrels query ids.
_ID_PREFIX = {"21": "trec-2021", "22": "trec-2022", "sigir": "sigir-2014"}


@dataclass(frozen=True)
class TrackSpec:
    """Everything needed to run one TREC track end-to-end."""

    key: str
    id_prefix: str
    trec_dir: Path
    profile_dir: Path
    summary_dir: Path
    db_path: Path
    output_dir: Path

    @property
    def name(self) -> str:
        return f"trec{self.key}"


def resolve_tracks(
    keys: list[str],
    *,
    data_dir: Path,
    results_root: Path,
) -> list[TrackSpec]:
    """Build :class:`TrackSpec` objects for the requested track keys.

    Layout under ``data_dir``: trec/ (topics + qrels cache),
    patients/trec<key>/{profiles,summaries}, search_<key> (index);
    results under ``results_root``/results_trec<key>.
    """
    data_dir = Path(data_dir)
    results_root = Path(results_root)
    specs: list[TrackSpec] = []
    for key in keys:
        if key not in TRACK_KEYS:
            raise ValueError(f"Unknown TREC track '{key}' (expected one of {TRACK_KEYS})")
        specs.append(
            TrackSpec(
                key=key,
                id_prefix=_ID_PREFIX[key],
                trec_dir=data_dir / "trec",
                profile_dir=data_dir / "patients" / f"trec{key}" / "profiles",
                summary_dir=data_dir / "patients" / f"trec{key}" / "summaries",
                db_path=data_dir / f"search_{key}",
                output_dir=results_root / f"results_trec{key}",
            )
        )
    return specs
