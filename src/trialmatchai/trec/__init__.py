"""TREC Clinical Trials evaluation harness.

Idempotent end-to-end runner over the TREC 2021/2022 and SIGIR patient-topic
tracks: converts topics to canonical profiles, builds a per-track LanceDB index
restricted to the track's collection, and runs matching with per-patient resume.
Exposed via the ``trialmatchai trec`` console command.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from trialmatchai.trec.corpus import TRACK_KEYS, TrackSpec, resolve_tracks


def __getattr__(name: str) -> Any:
    """Keep the public runner import without loading the full pipeline eagerly."""
    if name == "run_tracks":
        from trialmatchai.trec.runner import run_tracks

        return run_tracks
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    from trialmatchai.trec.runner import run_tracks as run_tracks

__all__ = ["TRACK_KEYS", "TrackSpec", "resolve_tracks", "run_tracks"]
