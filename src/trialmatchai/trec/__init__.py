"""TREC Clinical Trials evaluation harness.

Idempotent end-to-end runner over the TREC 2021/2022 and SIGIR patient-topic
tracks: converts topics to canonical profiles, builds a per-track LanceDB index
restricted to the track's collection, and runs matching with per-patient resume.
Exposed via the ``trialmatchai trec`` console command.
"""

from __future__ import annotations

from trialmatchai.trec.corpus import TRACK_KEYS, TrackSpec, resolve_tracks
from trialmatchai.trec.runner import run_tracks

__all__ = ["TRACK_KEYS", "TrackSpec", "resolve_tracks", "run_tracks"]
