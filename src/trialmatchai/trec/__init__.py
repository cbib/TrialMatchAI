"""TREC Clinical Trials evaluation harness for TrialMatchAI.

Provides an idempotent, end-to-end runner over the TREC 2021/2022 and SIGIR
patient-topic tracks: it converts the legacy patient topics into canonical
profiles, builds a per-track LanceDB search index restricted to the track's
document collection, and runs the matching pipeline with per-patient resume.

Exposed via the ``trialmatchai-trec`` console command (see
``trialmatchai.cli.trec``).
"""

from __future__ import annotations

from trialmatchai.trec.corpus import TRACK_KEYS, TrackSpec, resolve_tracks
from trialmatchai.trec.runner import run_tracks

__all__ = ["TRACK_KEYS", "TrackSpec", "resolve_tracks", "run_tracks"]
