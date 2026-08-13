"""The TREC funnel preset must not silently overwrite an explicit config.

_track_config used to assign the funnel unconditionally, so a config asking for
max_trials_second_level=1000 still ran at 500 and a run configured for a 600-trial shortlist
still got 250. Funnel experiments measured the preset instead of the variable under test.
"""

from trialmatchai.trec.corpus import TrackSpec
from trialmatchai.trec.runner import _track_config


def _spec(tmp_path):
    return TrackSpec(
        key="23",
        id_prefix="trec-2023",
        db_path=tmp_path / "search_23",
        profile_dir=tmp_path / "profiles",
        summary_dir=tmp_path / "summaries",
        output_dir=tmp_path / "out",
        trec_dir=tmp_path / "trec",
    )


def test_preset_applies_when_the_config_is_at_schema_defaults(tmp_path):
    """A config that expresses no opinion still gets the TREC funnel."""
    cfg = _track_config(
        {"search": {"max_trials_second_level": 100}, "rag": {"max_trials_rag": 20}},
        _spec(tmp_path),
    )
    assert cfg["search"]["max_trials_second_level"] == 500
    assert cfg["rag"]["max_trials_rag"] == 250
    assert cfg["search"]["second_level_keep_divisor"] == 1


def test_explicit_second_level_width_is_respected(tmp_path):
    """The regression that made the width A/B unrunnable: 1000 was silently forced to 500."""
    cfg = _track_config({"search": {"max_trials_second_level": 1000}, "rag": {}}, _spec(tmp_path))
    assert cfg["search"]["max_trials_second_level"] == 1000


def test_explicit_shortlist_size_is_respected(tmp_path):
    """The regression that made the depth run measure nothing: 600 was silently forced to 250."""
    cfg = _track_config({"search": {}, "rag": {"max_trials_rag": 600}}, _spec(tmp_path))
    assert cfg["rag"]["max_trials_rag"] == 600


def test_explicit_keep_divisor_is_respected(tmp_path):
    cfg = _track_config({"search": {"second_level_keep_divisor": 5}, "rag": {}}, _spec(tmp_path))
    assert cfg["search"]["second_level_keep_divisor"] == 5


def test_second_level_width_knobs_survive(tmp_path):
    """search.second_level is what actually widens retrieval; the preset must not touch it."""
    cfg = _track_config(
        {"search": {"second_level": {"per_query_size": 1000, "aggregation_threshold": 0.2}}, "rag": {}},
        _spec(tmp_path),
    )
    assert cfg["search"]["second_level"]["per_query_size"] == 1000
    assert cfg["search"]["second_level"]["aggregation_threshold"] == 0.2


def test_track_paths_are_still_swapped_in(tmp_path):
    cfg = _track_config({"search": {}, "rag": {}}, _spec(tmp_path))
    assert cfg["search_backend"]["db_path"] == str(tmp_path / "search_23")
    assert cfg["paths"]["output_dir"] == str(tmp_path / "out")
    assert cfg["query_expansion"]["enabled"] is True
    assert cfg["reporting"]["emit_html"] is False


def test_base_config_is_not_mutated(tmp_path):
    base = {"search": {"max_trials_second_level": 1000}, "rag": {}}
    _track_config(base, _spec(tmp_path))
    assert base["search"] == {"max_trials_second_level": 1000}
    assert base["rag"] == {}
