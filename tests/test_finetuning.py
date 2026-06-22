"""Tests for the fine-tuning data/CLI layer (no heavy deps or model loads)."""

from __future__ import annotations

import json

import pytest

from trialmatchai.finetuning.cli import build_parser
from trialmatchai.finetuning.config import FinetuneConfig
from trialmatchai.finetuning.data import (
    cot_row_to_messages,
    ner_row_to_entities,
    read_jsonl,
    reranker_row_to_messages,
)


def test_cot_row_messages_passthrough_and_instruct():
    msgs = cot_row_to_messages({"messages": [{"role": "user", "content": "hi"}]})
    assert msgs == [{"role": "user", "content": "hi"}]

    converted = cot_row_to_messages(
        {"instruction": "sys", "input": "q", "output": "a"}
    )
    assert converted[0] == {"role": "system", "content": "sys"}
    assert converted[-1] == {"role": "assistant", "content": "a"}


def test_reranker_row_messages_and_label():
    messages, label = reranker_row_to_messages(
        {"patient_text": "P", "criterion": "C", "label": "relevant"}
    )
    assert label == "Yes"  # normalized from "relevant"
    # Reuses the runtime reranker prompt (Statement A / Statement B).
    assert any("Statement A: P" in m["content"] for m in messages)
    assert any("Statement B: C" in m["content"] for m in messages)


def test_ner_char_spans_convert_to_surface_forms():
    out = ner_row_to_entities(
        {"text": "EGFR mutation in NSCLC", "ner": [[0, 4, "gene"], [17, 22, "disease"]]}
    )
    assert out["text"] == "EGFR mutation in NSCLC"
    assert out["entities"] == {"gene": ["EGFR"], "disease": ["NSCLC"]}


def test_ner_entities_mapping_passthrough():
    out = ner_row_to_entities(
        {"text": "EGFR positive", "entities": {"gene": ["EGFR"]}}
    )
    assert out["entities"] == {"gene": ["EGFR"]}


def test_ner_native_gliner2_format():
    out = ner_row_to_entities(
        {"input": "EGFR positive", "output": {"entities": {"gene": ["EGFR"]}}}
    )
    assert out["text"] == "EGFR positive"
    assert out["entities"] == {"gene": ["EGFR"]}


def test_read_jsonl_respects_max(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text("\n".join(json.dumps({"i": i}) for i in range(5)))
    assert len(read_jsonl(str(path), max_examples=2)) == 2


def test_cli_parses_each_subcommand():
    parser = build_parser()
    for component in ("cot", "reranker", "ner"):
        args = parser.parse_args(
            [component, "--base-model", "m", "--train-data", "t", "--output-dir", "o"]
        )
        assert args.component == component

    with pytest.raises(SystemExit):
        parser.parse_args(["cot"])  # missing required args


def test_finetune_config_training_args_lazy():
    cfg = FinetuneConfig(base_model="m", train_data="t", output_dir="o")
    assert cfg.lora_rank == 32 and cfg.bf16 is True
