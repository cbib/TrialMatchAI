"""Tests for the fine-tuning data/CLI layer (no heavy deps or model loads)."""

from __future__ import annotations

import json

import pytest

from trialmatchai.finetuning.cli import build_parser
from trialmatchai.finetuning.config import FinetuneConfig
from trialmatchai.finetuning.data import (
    cot_row_to_messages,
    ner_row_to_gliner,
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


def test_ner_char_spans_convert_to_token_indices():
    gliner = ner_row_to_gliner(
        {"text": "EGFR mutation positive", "ner": [[0, 4, "gene"]]}
    )
    assert gliner["tokenized_text"][0] == "EGFR"
    assert gliner["ner"] == [[0, 0, "gene"]]


def test_ner_native_format_passthrough():
    row = {"tokenized_text": ["A", "B"], "ner": [[0, 1, "x"]]}
    assert ner_row_to_gliner(row) == row


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
