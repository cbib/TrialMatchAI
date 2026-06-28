"""Tests for the fine-tuning data/CLI layer (no heavy deps or model loads)."""

from __future__ import annotations

import json

import pytest

from trialmatchai.finetuning.cli import _lora_config_from_args, build_parser
from trialmatchai.finetuning.config import FinetuneConfig
from trialmatchai.finetuning.data import (
    cot_row_to_messages,
    gliner2_row_to_training_record,
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


def test_ner_native_gliner2_text_schema_format_and_negative_entities():
    out = ner_row_to_entities(
        {
            "text": "No reportable biomarker",
            "schema": {
                "entities": {"gene": []},
                "entity_descriptions": {"gene": "Gene symbols"},
            },
        }
    )
    assert out["text"] == "No reportable biomarker"
    assert out["entities"] == {"gene": []}
    assert out["entity_descriptions"] == {"gene": "Gene symbols"}


def test_gliner2_schema_record_preserves_json_structures():
    record = gliner2_row_to_training_record(
        {
            "text": "Patient has EGFR exon 19 deletion and stage IV NSCLC.",
            "schema": {
                "entities": {"gene": ["EGFR"]},
                "json_structures": [
                    {
                        "biomarker": {
                            "gene": "EGFR",
                            "variant": "exon 19 deletion",
                            "disease_stage": "stage IV",
                        }
                    }
                ],
                "json_descriptions": {
                    "biomarker": {
                        "gene": "Gene symbol",
                        "variant": "Observed alteration",
                    }
                },
            },
        }
    )
    assert record == {
        "input": "Patient has EGFR exon 19 deletion and stage IV NSCLC.",
        "output": {
            "entities": {"gene": ["EGFR"]},
            "json_structures": [
                {
                    "biomarker": {
                        "gene": "EGFR",
                        "variant": "exon 19 deletion",
                        "disease_stage": "stage IV",
                    }
                }
            ],
            "json_descriptions": {
                "biomarker": {
                    "gene": "Gene symbol",
                    "variant": "Observed alteration",
                }
            },
        },
    }


def test_gliner2_record_preserves_classifications_and_relations():
    row = {
        "input": "Erlotinib targets EGFR.",
        "output": {
            "classifications": [
                {
                    "task": "actionability",
                    "labels": ["actionable", "not_actionable"],
                    "true_label": "actionable",
                }
            ],
            "relations": [{"targets": {"head": "Erlotinib", "tail": "EGFR"}}],
        },
    }
    assert gliner2_row_to_training_record(row) == row


def test_gliner2_top_level_structures_alias_normalizes_to_json_structures():
    record = gliner2_row_to_training_record(
        {
            "text": "Trial requires ECOG 0-1.",
            "structures": {"eligibility": {"performance_status": "ECOG 0-1"}},
        }
    )
    assert record["output"]["json_structures"] == [
        {"eligibility": {"performance_status": "ECOG 0-1"}}
    ]


def test_read_jsonl_respects_max(tmp_path):
    path = tmp_path / "d.jsonl"
    path.write_text("\n".join(json.dumps({"i": i}) for i in range(5)))
    assert len(read_jsonl(str(path), max_examples=2)) == 2
    assert read_jsonl(str(path), max_examples=0) == []


def test_read_jsonl_reports_invalid_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"ok": true}\nnot-json\n')
    with pytest.raises(ValueError, match="bad.jsonl:2"):
        read_jsonl(str(path))


def test_cli_parses_each_subcommand():
    parser = build_parser()
    for component in ("cot", "reranker", "ner"):
        args = parser.parse_args(
            [component, "--base-model", "m", "--train-data", "t", "--output-dir", "o"]
        )
        assert args.component == component

    merge_args = parser.parse_args(
        ["merge", "--base-model", "b", "--adapter", "a", "--output-dir", "o"]
    )
    assert merge_args.component == "merge" and merge_args.adapter == "a"

    with pytest.raises(SystemExit):
        parser.parse_args(["cot"])  # missing required args


def test_lora_cli_maps_training_controls():
    parser = build_parser()
    args = parser.parse_args(
        [
            "cot",
            "--base-model",
            "m",
            "--train-data",
            "t",
            "--output-dir",
            "o",
            "--eval-data",
            "e",
            "--target-modules",
            "q_proj,v_proj",
            "--eval-steps",
            "100",
            "--save-steps",
            "500",
            "--device-map",
            "none",
        ]
    )
    cfg = _lora_config_from_args(args)
    assert cfg.eval_data == "e"
    assert cfg.target_modules == ["q_proj", "v_proj"]
    assert cfg.eval_steps == 100
    assert cfg.save_steps == 500
    assert cfg.device_map is None


def test_finetune_config_training_args_lazy():
    cfg = FinetuneConfig(base_model="m", train_data="t", output_dir="o")
    assert cfg.lora_rank == 32 and cfg.bf16 is True
    assert cfg.target_modules == "all-linear"
