"""Dataset loading and prompt formatting for fine-tuning.

Training prompts reuse the exact runtime prompt builders so a fine-tuned model
sees the same format at train and inference time.

Input formats (JSONL, one object per line):
- CoT:      {"messages": [{"role","content"}, ...]}
            or {"instruction": str, "input": str, "output": str}
- Reranker: {"patient_text": str, "criterion": str, "label": "Yes"|"No"}
- GLiNER2: {"text": str, "entities": {"label": ["surface form", ...]}}
            or char-span {"text": str, "ner": [[start_char, end_char, "label"], ...]}
            or native {"input": str, "output": {"entities": {...}, "json_structures": [...]}}
            or native {"text": str, "schema": {"json_structures": [...]}}
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def read_jsonl(path: str, max_examples: Optional[int] = None) -> List[Dict[str, Any]]:
    if max_examples is not None and max_examples <= 0:
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object in {path}:{line_no}.")
            rows.append(row)
            if max_examples is not None and len(rows) >= max_examples:
                break
    return rows


# --------------------------------------------------------------------------- CoT


def cot_row_to_messages(row: Dict[str, Any]) -> List[Dict[str, str]]:
    """Normalize a CoT training row to a chat-message list."""
    if "messages" in row:
        return [
            {"role": str(message["role"]), "content": str(message["content"])}
            for message in row["messages"]
        ]
    instruction = (row.get("instruction") or "").strip()
    user = (row.get("input") or row.get("question") or "").strip()
    output = row.get("output") or row.get("answer") or ""
    messages: List[Dict[str, str]] = []
    if instruction:
        messages.append({"role": "system", "content": instruction})
    messages.append({"role": "user", "content": user})
    messages.append({"role": "assistant", "content": output})
    return messages


# ----------------------------------------------------------------------- Reranker


def reranker_row_to_messages(row: Dict[str, Any]) -> tuple[List[Dict[str, str]], str]:
    """Build the reranker prompt (matching LLMReranker.create_messages) + target."""
    from trialmatchai.models.llm.llm_reranker import LLMReranker

    patient_text = str(row.get("patient_text", "")).strip()
    criterion = str(row.get("criterion", row.get("trial_text", ""))).strip()
    label = str(row.get("label", "")).strip() or "No"
    if label not in {"Yes", "No"}:
        label = "Yes" if label.lower() in {"1", "true", "yes", "relevant"} else "No"
    messages = LLMReranker.create_messages(patient_text, criterion)
    return messages, label


# ----------------------------------------------------------------------- GLiNER2


def _entity_mapping(value: Any) -> Dict[str, List[str]]:
    entities: Dict[str, List[str]] = {}
    for label, surfaces in dict(value or {}).items():
        if surfaces is None:
            entities[str(label)] = []
        elif isinstance(surfaces, str):
            entities[str(label)] = [surfaces]
        else:
            entities[str(label)] = [str(surface) for surface in surfaces]
    return entities


def _copy_if_present(source: Dict[str, Any], target: Dict[str, Any], key: str) -> None:
    if key in source and source[key] is not None:
        target[key] = source[key]


def _json_structures(value: Any) -> Any:
    if isinstance(value, dict):
        return [{name: fields} for name, fields in value.items()]
    return value


def gliner2_row_to_training_record(row: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize supported GLiNER2 row shapes to ``{"input", "output"}``.

    The output schema is intentionally close to GLiNER2's native format so
    schema-based extraction tasks (JSON structures, classifications, relations)
    pass through to ``InputExample.from_dict`` without being flattened.
    """
    if "input" in row and "output" in row:
        text = str(row["input"])
        source = dict(row.get("output") or {})
    else:
        text = str(row.get("text", ""))
        source = dict(row.get("schema") or {})
        for key in (
            "entities",
            "entity_descriptions",
            "classifications",
            "json_structures",
            "structures",
            "json_descriptions",
            "relations",
        ):
            _copy_if_present(row, source, key)

    output: Dict[str, Any] = {}
    if "entities" in source:
        output["entities"] = _entity_mapping(source.get("entities"))
    if "ner" in row and "entities" not in output:
        output["entities"] = _entities_from_spans(text, row.get("ner") or [])
    _copy_if_present(source, output, "entity_descriptions")
    _copy_if_present(source, output, "classifications")
    if "json_structures" in source:
        output["json_structures"] = _json_structures(source["json_structures"])
    elif "structures" in source:
        output["json_structures"] = _json_structures(source["structures"])
    _copy_if_present(source, output, "json_descriptions")
    _copy_if_present(source, output, "relations")
    return {"input": text, "output": output}


def _entities_from_spans(text: str, spans: Any) -> Dict[str, List[str]]:
    entities: Dict[str, List[str]] = {}
    for span in spans:
        if not isinstance(span, (list, tuple)):
            continue
        if len(span) < 3:
            continue
        start_char, end_char, label = span[0], span[1], span[2]
        if not isinstance(start_char, int) or not isinstance(end_char, int):
            continue
        surface = text[start_char:end_char].strip()
        if surface:
            label = str(label)
            entities.setdefault(label, [])
            if surface not in entities[label]:
                entities[label].append(surface)
    return entities


def ner_row_to_entities(row: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize an NER row to GLiNER2's surface-form schema.

    Returns ``{"text", "entities": {label: [surface forms]}, "entity_descriptions"}``.
    Accepts the native GLiNER2 form, a simple ``entities`` mapping, or character
    spans (which are sliced from the text into surface forms).
    """
    record = gliner2_row_to_training_record(row)
    return {
        "text": record["input"],
        "entities": record["output"].get("entities", {}),
        "entity_descriptions": record["output"].get("entity_descriptions"),
    }
