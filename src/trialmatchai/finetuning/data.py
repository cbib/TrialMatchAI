"""Dataset loading and prompt formatting for fine-tuning.

Training prompts reuse the exact runtime prompt builders so a fine-tuned model
sees the same format at train and inference time.

Input formats (JSONL, one object per line):
- CoT:      {"messages": [{"role","content"}, ...]}
            or {"instruction": str, "input": str, "output": str}
- Reranker: {"patient_text": str, "criterion": str, "label": "Yes"|"No"}
- NER (GLiNER2): {"text": str, "entities": {"label": ["surface form", ...]}}
            or char-span {"text": str, "ner": [[start_char, end_char, "label"], ...]}
            or native {"input": str, "output": {"entities": {...}, "entity_descriptions": {...}}}
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def read_jsonl(path: str, max_examples: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_examples is not None and len(rows) >= max_examples:
                break
    return rows


# --------------------------------------------------------------------------- CoT


def cot_row_to_messages(row: Dict[str, Any]) -> List[Dict[str, str]]:
    """Normalize a CoT training row to a chat-message list."""
    if "messages" in row:
        return list(row["messages"])
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


# --------------------------------------------------------------------------- NER


def ner_row_to_entities(row: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize an NER row to GLiNER2's surface-form schema.

    Returns ``{"text", "entities": {label: [surface forms]}, "entity_descriptions"}``.
    Accepts the native GLiNER2 form, a simple ``entities`` mapping, or character
    spans (which are sliced from the text into surface forms).
    """
    # Native GLiNER2 JSONL: {"input": ..., "output": {"entities": ..., ...}}
    if "input" in row and "output" in row:
        output = row.get("output") or {}
        return {
            "text": str(row["input"]),
            "entities": dict(output.get("entities") or {}),
            "entity_descriptions": output.get("entity_descriptions"),
        }

    text = str(row.get("text", ""))

    # Already a {label: [forms]} mapping.
    if isinstance(row.get("entities"), dict):
        return {
            "text": text,
            "entities": {k: list(v) for k, v in row["entities"].items()},
            "entity_descriptions": row.get("entity_descriptions"),
        }

    # Character spans -> surface forms grouped by label.
    entities: Dict[str, List[str]] = {}
    for span in row.get("ner") or []:
        start_char, end_char, label = span[0], span[1], span[2]
        surface = text[start_char:end_char].strip()
        if surface:
            entities.setdefault(label, [])
            if surface not in entities[label]:
                entities[label].append(surface)
    return {
        "text": text,
        "entities": entities,
        "entity_descriptions": row.get("entity_descriptions"),
    }
