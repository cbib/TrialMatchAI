"""Dataset loading and prompt formatting for fine-tuning.

Training prompts reuse the exact runtime prompt builders so a fine-tuned model
sees the same format at train and inference time.

Input formats (JSONL, one object per line):
- CoT:      {"messages": [{"role","content"}, ...]}
            or {"instruction": str, "input": str, "output": str}
- Reranker: {"patient_text": str, "criterion": str, "label": "Yes"|"No"}
- NER:      {"text": str, "ner": [[start_char, end_char, "label"], ...]}
            or GLiNER-native {"tokenized_text": [...], "ner": [[s_tok, e_tok, label]]}
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterator, List, Optional


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

_WORD_RE = re.compile(r"\w+|[^\w\s]")


def _tokenize_with_spans(text: str) -> tuple[List[str], List[tuple[int, int]]]:
    tokens: List[str] = []
    spans: List[tuple[int, int]] = []
    for match in _WORD_RE.finditer(text):
        tokens.append(match.group())
        spans.append((match.start(), match.end()))
    return tokens, spans


def ner_row_to_gliner(row: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize an NER row to GLiNER's training format.

    Accepts GLiNER-native rows unchanged; converts char-span rows by mapping
    character offsets onto whitespace/punctuation tokens.
    """
    if "tokenized_text" in row:
        return {"tokenized_text": row["tokenized_text"], "ner": row.get("ner", [])}

    text = str(row.get("text", ""))
    raw_spans = row.get("ner")
    if raw_spans is None:
        raw_spans = [
            [ent["start"], ent["end"], ent["label"]] for ent in row.get("entities", [])
        ]

    tokens, spans = _tokenize_with_spans(text)
    ner: List[List[Any]] = []
    for start_char, end_char, label in raw_spans:
        start_tok = next(
            (i for i, (s, e) in enumerate(spans) if s <= start_char < e), None
        )
        end_tok = next(
            (i for i, (s, e) in enumerate(spans) if s < end_char <= e), None
        )
        if start_tok is not None and end_tok is not None and end_tok >= start_tok:
            ner.append([start_tok, end_tok, label])
    return {"tokenized_text": tokens, "ner": ner}


def iter_gliner_examples(
    path: str, max_examples: Optional[int] = None
) -> Iterator[Dict[str, Any]]:
    for row in read_jsonl(path, max_examples):
        yield ner_row_to_gliner(row)
