"""Runtime CoT query expansion (restored from the legacy pipeline).

Runs a chain-of-thought model over each patient's narrative to produce expanded
keywords (primary conditions + synonyms, secondary factors, expanded sentences) that
feed first-level retrieval. SYSTEM_PROMPT and generation behaviour are preserved
verbatim from the legacy matcher; model/backend are ``query_expansion`` config knobs.
Disabled by default; the TREC preset enables it.
"""

from __future__ import annotations

from typing import Any, Dict, List

from trialmatchai.matching.eligibility_base import BaseTrialProcessor
from trialmatchai.utils.json_utils import extract_json_object
from trialmatchai.utils.logging_config import setup_logging

logger = setup_logging(__name__)

# Verbatim from the legacy ClinicalSummarizer.generate_summary SYSTEM_PROMPT.
SYSTEM_PROMPT = """
You are a specialized medical assistant designed for precise and accurate clinical trial matching.
Analyze the patient's medical description carefully and extract clinically relevant information for trial eligibility assessment.

1. **Primary Condition**:
    - Determine the primary medical conditions based on explicit patient information and overall clinical context.
    - List up to 10 medically recognized synonyms, aliases, or closely related medical terms for the primary conditions.
    - Include the identified primary conditions and their associated synonyms or related terms within the "main_conditions" list.

2. **Secondary Clinical Factors**:
    - Provide up to 50 clinically significant additional factors, including comorbidities, concurrent medical conditions, molecular or genetic biomarkers, prior therapies, relevant medical history, and clinically notable patient characteristics explicitly mentioned in the patient description.
    - Provide these factors in the "other_conditions" list.

3. **Expanded Clinical Descriptions**:
    - Based solely on the original patient-provided data, generate semantically accurate and medically sound statements resembling real-life medical notes.
    - **Crucial**: Expanded descriptions must strictly reflect explicit patient-reported information without introducing new or inferred medical details.

Output:
Return a JSON object in the exact following structure without any additional commentary:

{
"main_conditions": ["PrimaryCondition", "Synonym1", "Synonym2", "..."],
"other_conditions": ["AdditionalCondition1", "AdditionalCondition2", "..."],
"expanded_sentences": [
    "Expanded note for sentence 1...",
    "Expanded note for sentence 2...",
    "..."
]
}
""".strip()

_EMPTY = {"main_conditions": [], "other_conditions": [], "expanded_sentences": []}

# JSON schema for grammar-constrained keyword expansion (vLLM structured outputs), so a verbose
# or reasoning model always returns valid keyword JSON instead of prose that fails to parse.
# maxItems bounds the array COUNT so the grammar forces each array closed instead of letting a
# verbose model emit items until it exhausts max_tokens (the original Baichuan runaway). Short-term
# fields (conditions) keep a small maxLength since they are terms; expanded_sentences deliberately
# has NO maxLength -- a hard char cap chopped real clinical sentences mid-word. Count is bounded by
# maxItems and total output by max_new_tokens, so a per-string cap is unnecessary and harmful.
_KEYWORDS_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "main_conditions": {"type": "array", "maxItems": 11, "items": {"type": "string", "maxLength": 120}},
        "other_conditions": {"type": "array", "maxItems": 50, "items": {"type": "string", "maxLength": 120}},
        "expanded_sentences": {"type": "array", "maxItems": 15, "items": {"type": "string"}},
    },
    "required": ["main_conditions", "other_conditions", "expanded_sentences"],
}


def _as_list(value: object) -> list:
    """Coerce an expansion field to a list; a bare string becomes ``["cancer"]``, not the
    per-character shredding of ``list("cancer")``."""
    if isinstance(value, list):
        return [v for v in value if v]
    if isinstance(value, str) and value.strip():
        return [value]
    return []


def _resolve_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    qe = dict(config.get("query_expansion") or {})
    model_cfg = config.get("model", {})
    return {
        # Default to the RAG/eligibility backend so the expander shares the cached CoT
        # engine instead of loading a second copy.
        "backend": qe.get("backend") or config.get("rag", {}).get("backend") or "vllm",
        "model": qe.get("model") or model_cfg.get("base_model"),
        "adapter": qe.get("adapter", model_cfg.get("cot_adapter_path")),
        "device": str(config.get("global", {}).get("device", 0)),
        "max_new_tokens": int(qe.get("max_new_tokens", 2048)),
        "trust_remote_code": bool(
            qe.get("trust_remote_code", model_cfg.get("trust_remote_code", False))
        ),
        "max_main_conditions": int(qe.get("max_main_conditions", 11)),
        "max_other_conditions": int(qe.get("max_other_conditions", 50)),
        # Suppress chain-of-thought for this extractive task: reasoning models otherwise
        # fill the token budget with <think>, never emit the JSON, and expansion falls back.
        "no_think": bool(qe.get("no_think", False)),
        "guided_json": bool(qe.get("guided_json", False)),
    }


class QueryExpander:
    """CoT expander; loads its model lazily so import stays base-deps safe."""

    # Overridable by subclasses that reuse this engine/template machinery for a different
    # extraction task (see FirstLevelQueryExpander).
    system_prompt: str = SYSTEM_PROMPT
    json_schema: Dict[str, Any] = _KEYWORDS_JSON_SCHEMA

    def __init__(self, settings: Dict[str, Any], config: Dict[str, Any]):
        self.settings = settings
        self.config = config
        self.backend = settings["backend"]
        if not settings.get("model"):
            raise ValueError("query_expansion requires a model (set query_expansion.model or model.base_model)")
        if self.backend == "vllm":
            self._init_vllm()
        elif self.backend == "transformers":
            self._init_transformers()
        else:
            raise ValueError(f"Unsupported query_expansion.backend: {self.backend}")

    # -- backends ---------------------------------------------------------- #
    def _init_transformers(self) -> None:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer

        s = self.settings
        self.tokenizer = AutoTokenizer.from_pretrained(
            s["model"], trust_remote_code=s["trust_remote_code"]
        )
        model = AutoModelForCausalLM.from_pretrained(
            s["model"],
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=s["trust_remote_code"],
        )
        if s.get("adapter"):
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, s["adapter"])
        model.eval()
        self.model = model

    def _init_vllm(self) -> None:
        from trialmatchai.models.llm.vllm_loader import load_vllm_engine

        s = self.settings
        # Same model_config/vllm_cfg shape as the RAG path so an identical request hits
        # the engine cache and shares ONE engine, not a second CoT copy.
        model_config = {
            **self.config.get("model", {}),
            "base_model": s["model"],
            "cot_adapter_path": s.get("adapter"),
        }
        self.engine, self.tokenizer, self.lora_request = load_vllm_engine(
            model_config=model_config, vllm_cfg=self.config.get("vllm", {})
        )

    # -- generation -------------------------------------------------------- #
    def _generate(self, narrative: str) -> str:
        no_think = bool(self.settings.get("no_think"))
        user_content = ("/no_think\n" + narrative) if no_think else narrative
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]
        # Qwen3.x-style templates take enable_thinking; harmless-and-ignored elsewhere (guarded).
        tmpl_kwargs = {"enable_thinking": False} if no_think else {}

        def _apply_template(tokenize):
            try:
                return self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=tokenize, **tmpl_kwargs
                )
            except TypeError:
                return self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=tokenize
                )

        if self.backend == "transformers":
            import torch

            prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(self.model.device)
            with torch.no_grad():
                out = self.model.generate(
                    prompt,
                    max_new_tokens=self.settings["max_new_tokens"],
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            return self.tokenizer.decode(out[0][prompt.shape[-1]:], skip_special_tokens=True)

        # vllm
        from vllm import SamplingParams

        prompt_text = _apply_template(tokenize=False)
        structured = None
        if self.settings.get("guided_json"):
            from vllm.sampling_params import StructuredOutputsParams  # type: ignore

            structured = StructuredOutputsParams(json=self.json_schema, disable_any_whitespace=True)
        params = SamplingParams(
            temperature=0.0,
            max_tokens=self.settings["max_new_tokens"],
            structured_outputs=structured,
        )
        results = self.engine.generate([prompt_text], params, lora_request=self.lora_request)
        return results[0].outputs[0].text if results and results[0].outputs else ""

    def expand(self, narrative_sentences: List[str]) -> Dict[str, Any]:
        """Return {main_conditions, other_conditions, expanded_sentences}."""
        narrative = " ".join(s for s in narrative_sentences if s).strip()
        if not narrative:
            return dict(_EMPTY)
        try:
            raw = self._generate(narrative)
            # Strip the <think> chain first (as the eligibility path does) so extraction
            # grabs the JSON answer, not the schema echoed inside the reasoning.
            parsed = extract_json_object(BaseTrialProcessor._strip_thinking_tags(raw))
            if not isinstance(parsed, dict):
                raise ValueError("expansion output was not a JSON object")
            return {key: _as_list(parsed.get(key)) for key in _EMPTY}
        except Exception as exc:
            logger.error("Query expansion failed; falling back to no expansion: %s", exc)
            return dict(_EMPTY)


def build_query_expander(config: Dict[str, Any]) -> QueryExpander | None:
    """Construct an expander when ``query_expansion.enabled`` is true, else None."""
    qe = config.get("query_expansion") or {}
    if not qe.get("enabled"):
        return None
    return QueryExpander(_resolve_settings(config), config)


def enrich_summary(
    summary: Dict[str, Any],
    expansion: Dict[str, Any],
    *,
    max_main_conditions: int = 11,
    max_other_conditions: int = 50,
) -> Dict[str, Any]:
    """Fold a CoT expansion into a matching summary (legacy keywords.json shape).

    ``expanded_sentences`` map to ``patient_narrative``; only non-empty fields
    overwrite, leaving the deterministic summary intact.
    """
    out = dict(summary)
    main = expansion.get("main_conditions") or []
    other = expansion.get("other_conditions") or []
    sentences = expansion.get("expanded_sentences") or []
    if main:
        out["main_conditions"] = main[:max_main_conditions]
    if other:
        out["other_conditions"] = other[:max_other_conditions]
    if sentences:
        out["patient_narrative"] = sentences
    return out


# --- first-level retrieval query expansion (the llm_expansion search channel) ------------ #

# Distinct from SYSTEM_PROMPT above. That one enriches the patient SUMMARY (conditions and
# narrative sentences). This one writes RETRIEVAL QUERIES: short noun phrases that should
# match trial titles, conditions and eligibility text. The two are not interchangeable --
# the first-level planner buckets these six fields into weighted query channels.
FIRST_LEVEL_SYSTEM_PROMPT = """
You expand a patient description into search queries for a clinical trial index.

Write SHORT NOUN PHRASES that would appear in a trial's title, condition list or
eligibility criteria. Do not write sentences, questions or explanations.

Fill these six fields:

1. "primary_queries": the patient's main disease as a trial would name it. Include the
   staging or subtype only when the patient description states it.
2. "disease_aliases": other names for that same disease -- synonyms, abbreviations, older
   or regional terminology, and the expanded form of any abbreviation.
3. "broader_queries": the parent disease categories a trial might recruit under, from
   narrower to wider. These deliberately trade precision for coverage.
4. "biomarker_queries": genes, mutations, fusions, receptor and expression status, and
   other molecular markers stated for this patient.
5. "treatment_queries": drugs, drug classes, procedures and prior therapies stated for
   this patient.
6. "discarded_or_uncertain": terms you considered but rejected, and anything you are not
   confident the patient description supports.

Rules:
- Use ONLY what the patient description states. Never infer a diagnosis, stage, biomarker
  or therapy that is not written there. Put anything doubtful in "discarded_or_uncertain".
- Leave a field as an empty list when the description supports nothing for it. An empty
  list is correct; an invented term is not.
- No duplicates within a field.

Return a JSON object with exactly those six keys and no other commentary.
""".strip()

_FIRST_LEVEL_FIELDS = (
    "primary_queries",
    "disease_aliases",
    "broader_queries",
    "biomarker_queries",
    "treatment_queries",
    "discarded_or_uncertain",
)

# maxItems bounds the array COUNT for the same reason as _KEYWORDS_JSON_SCHEMA: it forces a
# verbose model to close each array instead of emitting terms until max_tokens runs out.
# These are noun phrases, so a short maxLength is safe here (unlike expanded_sentences).
#
# The per-field caps are deliberately uneven. search.first_level.llm_max_terms is a SHARED
# budget across the five query fields, spent in field order (first_level_planner
# parse_llm_query_expansion), so a model that pads primary_queries starves the biomarker and
# treatment channels entirely. Capping primary_queries tightly -- a patient has one main
# disease, not twelve -- keeps the budget available for the later fields.
_FIRST_LEVEL_MAX_ITEMS = {
    "primary_queries": 3,
    "disease_aliases": 8,
    "broader_queries": 5,
    "biomarker_queries": 8,
    "treatment_queries": 8,
    "discarded_or_uncertain": 12,
}
_FIRST_LEVEL_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        field: {
            "type": "array",
            "maxItems": _FIRST_LEVEL_MAX_ITEMS[field],
            "items": {"type": "string", "maxLength": 120},
        }
        for field in _FIRST_LEVEL_FIELDS
    },
    "required": list(_FIRST_LEVEL_FIELDS),
}

_FIRST_LEVEL_EMPTY: Dict[str, List[str]] = {field: [] for field in _FIRST_LEVEL_FIELDS}


def _first_level_patient_text(profile: Any, matching_summary: Dict[str, Any]) -> str:
    """Compact patient description for the expander prompt.

    Built from the matching summary rather than the raw profile so it stays in step with
    what first-level retrieval actually searches on.
    """
    summary = matching_summary or {}
    parts: List[str] = []
    main = [c for c in _as_list(summary.get("main_conditions")) if c][:12]
    other = [c for c in _as_list(summary.get("other_conditions")) if c][:30]
    narrative = [s for s in _as_list(summary.get("patient_narrative")) if s][:12]
    if main:
        parts.append("Main conditions: " + "; ".join(main))
    if other:
        parts.append("Other conditions and factors: " + "; ".join(other))
    age, gender = summary.get("age"), summary.get("gender")
    demographics = [
        f"Age: {age}" for _ in (1,) if age not in (None, "", "all")
    ] + [f"Sex: {gender}" for _ in (1,) if gender not in (None, "", "all")]
    if demographics:
        parts.append(", ".join(demographics))
    if narrative:
        parts.append("Description: " + " ".join(narrative))
    return "\n".join(parts).strip()


class FirstLevelQueryExpander(QueryExpander):
    """Implements ``LLMQueryExpansionBackend`` for the first-level ``llm_expansion`` channel.

    Reuses QueryExpander's engine, chat-template and structured-output machinery -- so it
    shares the one cached vLLM engine rather than loading a second copy -- but swaps in the
    retrieval-query prompt and schema.
    """

    system_prompt = FIRST_LEVEL_SYSTEM_PROMPT
    json_schema = _FIRST_LEVEL_JSON_SCHEMA

    def expand_first_level_queries(
        self,
        *,
        profile: Any,
        matching_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        patient_text = _first_level_patient_text(profile, matching_summary)
        if not patient_text:
            return dict(_FIRST_LEVEL_EMPTY)
        try:
            raw = self._generate(patient_text)
            parsed = extract_json_object(BaseTrialProcessor._strip_thinking_tags(raw))
            if not isinstance(parsed, dict):
                raise ValueError("first-level expansion output was not a JSON object")
            return {field: _as_list(parsed.get(field)) for field in _FIRST_LEVEL_FIELDS}
        except Exception as exc:
            # Retrieval must not fail because expansion did: the channel is one of eight and
            # carries weight 0.5, so an empty expansion degrades recall rather than the run.
            logger.error(
                "First-level query expansion failed; continuing without that channel: %s", exc
            )
            return dict(_FIRST_LEVEL_EMPTY)


def build_first_level_expander(config: Dict[str, Any]) -> "FirstLevelQueryExpander | None":
    """Construct the expander when ``search.first_level.llm_expansion_enabled`` is true.

    Independent of ``query_expansion.enabled``: that flag governs the separate summary
    enrichment stage. Both may run, and they share one engine.
    """
    first_level = (config.get("search") or {}).get("first_level") or {}
    if not first_level.get("llm_expansion_enabled"):
        return None
    try:
        return FirstLevelQueryExpander(_resolve_settings(config), config)
    except Exception as exc:
        logger.error(
            "search.first_level.llm_expansion_enabled is set but the expander could not be "
            "built; first-level search continues without that channel: %s",
            exc,
        )
        return None
