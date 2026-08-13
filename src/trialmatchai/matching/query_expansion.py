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

4. **Broader Disease Categories**:
    - Provide up to 8 BROADER categories the primary conditions belong to, from narrower to
      wider (e.g. "Primary Open Angle Glaucoma" -> "Open-Angle Glaucoma", "Glaucoma";
      "COPD" -> "Lung Diseases", "Respiratory Tract Diseases").
    - These deliberately trade precision for coverage: a trial may recruit under the category
      rather than the specific diagnosis.
    - Do NOT repeat the primary conditions or their synonyms here, and do NOT list a
      comorbidity as a broader category of the primary condition.
    - Provide these in the "broader_conditions" list.

5. **Discarded or Uncertain**:
    - List terms you considered but must NOT be searched, in particular anything the patient
      description NEGATES ("no prior chemotherapy", "absence of metastases"). Include BOTH the
      negated phrase and its un-negated form, because searching the bare term would retrieve
      exactly the wrong trials.
    - Also include anything you are not confident the description supports.
    - Provide these in the "discarded_or_uncertain" list, and keep them out of every other list.

Output:
Return a JSON object in the exact following structure without any additional commentary:

{
"main_conditions": ["PrimaryCondition", "Synonym1", "Synonym2", "..."],
"other_conditions": ["AdditionalCondition1", "AdditionalCondition2", "..."],
"broader_conditions": ["BroaderCategory1", "BroaderCategory2", "..."],
"discarded_or_uncertain": ["NegatedOrUnsupportedTerm1", "..."],
"expanded_sentences": [
    "Expanded note for sentence 1...",
    "Expanded note for sentence 2...",
    "..."
]
}
""".strip()

_EMPTY = {
    "main_conditions": [],
    "other_conditions": [],
    # Broader categories were previously produced by a SECOND llm pass (the llm_expansion
    # channel). Measured on 4 TREC 2023 patients, 82% of that pass's terms were already in
    # keywords.json -- it re-read its own input and handed the aliases back. The only thing
    # it contributed was these broader categories, so they are generated here instead and the
    # second pass is gone. One LLM call per patient rather than two.
    "broader_conditions": [],
    # Negated terms that must NOT be searched. "No prior cataract surgery" means searching
    # "cataract surgery" retrieves exactly the wrong trials.
    "discarded_or_uncertain": [],
    "expanded_sentences": [],
}

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
        "broader_conditions": {"type": "array", "maxItems": 8, "items": {"type": "string", "maxLength": 120}},
        "discarded_or_uncertain": {"type": "array", "maxItems": 20, "items": {"type": "string", "maxLength": 120}},
        "expanded_sentences": {"type": "array", "maxItems": 15, "items": {"type": "string"}},
    },
    "required": [
        "main_conditions",
        "other_conditions",
        "broader_conditions",
        "discarded_or_uncertain",
        "expanded_sentences",
    ],
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
    max_broader_conditions: int = 8,
) -> Dict[str, Any]:
    """Fold a CoT expansion into a matching summary (legacy keywords.json shape).

    ``expanded_sentences`` map to ``patient_narrative``; only non-empty fields
    overwrite, leaving the deterministic summary intact.

    ``broader_conditions`` is kept as its own key rather than merged into main/other, because
    the planner routes it to the broader_disease channel at weight 0.35. Folding broad
    categories into main_conditions would search them at weight 1.0, flooding the pool with
    loosely related trials.
    """
    out = dict(summary)
    main = expansion.get("main_conditions") or []
    other = expansion.get("other_conditions") or []
    broader = expansion.get("broader_conditions") or []
    discarded = expansion.get("discarded_or_uncertain") or []
    sentences = expansion.get("expanded_sentences") or []
    if main:
        out["main_conditions"] = main[:max_main_conditions]
    if other:
        out["other_conditions"] = other[:max_other_conditions]
    if broader:
        out["broader_conditions"] = broader[:max_broader_conditions]
    if discarded:
        # Recorded for provenance and so downstream stages can avoid them; never searched.
        out["discarded_or_uncertain"] = discarded
    if sentences:
        out["patient_narrative"] = sentences
    return out
