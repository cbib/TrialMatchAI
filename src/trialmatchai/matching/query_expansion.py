"""Runtime CoT query expansion (re-introduced from the legacy pipeline).

The legacy matcher ran a chain-of-thought model over each patient's narrative to
produce the expanded ``keywords.json`` (primary conditions + synonyms, secondary
factors, expanded sentences) that feeds first-level retrieval. The refactor
dropped this; this module restores it faithfully while making the model a config
knob.

Verbatim SYSTEM_PROMPT and generation behaviour are preserved from
``source/Matcher/pipeline/phenopacket_processor.py`` (main branch). The model and
backend default to the configured CoT reasoning model but are overridable via the
``query_expansion`` config block. Disabled by default; the TREC preset turns it on.
"""

from __future__ import annotations

from typing import Any, Dict, List

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


def _resolve_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    qe = dict(config.get("query_expansion") or {})
    model_cfg = config.get("model", {})
    return {
        "backend": qe.get("backend") or config.get("cot_backend") or "transformers",
        "model": qe.get("model") or model_cfg.get("base_model"),
        "adapter": qe.get("adapter", model_cfg.get("cot_adapter_path")),
        "device": str(config.get("global", {}).get("device", 0)),
        "max_new_tokens": int(qe.get("max_new_tokens", 2048)),
        "trust_remote_code": bool(
            qe.get("trust_remote_code", model_cfg.get("trust_remote_code", False))
        ),
        "max_main_conditions": int(qe.get("max_main_conditions", 11)),
        "max_other_conditions": int(qe.get("max_other_conditions", 50)),
    }


class QueryExpander:
    """CoT expander; loads its model lazily so import stays base-deps safe."""

    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
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
        self.engine, self.tokenizer, self.lora_request = load_vllm_engine(
            s["model"], adapter_path=s.get("adapter")
        )

    # -- generation -------------------------------------------------------- #
    def _generate(self, narrative: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": narrative},
        ]
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

        prompt_text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        params = SamplingParams(temperature=0.0, max_tokens=self.settings["max_new_tokens"])
        results = self.engine.generate([prompt_text], params, lora_request=self.lora_request)
        return results[0].outputs[0].text if results and results[0].outputs else ""

    def expand(self, narrative_sentences: List[str]) -> Dict[str, Any]:
        """Return {main_conditions, other_conditions, expanded_sentences}."""
        narrative = " ".join(s for s in narrative_sentences if s).strip()
        if not narrative:
            return dict(_EMPTY)
        try:
            raw = self._generate(narrative)
            parsed = extract_json_object(raw)
            if not isinstance(parsed, dict):
                raise ValueError("expansion output was not a JSON object")
            return {key: list(parsed.get(key) or []) for key in _EMPTY}
        except Exception as exc:
            logger.error("Query expansion failed; falling back to no expansion: %s", exc)
            return dict(_EMPTY)


def build_query_expander(config: Dict[str, Any]) -> QueryExpander | None:
    """Construct an expander when ``query_expansion.enabled`` is true, else None."""
    qe = config.get("query_expansion") or {}
    if not qe.get("enabled"):
        return None
    return QueryExpander(_resolve_settings(config))


def enrich_summary(
    summary: Dict[str, Any],
    expansion: Dict[str, Any],
    *,
    max_main_conditions: int = 11,
    max_other_conditions: int = 50,
) -> Dict[str, Any]:
    """Fold a CoT expansion into a matching summary (legacy keywords.json shape).

    The legacy ``expanded_sentences`` map to the summary's ``patient_narrative``.
    Only non-empty expansion fields overwrite; otherwise the deterministic
    summary is left intact.
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
