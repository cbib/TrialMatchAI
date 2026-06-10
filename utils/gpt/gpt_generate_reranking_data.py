import json
import os
import random
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain.schema import HumanMessage
from langchain_community.chat_models import ChatOpenAI
from pydantic import BaseModel

load_dotenv()

INPUT_MEDNLI_FILE = "mednli_train.jsonl"
OUTPUT_AUGMENTED_FILE = "mednli_yesno_aug.jsonl"

N_VARIANTS_PER_SEED = 3
MAX_SEED_EXAMPLES = 500

MODEL_NAME = os.environ.get("UMGPT_MODEL", "gpt-4o-mini")
TEMPERATURE = 0.7

INSTRUCTION_TEXT = (
    "You are a clinical assistant tasked with determining whether the patient "
    "information (Statement A) provides enough details to evaluate whether the "
    "patient satisfies or violates the clinical trial eligibility criterion "
    "(Statement B). Respond with 'Yes' if Statement A contains sufficient "
    "information to make this evaluation, or 'No' if it does not."
)


class MedNLISeed(BaseModel):
    sentence1: str
    sentence2: str
    gold_label: str


class YesNoExample(BaseModel):
    instruction: str
    sentence1: str
    sentence2: str
    gold_label: str


llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=TEMPERATURE,
    top_p=0.9,
    openai_api_key=os.environ["UMGPT_API_KEY"],
    openai_api_base=os.environ["UMGPT_BASE_URL"],
)


def load_mednli_jsonl(path: str) -> List[MedNLISeed]:
    examples: List[MedNLISeed] = []
    with open(path, "r") as f:
        for line in f:
            raw = json.loads(line.strip())
            examples.append(
                MedNLISeed(
                    sentence1=raw["sentence1"],
                    sentence2=raw["sentence2"],
                    gold_label=raw["gold_label"],
                )
            )
    return examples


def write_jsonl(path: str, data: List[Dict]):
    with open(path, "a", encoding="utf-8") as f:
        for obj in data:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def generate_yesno_variants(
    seed_example: MedNLISeed, n_variants: int
) -> List[YesNoExample]:
    prompt = f"""
You are generating training data for a clinical trial matching system.

We want examples following this pattern:

- Statement A: patient information.
- Statement B: clinical trial eligibility criterion.
- A fixed instruction string.
- A label "Yes" or "No".

Semantics:
- "Yes": Statement A contains enough information to determine if the patient satisfies or violates Statement B.
- "No": Statement A does not contain enough information.

MedNLI mapping:
- entailment/contradiction → Yes
- neutral → No

Seed example:
{{
  "sentence1": {json.dumps(seed_example.sentence1)},
  "sentence2": {json.dumps(seed_example.sentence2)},
  "gold_label": {json.dumps(seed_example.gold_label)}
}}

Generate {n_variants} new examples. Each example must have:
- "instruction": {json.dumps(INSTRUCTION_TEXT)}
- "sentence1": Statement A
- "sentence2": Statement B
- "gold_label": "Yes" or "No"

Return only valid JSON: a list of objects with exactly these keys.
""".strip()

    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").replace("json", "", 1).strip()

    try:
        parsed = json.loads(raw)
        out = []
        for item in parsed:
            try:
                ex = YesNoExample(**item)
                if ex.gold_label in {"Yes", "No"}:
                    ex.instruction = INSTRUCTION_TEXT
                    out.append(ex)
            except Exception:
                pass
        return out
    except Exception:
        return []


def augment_mednli_yesno(
    input_path: str,
    output_path: str,
    n_variants_per_seed: int = 3,
    max_seed_examples: Optional[int] = None,
):
    all_seeds = load_mednli_jsonl(input_path)

    if max_seed_examples is not None and max_seed_examples < len(all_seeds):
        seed_examples = random.sample(all_seeds, max_seed_examples)
    else:
        seed_examples = all_seeds

    open(output_path, "w").close()

    for seed in seed_examples:
        generated = generate_yesno_variants(seed, n_variants_per_seed)
        if not generated:
            continue
        to_write = [
            {
                "instruction": ex.instruction,
                "sentence1": ex.sentence1,
                "sentence2": ex.sentence2,
                "gold_label": ex.gold_label,
            }
            for ex in generated
        ]
        write_jsonl(output_path, to_write)


if __name__ == "__main__":
    augment_mednli_yesno(
        input_path=INPUT_MEDNLI_FILE,
        output_path=OUTPUT_AUGMENTED_FILE,
        n_variants_per_seed=N_VARIANTS_PER_SEED,
        max_seed_examples=MAX_SEED_EXAMPLES,
    )
