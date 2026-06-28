# Fine-tuning & bringing your own models

TrialMatchAI ships with capable default models, but every model in the pipeline
is **swappable** and **fine-tunable**. You can point the pipeline at your own
checkpoints/adapters via config, and train those adapters with the built-in
`trialmatchai-finetune` command.

| Component | Default | Config key | Fine-tune target |
|-----------|---------|------------|------------------|
| Biomedical extraction | `fastino/gliner2-base-v1` | `entity_extraction.model_name` | GLiNER2 checkpoint |
| Reranker | `google/gemma-2-2b-it` | `model.reranker_adapter_path` | LoRA adapter |
| CoT eligibility | configured CoT model | `model.cot_adapter_path` | LoRA adapter |

> **Where does the training data come from?** Fine-tuning is **optional** — the
> ready-to-use CoT and reranker adapters are downloaded by `trialmatchai-bootstrap-data`,
> so most deployments never need to train. The **raw training datasets are not
> published**; to re-train, bring your own JSONL in the schemas shown below
> (`data/finetune/*.jsonl`). Each line is a self-contained example, so you can
> assemble a dataset from your own annotated patient–trial pairs.

Install the training dependencies:

```bash
uv sync --extra finetune
```

## 1. Using a custom or fine-tuned model (no training)

Already have a checkpoint or adapter? Just point the config at it — no code
changes needed.

- **GLiNER2 extraction:** set `entity_extraction.model_name` to your GLiNER2
  checkpoint (local path or Hub id), backend `gliner2`. LoRA adapters load via
  `GLiNER2.load_adapter`.
- **Reranker:** set `model.reranker_adapter_path` to your LoRA adapter directory.
- **CoT:** set `model.cot_adapter_path` to your LoRA adapter directory.

The reranker and CoT both run on **vLLM (the only LLM backend)**, which serves
the LoRA adapter natively via `LoRARequest` — no merge step required. If you
prefer a single self-contained model instead of base + adapter, merge them:

```bash
trialmatchai-finetune merge \
  --base-model google/gemma-2-2b-it \
  --adapter models/reranker-adapter \
  --output-dir models/reranker-merged
```

Then point the config at the merged directory (`model.reranker_model_path` for
the reranker, `model.base_model` for the merged CoT model) and leave the matching
adapter path (`model.reranker_adapter_path` / `model.cot_adapter_path`) empty.

## 2. Fine-tuning

### CoT eligibility model (LoRA)

Train an adapter that improves chain-of-thought eligibility evaluation.

Data — JSONL, one example per line, either chat or instruct form:

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "{...evaluation JSON...}"}]}
{"instruction": "...", "input": "...", "output": "..."}
```

```bash
trialmatchai-finetune cot \
  --base-model microsoft/phi-4 \
  --train-data data/finetune/cot.jsonl \
  --eval-data data/finetune/cot.eval.jsonl \
  --output-dir models/cot-adapter \
  --epochs 2 --lora-rank 32 --lora-alpha 64
```

Then set `model.cot_adapter_path: models/cot-adapter`.

By default, LoRA targets all linear layers (`--target-modules all-linear`),
which works across common Gemma/Llama/Phi-style architectures. Use
`--target-modules auto` to let PEFT choose its built-in mapping, or pass a
comma-separated suffix list such as `q_proj,v_proj`.

### Reranker (LoRA, Yes/No)

The reranker decides whether the patient text contains enough information to
evaluate a criterion. Training teaches the model to emit `Yes`/`No`.

Data — JSONL:

```json
{"patient_text": "...", "criterion": "...", "label": "Yes"}
{"patient_text": "...", "criterion": "...", "label": "No"}
```

```bash
trialmatchai-finetune reranker \
  --base-model google/gemma-2-2b-it \
  --train-data data/finetune/reranker.jsonl \
  --eval-data data/finetune/reranker.eval.jsonl \
  --output-dir models/reranker-adapter
```

Then set `model.reranker_adapter_path: models/reranker-adapter`.

### GLiNER2 Schema Extraction

Uses the native GLiNER2 training stack (`GLiNER2Trainer`). The CLI subcommand is
still named `ner` for compatibility, but the training data can now include flat
entities, schema-based JSON structures, classifications, relations, or a mix.
Flat entity data maps entity-type labels to **surface forms**:

```json
{"text": "EGFR exon 19 deletion in NSCLC", "entities": {"gene": ["EGFR"], "disease": ["NSCLC"]}}
{"text": "EGFR positive", "ner": [[0, 4, "gene"]]}
{"input": "EGFR positive", "output": {"entities": {"gene": ["EGFR"]}}}
{"text": "No reportable biomarker", "schema": {"entities": {"gene": []}}}
```

Structured JSON examples use GLiNER2's native `json_structures` and optional
`json_descriptions` keys:

```json
{"text": "Patient has EGFR exon 19 deletion and stage IV NSCLC.", "schema": {"entities": {"gene": ["EGFR"]}, "json_structures": [{"biomarker": {"gene": "EGFR", "variant": "exon 19 deletion", "disease_stage": "stage IV"}}], "json_descriptions": {"biomarker": {"gene": "Gene symbol", "variant": "Observed alteration", "disease_stage": "Disease stage"}}}}
{"input": "Erlotinib targets EGFR.", "output": {"relations": [{"targets": {"head": "Erlotinib", "tail": "EGFR"}}]}}
{"text": "Trial requires ECOG 0-1.", "structures": {"eligibility": {"performance_status": "ECOG 0-1"}}}
```

`entity_descriptions` are back-filled from your entity schema (`--schema-path`)
for flat entity labels, so entity examples share the runtime label semantics.
TrialMatchAI's current runtime annotator consumes the flat `entities` output;
structured JSON training data is useful for GLiNER2 adapters you call directly
or for future structured extraction integration.

```bash
trialmatchai-finetune ner \
  --base-model fastino/gliner2-base-v1 \
  --train-data data/finetune/ner.jsonl \
  --output-dir models/ner \
  --schema-path src/trialmatchai/entity_schemas/trialmatchai.yaml \
  --epochs 10            # LoRA by default; add --no-lora for a full fine-tune
```

- LoRA run saves the adapter to `models/ner/final`; a full run saves
  `models/ner/best`. Set `entity_extraction.model_name` to that path.
- Encoder vs. task-head learning rates are tuned separately
  (`--encoder-lr 1e-5 --task-lr 5e-4`, GLiNER2 defaults).

## Notes

- Training prompts reuse the **exact runtime prompts**, so a fine-tuned model
  sees the same format at train and inference time.
- LoRA SFT masks the prompt tokens and computes loss only on the completion.
- `--eval-data` is optional for LoRA SFT. When supplied, evaluation loss is used
  for best-checkpoint tracking; `--save-steps` must be a multiple of
  `--eval-steps` (defaults align at 500).
- 4-bit quantized loading is on by default (`--no-4bit` to disable) and requires
  bitsandbytes on a CUDA-capable machine; `bf16` is default (`--fp16` to switch).
  See `trialmatchai-finetune <component> --help` for all flags.
- GLiNER2's training API can vary by version; if your installed `gliner2` exposes
  a different interface, adapt `src/trialmatchai/finetuning/ner.py`.
