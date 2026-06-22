# Fine-tuning & bringing your own models

TrialMatchAI ships with capable default models, but every model in the pipeline
is **swappable** and **fine-tunable**. You can point the pipeline at your own
checkpoints/adapters via config, and train those adapters with the built-in
`trialmatchai-finetune` command.

| Component | Default | Config key | Fine-tune target |
|-----------|---------|------------|------------------|
| Biomedical NER | `fastino/gliner2-base` | `entity_extraction.model_name` | GLiNER checkpoint |
| Reranker | `google/gemma-2-2b-it` | `model.reranker_adapter_path` | LoRA adapter |
| CoT eligibility | configured CoT model | `model.cot_adapter_path` | LoRA adapter |

Install the training dependencies:

```bash
uv sync --extra finetune
```

## 1. Using a custom or fine-tuned model (no training)

Already have a checkpoint or adapter? Just point the config at it — no code
changes needed.

- **NER:** set `entity_extraction.model_name` to your GLiNER2 checkpoint (local
  path or Hub id), backend `gliner2`. LoRA NER adapters load via `GLiNER2.load_adapter`.
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

Then point the config at the merged directory (`reranker_model_path` /
`cot model`) and leave the adapter path empty.

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
  --output-dir models/cot-adapter \
  --epochs 2 --lora-rank 32 --lora-alpha 64
```

Then set `model.cot_adapter_path: models/cot-adapter`.

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
  --output-dir models/reranker-adapter
```

Then set `model.reranker_adapter_path: models/reranker-adapter`.

### NER (GLiNER2)

Uses the native GLiNER2 training stack (`GLiNER2Trainer`). GLiNER2 NER data maps
entity-type labels to **surface forms**. Three input shapes are accepted:

```json
{"text": "EGFR exon 19 deletion in NSCLC", "entities": {"gene": ["EGFR"], "disease": ["NSCLC"]}}
{"text": "EGFR positive", "ner": [[0, 4, "gene"]]}
{"input": "EGFR positive", "output": {"entities": {"gene": ["EGFR"]}}}
```

`entity_descriptions` are back-filled from your entity schema (`--schema-path`)
so the fine-tuned model shares the runtime label semantics.

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
- 4-bit quantized loading is on by default (`--no-4bit` to disable); `bf16` is
  default (`--fp16` to switch). See `trialmatchai-finetune <component> --help`
  for all flags.
- GLiNER's training API varies by version; if your installed `gliner` exposes a
  different interface, adapt `src/trialmatchai/finetuning/ner.py`.
