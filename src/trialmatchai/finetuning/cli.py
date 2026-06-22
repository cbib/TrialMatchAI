"""``trialmatchai-finetune`` — train custom NER / reranker / CoT models.

Examples:
  trialmatchai-finetune cot      --base-model microsoft/phi-4 \
      --train-data data/cot.jsonl --output-dir models/cot-adapter
  trialmatchai-finetune reranker --base-model google/gemma-2-2b-it \
      --train-data data/reranker.jsonl --output-dir models/reranker-adapter
  trialmatchai-finetune ner      --base-model fastino/gliner2-base \
      --train-data data/ner.jsonl --output-dir models/ner

Plug the result back into config: entity_extraction.model_name (NER),
model.reranker_adapter_path (reranker), model.cot_adapter_path (CoT).
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from trialmatchai.finetuning.config import FinetuneConfig


def _add_common_lora_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eval-data", default=None)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit loading")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 instead of bf16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--hf-token", default=None)


def _lora_config_from_args(args: argparse.Namespace) -> FinetuneConfig:
    return FinetuneConfig(
        base_model=args.base_model,
        train_data=args.train_data,
        output_dir=args.output_dir,
        eval_data=args.eval_data,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_seq_length=args.max_seq_length,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        max_examples=args.max_examples,
        load_in_4bit=not args.no_4bit,
        bf16=not args.fp16,
        trust_remote_code=args.trust_remote_code,
        hf_token=args.hf_token,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trialmatchai-finetune",
        description="Fine-tune TrialMatchAI's NER, reranker, or CoT models.",
    )
    sub = parser.add_subparsers(dest="component", required=True)

    cot = sub.add_parser("cot", help="LoRA SFT for the CoT eligibility model")
    _add_common_lora_args(cot)

    reranker = sub.add_parser("reranker", help="LoRA SFT for the reranker (Yes/No)")
    _add_common_lora_args(reranker)

    ner = sub.add_parser("ner", help="Fine-tune the GLiNER2 NER model")
    ner.add_argument("--base-model", default="fastino/gliner2-base-v1")
    ner.add_argument("--train-data", required=True)
    ner.add_argument("--output-dir", required=True)
    ner.add_argument("--eval-data", default=None)
    ner.add_argument("--epochs", type=float, default=10.0)
    ner.add_argument("--batch-size", type=int, default=8)
    ner.add_argument("--encoder-lr", type=float, default=1e-5)
    ner.add_argument("--task-lr", type=float, default=5e-4)
    ner.add_argument("--lora-r", type=int, default=8)
    ner.add_argument("--lora-alpha", type=float, default=16.0)
    ner.add_argument("--no-lora", action="store_true", help="Full fine-tune instead of LoRA")
    ner.add_argument("--fp32", action="store_true", help="Disable fp16 mixed precision")
    ner.add_argument("--schema-path", default=None, help="Entity schema for label descriptions")
    ner.add_argument("--max-examples", type=int, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.component == "cot":
        from trialmatchai.finetuning.cot import finetune_cot

        finetune_cot(_lora_config_from_args(args))
    elif args.component == "reranker":
        from trialmatchai.finetuning.reranker import finetune_reranker

        finetune_reranker(_lora_config_from_args(args))
    elif args.component == "ner":
        from trialmatchai.finetuning.ner import NERFinetuneConfig, finetune_ner

        finetune_ner(
            NERFinetuneConfig(
                train_data=args.train_data,
                output_dir=args.output_dir,
                base_model=args.base_model,
                eval_data=args.eval_data,
                epochs=args.epochs,
                batch_size=args.batch_size,
                encoder_lr=args.encoder_lr,
                task_lr=args.task_lr,
                use_lora=not args.no_lora,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                fp16=not args.fp32,
                schema_path=args.schema_path,
                max_examples=args.max_examples,
            )
        )
    else:  # pragma: no cover - argparse enforces choices
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
