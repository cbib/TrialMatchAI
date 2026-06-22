import time
from typing import Dict, List

from trialmatchai.matching.eligibility_base import BaseTrialProcessor
from trialmatchai.utils.logging_config import setup_logging

logger = setup_logging(__name__)

try:
    import torch
except ImportError:  # pragma: no cover - exercised by lean package imports
    torch = None  # type: ignore[assignment]


def _require_torch():
    if torch is None:
        raise RuntimeError(
            "BatchTrialProcessor requires PyTorch. Install the ML extras with "
            "`uv sync --extra llm` or `pip install 'trialmatchai[llm]'`."
        )
    return torch


class BatchTrialProcessor(BaseTrialProcessor):
    def __init__(
        self,
        model,
        tokenizer,
        device: int,
        batch_size: int = 4,
        use_cot: bool = True,
        max_new_tokens: int = 5000,  # keep long answers
    ):
        """
        Optimized for throughput while preserving long outputs.

        Key improvements:
        - model.eval(), use_cache, TF32 hints, and SDPA/FlashAttention2 when available
        - length bucketing (sort by prompt token length) to reduce padding waste
        - telemetry for tokens/sec and stage timings
        """
        torch_module = _require_torch()
        self.device = device
        self.device_str = f"cuda:{device}"
        self.batch_size = batch_size
        self.model = model
        self.tokenizer = tokenizer
        self.use_cot = use_cot
        self.max_new_tokens = max_new_tokens

        # ---- Inference-time performance knobs (safe if available) ----
        self.model.eval()
        try:
            # Allow TF32 on Ampere+ (gives a free speedup for matmuls with minimal accuracy loss)
            torch_module.backends.cuda.matmul.allow_tf32 = True
            torch_module.set_float32_matmul_precision("high")
        except Exception:
            pass

        # Prefer fast attention kernels if supported by your install
        try:
            from transformers.utils.import_utils import is_flash_attn_2_available

            if hasattr(self.model, "config"):
                if is_flash_attn_2_available():
                    self.model.config.attn_implementation = "flash_attention_2"
                else:
                    # SDPA is the PyTorch fused attention (fast on recent torch)
                    self.model.config.attn_implementation = "sdpa"
        except Exception:
            # Fall back silently; HF will pick the best available
            pass

        # Ensure caching is on for generation
        try:
            if hasattr(self.model, "config") and hasattr(
                self.model.config, "use_cache"
            ):
                self.model.config.use_cache = True
        except Exception:
            pass

        # Pad token handling (avoids warnings for decoder-only models)
        try:
            if (
                self.tokenizer.pad_token_id is None
                and self.tokenizer.eos_token_id is not None
            ):
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception:
            pass

    def _progress_desc(self) -> str:
        return f"GPU {self.device} Processing Trials"

    # ---------------------- Core batch path ----------------------

    def _process_batch(self, batch: List[Dict], output_folder: str):
        """
        Expects a list of dicts with keys: nct_id, prompt
        """
        torch_module = _require_torch()
        try:
            t0 = time.time()
            # Tokenize once; pad to the longest in this batch only
            tokenized = self.tokenizer(
                [item["prompt"] for item in batch],
                padding=True,
                truncation=True,  # keep to model max length to avoid OOM
                return_tensors="pt",
            )
            input_len = tokenized["input_ids"].shape[1]
            tokenized = tokenized.to(self.device_str)
            t1 = time.time()

            # Autocast to model dtype if it's half/bfloat16 for extra speed
            model_dtype = next(self.model.parameters()).dtype
            use_autocast = model_dtype in (torch_module.float16, torch_module.bfloat16)

            with torch_module.inference_mode():
                ctx = (
                    torch_module.autocast(device_type="cuda", dtype=model_dtype)
                    if use_autocast
                    else torch_module.cuda.amp.autocast(enabled=False)
                )
                with ctx:
                    outputs = self.model.generate(
                        **tokenized,
                        max_new_tokens=self.max_new_tokens,  # long answers kept
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=self.tokenizer.pad_token_id
                        if self.tokenizer.pad_token_id is not None
                        else self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        num_return_sequences=1,
                        return_dict_in_generate=False,
                    )
            t2 = time.time()

            # Decode only the generated tail
            gen_len = outputs.shape[1] - input_len
            decoded_responses = self.tokenizer.batch_decode(
                outputs[:, input_len:], skip_special_tokens=True
            )

            # Persist outputs
            for item, response in zip(batch, decoded_responses):
                self._save_outputs(item["nct_id"], response, output_folder)

            # Telemetry
            total_gen_tokens = gen_len * len(batch)
            gen_time = t2 - t1
            logger.info(
                f"[GPU {self.device}] batch={len(batch)} | in_len={input_len} | "
                f"out_len≈{gen_len} | tokenize+H2D={t1 - t0:.2f}s | "
                f"generate={gen_time:.2f}s | ~{(total_gen_tokens / gen_time) if gen_time > 0 else 0:.1f} tok/s"
            )
        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")
            for item in batch:
                logger.error(f"Failed trial: {item['nct_id']}")
