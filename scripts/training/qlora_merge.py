#!/usr/bin/env python3
"""
QLoRA adapter merge — merges a LoRA adapter into the base model weights.

Takes a trained LoRA adapter directory and merges it with the base model,
producing a single full-weight model that can be served directly.

Output: DATA_DIR/lora_adapters/<output-name>-merged/
"""

import argparse
import os
import sys
from pathlib import Path


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True, help="Base model ID or path")
    p.add_argument("--base-adapter", required=True, help="Path to LoRA adapter directory")
    p.add_argument("--job-id", required=True)
    p.add_argument("--log-file", required=True)
    p.add_argument("--output-name", default=None)
    # Unused but accepted for API uniformity
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    return p.parse_args()


def log(msg):
    print(msg, flush=True)


def main():
    args = _parse()
    output_name = args.output_name or f"merged-{args.job_id}"

    data_dir = Path(os.environ.get("DATA_DIR", "data"))
    output_dir = data_dir / "lora_adapters" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter_path = Path(args.base_adapter)
    if not adapter_path.exists():
        log(f"[ERROR] Adapter path not found: {adapter_path}")
        sys.exit(1)

    log(f"[QLoRA-Merge] Job {args.job_id} starting")
    log(f"[QLoRA-Merge] Base model: {args.model_id}")
    log(f"[QLoRA-Merge] Adapter: {adapter_path}")
    log(f"[QLoRA-Merge] Output: {output_dir}")

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        log(f"[ERROR] Missing dependency: {e}")
        log("[ERROR] Install: pip install peft transformers torch")
        sys.exit(1)

    log(f"[QLoRA-Merge] CUDA: {torch.cuda.is_available()}")
    log("[QLoRA-Merge] Loading base model (float16, no quantization for merge)...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )

    log("[QLoRA-Merge] Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, str(adapter_path))

    log("[QLoRA-Merge] Merging weights...")
    model = model.merge_and_unload()

    log(f"[QLoRA-Merge] Saving merged model to {output_dir} ...")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    log("[QLoRA-Merge] Done — merged model can be loaded directly with from_pretrained")
    log("[TRAINING_OK]")


if __name__ == "__main__":
    main()
