#!/usr/bin/env python3
"""
Direct Preference Optimization (DPO) with TRL + QLoRA.

Dataset format: JSONL with {"prompt": str, "chosen": str, "rejected": str}
or a HuggingFace dataset with those columns.

Outputs adapter to DATA_DIR/lora_adapters/<output-name>/
"""

import argparse
import os
import sys
from pathlib import Path


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True)
    p.add_argument("--job-id", required=True)
    p.add_argument("--log-file", required=True)
    p.add_argument("--dataset", default="trl-internal-testing/hh-rlhf-helpful-base-trl-style")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--output-name", default=None)
    return p.parse_args()


def log(msg):
    print(msg, flush=True)


def main():
    args = _parse()
    output_name = args.output_name or f"dpo-{args.job_id}"

    data_dir = Path(os.environ.get("DATA_DIR", "data"))
    output_dir = data_dir / "lora_adapters" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    log(f"[DPO] Job {args.job_id} starting")
    log(f"[DPO] Model: {args.model_id}")
    log(f"[DPO] Dataset: {args.dataset}")
    log(f"[DPO] Output: {output_dir}")

    try:
        from datasets import load_dataset
        from peft import LoraConfig, TaskType
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import DPOConfig, DPOTrainer
    except ImportError as e:
        log(f"[ERROR] Missing dependency: {e}")
        log("[ERROR] Install: pip install trl peft transformers datasets bitsandbytes")
        sys.exit(1)

    import torch

    log(f"[DPO] CUDA: {torch.cuda.is_available()}")

    dset_path = Path(args.dataset)
    if dset_path.exists():
        ds = load_dataset("json", data_files=str(dset_path), split="train")
    else:
        ds = load_dataset(args.dataset, split="train")

    log(f"[DPO] Dataset size: {len(ds)} examples")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if torch.cuda.is_available():
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        except Exception:
            pass

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=quantization_config,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
    )

    dpo_config = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        fp16=torch.cuda.is_available(),
        logging_steps=5,
        save_steps=50,
        seed=args.seed,
        report_to="none",
        beta=0.1,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    log("[DPO] Starting training...")
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    log(f"[DPO] Adapter saved to {output_dir}")
    log("[TRAINING_OK]")


if __name__ == "__main__":
    main()
