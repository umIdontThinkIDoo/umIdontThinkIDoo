#!/usr/bin/env python3
"""
Supervised Fine-Tuning (SFT) with TRL + QLoRA.

Usage (via training_routes.py):
  python sft_train.py --model-id MODEL --dataset DATASET_PATH [OPTIONS]

Expected dataset format: JSONL with {"messages": [{"role":..,"content":..}, ...]}
or Hugging Face dataset id.

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
    p.add_argument("--dataset", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--output-name", default=None)
    return p.parse_args()


def log(msg):
    print(msg, flush=True)


def main():
    args = _parse()
    output_name = args.output_name or f"sft-{args.job_id}"

    data_dir = Path(os.environ.get("DATA_DIR", "data"))
    output_dir = data_dir / "lora_adapters" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    log(f"[SFT] Job {args.job_id} starting")
    log(f"[SFT] Model: {args.model_id}")
    log(f"[SFT] Dataset: {args.dataset or '(none — provide via --dataset)'}")
    log(f"[SFT] Output: {output_dir}")
    log(f"[SFT] LoRA rank={args.lora_rank}, alpha={args.lora_alpha}, lr={args.lr}")

    try:
        from datasets import load_dataset
        from peft import LoraConfig, TaskType
        from transformers import AutoTokenizer, TrainingArguments
        from trl import SFTTrainer, SFTConfig
    except ImportError as e:
        log(f"[ERROR] Missing dependency: {e}")
        log("[ERROR] Install: pip install trl peft transformers datasets bitsandbytes")
        sys.exit(1)

    import torch

    log(f"[SFT] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"[SFT] GPU: {torch.cuda.get_device_name(0)}")

    # Load dataset
    if args.dataset:
        dset_path = Path(args.dataset)
        if dset_path.exists():
            log(f"[SFT] Loading local dataset from {dset_path}")
            ds = load_dataset("json", data_files=str(dset_path), split="train")
        else:
            log(f"[SFT] Loading HuggingFace dataset: {args.dataset}")
            ds = load_dataset(args.dataset, split="train")
    else:
        log("[SFT] No dataset provided. Using tiny synthetic example.")
        from datasets import Dataset
        ds = Dataset.from_list([
            {"messages": [
                {"role": "user", "content": "What is supervised fine-tuning?"},
                {"role": "assistant", "content": "SFT trains a model to follow instructions by showing it examples of desired outputs."}
            ]}
        ] * 8)

    log(f"[SFT] Dataset size: {len(ds)} examples")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
    )

    train_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        fp16=torch.cuda.is_available(),
        bf16=False,
        logging_steps=5,
        save_steps=50,
        seed=args.seed,
        report_to="none",
    )

    from transformers import AutoModelForCausalLM
    load_in_4bit = torch.cuda.is_available()

    quantization_config = None
    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            log("[SFT] Using 4-bit QLoRA quantization")
        except Exception:
            log("[SFT] bitsandbytes 4-bit not available — loading in full precision")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=quantization_config,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        torch_dtype=torch.float16 if (torch.cuda.is_available() and not quantization_config) else None,
    )

    trainer = SFTTrainer(
        model=model,
        args=train_config,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    log("[SFT] Starting training...")
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    log(f"[SFT] Adapter saved to {output_dir}")
    log("[TRAINING_OK]")


if __name__ == "__main__":
    main()
