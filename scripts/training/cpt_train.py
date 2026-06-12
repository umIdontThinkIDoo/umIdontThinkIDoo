#!/usr/bin/env python3
"""
Continued Pre-Training (CPT) with TRL + QLoRA.

Used to inject domain knowledge into a model before SFT/DPO.
Dataset format: JSONL with {"text": str} or plain .txt files.

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
    output_name = args.output_name or f"cpt-{args.job_id}"

    data_dir = Path(os.environ.get("DATA_DIR", "data"))
    output_dir = data_dir / "lora_adapters" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    log(f"[CPT] Job {args.job_id} starting")
    log(f"[CPT] Model: {args.model_id}")
    log(f"[CPT] Dataset: {args.dataset or '(none)'}")
    log(f"[CPT] Output: {output_dir}")

    try:
        from datasets import Dataset, load_dataset
        from peft import LoraConfig, TaskType
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForLanguageModeling
        from trl import SFTConfig, SFTTrainer
    except ImportError as e:
        log(f"[ERROR] Missing dependency: {e}")
        log("[ERROR] Install: pip install trl peft transformers datasets bitsandbytes")
        sys.exit(1)

    import torch

    log(f"[CPT] CUDA: {torch.cuda.is_available()}")

    # Load raw text dataset
    if args.dataset:
        dset_path = Path(args.dataset)
        if dset_path.exists() and dset_path.suffix == ".txt":
            text = dset_path.read_text(errors="ignore")
            # Chunk into ~512-word passages
            words = text.split()
            passages = [" ".join(words[i:i+512]) for i in range(0, len(words), 384)]
            ds = Dataset.from_list([{"text": p} for p in passages if len(p) > 50])
        elif dset_path.exists():
            ds = load_dataset("json", data_files=str(dset_path), split="train")
        else:
            ds = load_dataset(args.dataset, split="train")
    else:
        log("[CPT] No dataset — using tiny synthetic corpus")
        ds = Dataset.from_list([{"text": "Language models learn from large corpora of text. Continued pre-training extends this with domain-specific data."}] * 16)

    log(f"[CPT] Dataset size: {len(ds)} passages")

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
        # Force fp16: with dtype unset, non-quantized modules and LoRA adapters
        # take the checkpoint's native dtype (often bf16), and the fp16
        # GradScaler crashes in unscale_. fp16 is also the only safe choice on
        # Pascal (cc 6.0 has no bf16).
        dtype=torch.float16 if torch.cuda.is_available() else None,
    )

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
        logging_steps=5,
        save_steps=50,
        seed=args.seed,
        report_to="none",
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=train_config,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # TRL force-casts QLoRA adapter weights to bf16 on 4-bit models regardless
    # of fp16/bf16 args; bf16 grads crash the fp16 GradScaler and Pascal
    # (cc 6.0) has no bf16 at all. Cast trainables back to fp32 (peft's own
    # QLoRA layout) so the scaler sees supported dtypes.
    for param in trainer.model.parameters():
        if param.requires_grad and param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float32)

    log("[CPT] Starting training...")
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    log(f"[CPT] Adapter saved to {output_dir}")
    log("[TRAINING_OK]")


if __name__ == "__main__":
    main()
