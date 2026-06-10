#!/usr/bin/env python3
"""
Student-to-PhD RL Loop using TRL GRPOTrainer.

Pipeline:
  1. Parse the book (PDF/EPUB/TXT) into passages
  2. Generate Q&A pairs at each curriculum level (Student → PhD)
  3. Use GRPO to reinforce good answers with a learned reward signal
  4. Progress through levels; save adapter after each level

The "student to PhD" system creates 5 curriculum levels:
  L0 student:       simple factual questions from the text
  L1 intermediate:  explanation and comparison questions
  L2 advanced:      analysis and synthesis questions
  L3 expert:        application and critique questions
  L4 phd:           novel hypothesis generation from the material

The model is rewarded based on similarity to reference answers extracted
from the book, pushing it to become a genuine knowledge expert.

Output: DATA_DIR/lora_adapters/<output-name>/
"""

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Optional


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True)
    p.add_argument("--book", required=True, help="Path to PDF/EPUB/TXT book file")
    p.add_argument("--job-id", required=True)
    p.add_argument("--log-file", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--output-name", default=None)
    # Unused
    p.add_argument("--dataset", default=None)
    p.add_argument("--base-adapter", default=None)
    return p.parse_args()


LEVEL_NAMES = ["Student", "Intermediate", "Advanced", "Expert", "PhD"]

LEVEL_PROMPTS = {
    0: "You are a curious student reading this material for the first time. Ask a simple, factual question about the following passage. Provide the answer as found in the text.",
    1: "You are studying this topic at an intermediate level. Ask a question that requires explaining a concept or comparing two ideas from the passage. Provide a thorough answer.",
    2: "You are an advanced student. Ask a question that requires analyzing the passage — identifying patterns, causes, or implications. Give a detailed analytical answer.",
    3: "You are an expert. Ask a question that requires applying the ideas to a novel scenario or critiquing an argument in the passage. Provide a sophisticated answer.",
    4: "You are a PhD researcher. Based on this passage, formulate a novel research hypothesis that could be tested. Explain the theoretical basis and potential methodology.",
}

SYSTEM_PROMPTS = {
    0: "You are a knowledgeable tutor. Answer questions clearly and accurately based on the material provided, at a level appropriate for a student encountering this topic for the first time.",
    1: "You are an experienced instructor. Provide thorough explanations that build understanding. Connect ideas and clarify concepts at an intermediate level.",
    2: "You are a subject matter expert. Provide rigorous, analytical answers that demonstrate deep understanding of the material and its implications.",
    3: "You are a leading expert in this field. Your answers synthesize broad knowledge, critique assumptions, and apply theory to novel contexts.",
    4: "You are a world-class researcher. Your responses demonstrate mastery equivalent to a PhD: you identify research gaps, propose hypotheses, connect disparate ideas, and reason at the frontier of the field.",
}


def log(msg):
    print(msg, flush=True)


def _extract_text_from_book(book_path: Path) -> str:
    """Extract raw text from PDF, EPUB, or TXT/MD."""
    ext = book_path.suffix.lower()
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(book_path))
            return "\n".join(p.extract_text() or "" for p in reader.pages)
        except ImportError:
            log("[RL] pypdf not installed — install: pip install pypdf")
            sys.exit(1)
        except Exception as e:
            log(f"[RL] PDF extraction error: {e}")
            sys.exit(1)
    elif ext == ".epub":
        try:
            import ebooklib
            from ebooklib import epub
            from html.parser import HTMLParser

            class _Text(HTMLParser):
                def __init__(self): super().__init__(); self._chunks = []
                def handle_data(self, d): self._chunks.append(d)
                def get(self): return " ".join(self._chunks)

            book = epub.read_epub(str(book_path))
            parts = []
            for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                h = _Text(); h.feed(item.get_body_content().decode("utf-8", errors="ignore")); parts.append(h.get())
            return "\n".join(parts)
        except ImportError:
            log("[RL] ebooklib not installed — install: pip install ebooklib")
            sys.exit(1)
    else:
        return book_path.read_text(errors="ignore")


def _chunk_text(text: str, size: int = 600, overlap: int = 100) -> list[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+size]))
        i += size - overlap
    return [c for c in chunks if len(c.split()) > 50]


def _generate_qa(model, tokenizer, passage: str, level: int, device) -> Optional[tuple[str, str]]:
    """Generate a Q&A pair at the given curriculum level for a passage."""
    prompt_text = LEVEL_PROMPTS[level]
    full_prompt = f"{prompt_text}\n\nPassage:\n{passage[:2000]}\n\nQuestion and answer:"

    import torch
    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    # Try to split into Q and A
    for sep in ["Answer:", "A:", "\n\n"]:
        if sep in generated:
            parts = generated.split(sep, 1)
            q = parts[0].replace("Question:", "").strip()
            a = parts[1].strip()
            if q and a:
                return q, a
    # Fallback: treat whole output as question, passage as answer
    return generated.strip()[:200], passage[:300]


def _reward_fn(generated: str, reference: str) -> float:
    """Simple lexical overlap reward in [0, 1]."""
    gen_words = set(re.findall(r"\w+", generated.lower()))
    ref_words = set(re.findall(r"\w+", reference.lower()))
    if not ref_words:
        return 0.0
    overlap = len(gen_words & ref_words) / len(ref_words)
    length_bonus = min(1.0, len(generated.split()) / 50) * 0.2
    return min(1.0, overlap + length_bonus)


def main():
    args = _parse()
    output_name = args.output_name or f"rl-loop-{args.job_id}"
    random.seed(args.seed)

    data_dir = Path(os.environ.get("DATA_DIR", "data"))
    output_dir = data_dir / "lora_adapters" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    book_path = Path(args.book)
    if not book_path.exists():
        log(f"[ERROR] Book not found: {book_path}")
        sys.exit(1)

    log(f"[RL-Loop] Job {args.job_id} starting")
    log(f"[RL-Loop] Model: {args.model_id}")
    log(f"[RL-Loop] Book: {book_path.name}")
    log(f"[RL-Loop] Seed: {args.seed}")
    log(f"[RL-Loop] Output: {output_dir}")

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as e:
        log(f"[ERROR] Missing dependency: {e}")
        log("[ERROR] Install: pip install trl peft transformers datasets bitsandbytes pypdf ebooklib")
        sys.exit(1)

    log(f"[RL-Loop] CUDA: {torch.cuda.is_available()}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Extract book text ──────────────────────────────────────────────────
    log(f"[RL-Loop] Extracting text from {book_path.suffix.lower()} ...")
    text = _extract_text_from_book(book_path)
    if not text.strip():
        log("[ERROR] Could not extract text from book")
        sys.exit(1)

    chunks = _chunk_text(text)
    log(f"[RL-Loop] Book: {len(text.split())} words → {len(chunks)} passages")

    # ── Load model ────────────────────────────────────────────────────────
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

    if quantization_config:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Curriculum loop: Student → PhD ────────────────────────────────────
    max_chunks_per_level = max(8, len(chunks) // 5)

    for level in range(5):
        level_name = LEVEL_NAMES[level]
        sys_prompt = SYSTEM_PROMPTS[level]
        log(f"\n[RL-Loop] ━━ Level {level}: {level_name} ━━")

        # Sample passages for this level
        level_chunks = random.sample(chunks, min(max_chunks_per_level, len(chunks)))
        log(f"[RL-Loop] Generating {len(level_chunks)} Q&A pairs at {level_name} level...")

        # Generate Q&A pairs for this level
        pairs = []
        for i, passage in enumerate(level_chunks):
            qa = _generate_qa(model, tokenizer, passage, level, device)
            if qa:
                q, a = qa
                pairs.append({
                    "prompt": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": q},
                    ],
                    "reference": a,
                    "passage": passage[:500],
                })
            if (i + 1) % 10 == 0:
                log(f"[RL-Loop]   Generated {i+1}/{len(level_chunks)} pairs")

        if not pairs:
            log(f"[RL-Loop] No pairs generated for level {level} — skipping")
            continue

        log(f"[RL-Loop] Training on {len(pairs)} pairs at {level_name} level...")

        # Build GRPO dataset
        def _make_reward(reference: str):
            def reward_fn(completions, **kwargs):
                return [_reward_fn(c, reference) for c in completions]
            return reward_fn

        # GRPO expects: list of dicts with "prompt" key and a reward_fn
        grpo_data = [{"prompt": p["prompt"]} for p in pairs]
        ds = Dataset.from_list(grpo_data)
        references = [p["reference"] for p in pairs]

        def batched_reward_fn(completions, prompts=None, **kwargs):
            scores = []
            for i, comp in enumerate(completions):
                ref = references[i % len(references)]
                scores.append(_reward_fn(comp if isinstance(comp, str) else str(comp), ref))
            return scores

        grpo_config = GRPOConfig(
            output_dir=str(output_dir / f"level_{level}"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            fp16=torch.cuda.is_available(),
            logging_steps=5,
            seed=args.seed,
            report_to="none",
            num_generations=2,
            max_new_tokens=128,
        )

        trainer = GRPOTrainer(
            model=model,
            args=grpo_config,
            reward_funcs=batched_reward_fn,
            train_dataset=ds,
            processing_class=tokenizer,
        )

        trainer.train()
        log(f"[RL-Loop] Level {level} ({level_name}) complete ✓")

        # Save checkpoint after each level
        level_out = output_dir / f"level_{level}_checkpoint"
        model.save_pretrained(str(level_out))
        tokenizer.save_pretrained(str(level_out))
        log(f"[RL-Loop] Level checkpoint saved: {level_out}")

    # ── Final save ────────────────────────────────────────────────────────
    log(f"\n[RL-Loop] Saving final PhD-level adapter to {output_dir} ...")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Write metadata
    meta = {
        "book": book_path.name,
        "model_id": args.model_id,
        "job_id": args.job_id,
        "levels_completed": 5,
        "seed": args.seed,
    }
    (output_dir / "training_meta.json").write_text(json.dumps(meta, indent=2))

    log("[RL-Loop] Training complete — model is now a knowledge expert on this material.")
    log("[TRAINING_OK]")


if __name__ == "__main__":
    main()
