"""
train_generator.py
==================
Fine-tunes Mistral-7B-Instruct-v0.2 using LoRA + 4-bit quantization
on academic cited QA examples from QASPER.

What this teaches the model:
  - Answer ONLY from provided context (no hallucination)
  - Every claim must cite its source: [SOURCE N: paper, section]
  - Say "Not found in provided papers" when context lacks the answer
  - Academic answer style: precise, specific, concise

Why LoRA instead of full fine-tuning:
  Full fine-tuning: update all 7B params → needs 28GB VRAM ❌
  LoRA: freeze 7B params, add small A,B matrices per attention layer
        W' = W + BA  where B,A are low-rank (r=16)
        Only trains ~0.3% of parameters → fits in 6GB VRAM ✅
        Quality difference: ~3-5% vs full fine-tuning

Why 4-bit quantization:
  Mistral-7B in fp16: ~14GB VRAM  ❌
  Mistral-7B in 4-bit: ~4.5GB VRAM ✅
  Technique: NF4 (NormalFloat4) quantization via bitsandbytes
  Compute still in fp16 (bnb_4bit_compute_dtype)

Hardware:
  RTX 4050 8GB VRAM
  batch_size=2, gradient_accumulation=16 → effective batch=32
  Training time: ~60-90 minutes

Run:
  python -m finetune.train_generator
"""

import os
import json
import logging
import torch
from typing import List, Dict
from dataclasses import dataclass
from datasets import Dataset
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────

CONFIG = {
    # Model
    "base_model":      "mistralai/Mistral-7B-Instruct-v0.2",
    "output_path":     "models/generator",

    # Data
    "train_path":      "data/processed/qa_examples/train.jsonl",
    "val_path":        "data/processed/qa_examples/val.jsonl",
    "max_train":       None,     # set to int to limit e.g. 2000

    # LoRA config
    # r=16: rank of the low-rank matrices — higher = more params, more quality
    # alpha=32: scaling factor = alpha/r = 2x scaling
    # target_modules: which attention layers to add LoRA to
    # dropout=0.05: small dropout on LoRA layers for regularization
    "lora_r":           16,
    "lora_alpha":       32,
    "lora_dropout":     0.05,
    "lora_target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",  # attention projections
        "gate_proj", "up_proj", "down_proj"        # MLP layers
    ],

    # Training — tuned for RTX 4050 8GB
    # Effective batch = batch_size * gradient_accumulation = 2 * 16 = 32
    "batch_size":              2,
    "gradient_accumulation":   16,
    "epochs":                  3,
    "lr":                      2e-4,        # higher LR than full fine-tuning (LoRA standard)
    "warmup_ratio":            0.05,
    "max_seq_length":          1024,        # max input+output tokens
    "lr_scheduler":            "cosine",

    # Quantization
    "load_in_4bit":            True,
    "bnb_4bit_quant_type":     "nf4",       # NormalFloat4 — best quality
    "bnb_4bit_compute_dtype":  "float16",   # compute in fp16 for speed
    "bnb_4bit_double_quant":   True,        # quantize quantization constants too

    # W&B
    "wandb_project":  "researchlens",
    "wandb_run":      "generator-finetune-lora",
}

# ─── Prompt template ──────────────────────────────────────────────────────────
# This exact format is used BOTH during training and inference.
# Mistral uses [INST] ... [/INST] instruction format.

SYSTEM_PROMPT = """You are ResearchLens, an expert research assistant.
Answer questions ONLY using the provided source documents.
Every factual claim in your answer MUST include a citation in the format [SOURCE N: paper_title, section].
If the answer cannot be found in the provided sources, respond with: "Not found in the provided papers."
Be precise, specific, and concise. Avoid vague or generic statements."""

def format_prompt(question: str, context: str) -> str:
    """Format a training example as Mistral instruction prompt."""
    return f"""<s>[INST] <<SYS>>
{SYSTEM_PROMPT}
<</SYS>>

SOURCES:
{context}

QUESTION: {question} [/INST]"""

def format_full_example(question: str, context: str, answer: str) -> str:
    """Format full training example including answer (for SFT training)."""
    prompt = format_prompt(question, context)
    return f"{prompt} {answer} </s>"


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_qa_examples(path: str, max_samples: int = None) -> List[Dict]:
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            item = json.loads(line)
            examples.append({
                "text": format_full_example(
                    item["question"],
                    item["context"],
                    item["answer"]
                )
            })
    log.info(f"Loaded {len(examples)} QA examples from {path}")
    return examples


# ─── Model setup ──────────────────────────────────────────────────────────────

def load_quantized_model(model_name: str):
    """
    Load Mistral-7B with 4-bit NF4 quantization.
    
    BitsAndBytesConfig breakdown:
      load_in_4bit: store weights as 4-bit integers
      bnb_4bit_quant_type="nf4": NormalFloat4 — quantization that
        preserves normal distribution of weights (better than int4)
      bnb_4bit_compute_dtype=fp16: during forward pass, dequantize
        to fp16 for computation (quality/speed tradeoff)
      bnb_4bit_use_double_quant: also quantize the scale factors
        → saves extra ~0.3GB VRAM
    """
    log.info("Loading quantization config (4-bit NF4)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=CONFIG["load_in_4bit"],
        bnb_4bit_quant_type=CONFIG["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=CONFIG["bnb_4bit_double_quant"],
    )
    
    log.info(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",              # automatically puts layers on GPU/CPU
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    
    # Required before applying LoRA to quantized model
    # Casts layer norms to float32 for stability
    model = prepare_model_for_kbit_training(model)
    
    log.info(f"Model loaded — VRAM used: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    return model


def apply_lora(model):
    """
    Apply LoRA adapters to the quantized model.
    
    LoRA math:
      Original: y = Wx
      LoRA:     y = Wx + (B·A)x  where B ∈ R^(d×r), A ∈ R^(r×k)
      
      r=16 means the update matrix is factored through rank-16
      Much smaller than full d×k matrix
      
    We target ALL attention and MLP projection layers because:
      q_proj, k_proj, v_proj, o_proj → attention transformation
      gate_proj, up_proj, down_proj  → feed-forward network
    This gives best quality vs targeting only attention layers.
    """
    log.info(f"Applying LoRA (r={CONFIG['lora_r']}, alpha={CONFIG['lora_alpha']})...")
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        target_modules=CONFIG["lora_target_modules"],
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    
    # Log trainable parameter count
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    log.info(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    return model


# ─── Training ─────────────────────────────────────────────────────────────────

def train():
    log.info("="*60)
    log.info("GENERATOR FINE-TUNING (Mistral-7B + LoRA + 4-bit)")
    log.info("="*60)
    log.info(f"Device: {'CUDA — ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    log.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB" if torch.cuda.is_available() else "")

    # ── W&B init ─────────────────────────────────────────────────────────────
    wandb.init(
        project=CONFIG["wandb_project"],
        name=CONFIG["wandb_run"],
        config=CONFIG
    )

    # ── Tokenizer ────────────────────────────────────────────────────────────
    log.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["base_model"],
        trust_remote_code=True
    )
    # Mistral has no pad token by default — set to eos token
    # This is required for batch training
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"   # pad on right for causal LM

    # ── Model ─────────────────────────────────────────────────────────────────
    model = load_quantized_model(CONFIG["base_model"])
    model = apply_lora(model)

    # ── Data ─────────────────────────────────────────────────────────────────
    train_data = load_qa_examples(CONFIG["train_path"], CONFIG["max_train"])
    val_data   = load_qa_examples(CONFIG["val_path"])
    
    train_dataset = Dataset.from_list(train_data)
    val_dataset   = Dataset.from_list(val_data)

    # ── Training arguments ───────────────────────────────────────────────────
    # bf16 is better than fp16 on RTX 4050 (Ampere architecture)
    # gradient_checkpointing saves VRAM at cost of ~20% speed
    training_args = TrainingArguments(
        output_dir=CONFIG["output_path"],
        num_train_epochs=CONFIG["epochs"],
        per_device_train_batch_size=CONFIG["batch_size"],
        per_device_eval_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation"],
        learning_rate=CONFIG["lr"],
        lr_scheduler_type=CONFIG["lr_scheduler"],
        warmup_ratio=CONFIG["warmup_ratio"],
        fp16=True,                         # use fp16 on RTX 4050
        bf16=False,                        # bf16 needs Ampere A100+
        gradient_checkpointing=True,       # save VRAM
        optim="paged_adamw_8bit",          # memory-efficient optimizer
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,                # keep only 2 checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="wandb",
        run_name=CONFIG["wandb_run"],
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    # ── SFT Trainer ──────────────────────────────────────────────────────────
    # SFTTrainer (Supervised Fine-Tuning Trainer from trl library)
    # Handles: proper loss masking (only compute loss on answer tokens, not prompt)
    # This is critical — we don't want to penalize the model for the prompt tokens
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=CONFIG["max_seq_length"],
        packing=False,              # don't pack multiple examples per sequence
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    log.info(f"\nTraining: {len(train_data)} examples | {CONFIG['epochs']} epochs")
    log.info(f"Effective batch size: {CONFIG['batch_size'] * CONFIG['gradient_accumulation']}")
    log.info(f"Max sequence length: {CONFIG['max_seq_length']} tokens")
    log.info("\nStarting training...")
    
    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────────
    log.info(f"\nSaving LoRA adapters → {CONFIG['output_path']}")
    trainer.model.save_pretrained(CONFIG["output_path"])
    tokenizer.save_pretrained(CONFIG["output_path"])
    
    # Also save training config for reproducibility
    with open(os.path.join(CONFIG["output_path"], "train_config.json"), "w") as f:
        json.dump(CONFIG, f, indent=2)

    log.info("="*60)
    log.info("TRAINING COMPLETE")
    log.info("="*60)
    log.info(f"LoRA adapters saved → {CONFIG['output_path']}")
    log.info("At inference: load base Mistral-7B + merge these adapters")
    log.info("Next step: run src/pipeline.py to test end-to-end")
    
    wandb.finish()


if __name__ == "__main__":
    if not os.path.exists(CONFIG["train_path"]):
        log.error(f"Training data not found: {CONFIG['train_path']}")
        log.error("Run: python -m finetune.prep_data first")
        exit(1)

    # VRAM check
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(f"VRAM available: {vram_gb:.1f}GB")
        if vram_gb < 7:
            log.warning("Less than 7GB VRAM detected. Reduce batch_size to 1.")
    
    train()
