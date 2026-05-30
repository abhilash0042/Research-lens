"""
train_embedder.py
=================
Fine-tunes all-MiniLM-L6-v2 on academic QA triplets using
TripletLoss (contrastive learning).

What this teaches the model:
  - Academic questions should be close to their evidence paragraphs
  - Academic questions should be far from irrelevant paragraphs
  - Domain-specific terminology maps correctly

Architecture:
  Base:      sentence-transformers/all-MiniLM-L6-v2
  Loss:      TripletLoss (cosine distance)
  Evaluator: InformationRetrievalEvaluator (Recall@5, Recall@10, MRR)
  Output:    models/embedder/

Hardware:
  RTX 4050 8GB VRAM — batch_size=32 is comfortable
  Training time: ~20-25 minutes

Run:
  python -m finetune.train_embedder
"""

import os
import json
import logging
import torch
from typing import List, Dict
from torch.utils.data import DataLoader
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    evaluation
)
import wandb
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────

CONFIG = {
    # Model
    "base_model":     "sentence-transformers/all-MiniLM-L6-v2",
    "output_path":    "models/embedder",

    # Data
    "train_path":     "data/processed/triplets/train.jsonl",
    "val_path":       "data/processed/triplets/val.jsonl",

    # Training — tuned for RTX 4050 8GB
    "batch_size":     32,
    "epochs":         3,
    "warmup_ratio":   0.1,       # 10% of steps for warmup
    "lr":             2e-5,
    "max_train":      None,      # set to int to limit for quick testing e.g. 5000

    # Triplet loss margin
    # Higher margin = stricter separation between positive and negative
    # 0.5 is standard, 0.3 works well for academic text
    "triplet_margin": 0.3,

    # Evaluation
    "eval_batch_size": 64,

    # W&B
    "wandb_project":  "researchlens",
    "wandb_run":      "embedder-finetune",
}


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_triplets(path: str, max_samples: int = None) -> List[InputExample]:
    """
    Load triplets and convert to sentence-transformers InputExample format.
    InputExample with 3 texts = (anchor, positive, negative) for TripletLoss.
    """
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            item = json.loads(line)
            examples.append(InputExample(
                texts=[
                    item["anchor"],    # question
                    item["positive"],  # evidence that answers it
                    item["negative"]   # irrelevant chunk
                ]
            ))
    log.info(f"Loaded {len(examples)} triplets from {path}")
    return examples


def build_ir_evaluator(val_path: str, max_samples: int = 500):
    """
    InformationRetrievalEvaluator measures retrieval quality.
    
    Metrics:
      Recall@5:  is the positive in top-5 retrieved?
      Recall@10: is the positive in top-10 retrieved?
      MRR:       mean reciprocal rank of the positive
      NDCG@10:   normalized discounted cumulative gain
    
    We build a small corpus from val triplets:
      queries   = {id: question}
      corpus    = {id: paragraph}
      relevant  = {query_id: {relevant_corpus_id}}
    """
    queries   = {}
    corpus    = {}
    relevant  = {}
    
    with open(val_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            item = json.loads(line)
            
            q_id = f"q_{i}"
            p_id = f"p_{i}"
            
            queries[q_id]  = item["anchor"]
            corpus[p_id]   = item["positive"]
            relevant[q_id] = {p_id}
            
            # Also add negatives to corpus (makes evaluation realistic)
            n_id = f"n_{i}"
            corpus[n_id] = item["negative"]
    
    evaluator = evaluation.InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant,
        name="academic-retrieval",
        show_progress_bar=False,
        batch_size=64,
        score_functions={"cosine": evaluation.InformationRetrievalEvaluator.cos_sim}
    )
    log.info(f"Built IR evaluator: {len(queries)} queries, {len(corpus)} corpus docs")
    return evaluator


# ─── Training ─────────────────────────────────────────────────────────────────

def train():
    log.info("="*60)
    log.info("EMBEDDER FINE-TUNING")
    log.info("="*60)
    log.info(f"Base model: {CONFIG['base_model']}")
    log.info(f"Device: {'CUDA — ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    log.info(f"VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB" if torch.cuda.is_available() else "")

    # ── W&B init ─────────────────────────────────────────────────────────────
    wandb.init(
        project=CONFIG["wandb_project"],
        name=CONFIG["wandb_run"],
        config=CONFIG
    )

    # ── Load model ───────────────────────────────────────────────────────────
    log.info(f"\nLoading base model: {CONFIG['base_model']}")
    model = SentenceTransformer(CONFIG["base_model"])
    
    # Log baseline performance before training
    log.info("Evaluating BASE model before training...")

    # ── Load data ────────────────────────────────────────────────────────────
    train_examples = load_triplets(CONFIG["train_path"], CONFIG["max_train"])
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=CONFIG["batch_size"]
    )

    # ── Loss function ─────────────────────────────────────────────────────────
    # TripletLoss with cosine distance
    # Loss = max(cos(anchor,neg) - cos(anchor,pos) + margin, 0)
    # Forces: anchor closer to positive than negative by at least margin
    train_loss = losses.TripletLoss(
        model=model,
        distance_metric=losses.TripletDistanceMetric.COSINE,
        triplet_margin=CONFIG["triplet_margin"]
    )

    # ── Evaluator ────────────────────────────────────────────────────────────
    evaluator = build_ir_evaluator(CONFIG["val_path"])

    # ── Warmup steps ─────────────────────────────────────────────────────────
    total_steps   = len(train_dataloader) * CONFIG["epochs"]
    warmup_steps  = int(total_steps * CONFIG["warmup_ratio"])
    log.info(f"\nTraining: {len(train_examples)} examples | {CONFIG['epochs']} epochs")
    log.info(f"Steps: {total_steps} total | {warmup_steps} warmup")
    log.info(f"Batch size: {CONFIG['batch_size']} | LR: {CONFIG['lr']}")

    # ── Train ────────────────────────────────────────────────────────────────
    os.makedirs(CONFIG["output_path"], exist_ok=True)
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=CONFIG["epochs"],
        warmup_steps=warmup_steps,
        optimizer_params={"lr": CONFIG["lr"]},
        output_path=CONFIG["output_path"],
        evaluation_steps=500,          # evaluate every 500 steps
        save_best_model=True,          # saves best checkpoint automatically
        show_progress_bar=True,
        callback=lambda score, epoch, steps: wandb.log({
            "eval/ndcg@10": score,
            "epoch": epoch,
            "steps": steps
        })
    )

    log.info(f"\nBest model saved → {CONFIG['output_path']}")

    # ── Final evaluation ─────────────────────────────────────────────────────
    log.info("\nFinal evaluation of fine-tuned model...")
    final_score = evaluator(model)
    wandb.log({"final/ndcg@10": final_score})
    
    log.info("="*60)
    log.info(f"TRAINING COMPLETE — model saved to {CONFIG['output_path']}")
    log.info("="*60)
    log.info("Next step: run finetune/train_reranker.py")
    
    wandb.finish()


if __name__ == "__main__":
    # Quick sanity check before full training
    if not os.path.exists(CONFIG["train_path"]):
        log.error(f"Training data not found: {CONFIG['train_path']}")
        log.error("Run: python -m finetune.prep_data first")
        exit(1)
    train()
