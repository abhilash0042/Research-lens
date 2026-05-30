"""
train_reranker.py
=================
Fine-tunes cross-encoder/ms-marco-MiniLM-L-6-v2 on academic QA pairs
using binary cross-entropy loss.

What this teaches the model:
  - Given (query, chunk) together: score 1 if chunk answers query, 0 if not
  - Cross-attention lets query tokens attend directly to chunk tokens
  - Learns academic relevance signal vs generic web relevance

Architecture:
  Base:      cross-encoder/ms-marco-MiniLM-L-6-v2
  Input:     [CLS] query [SEP] chunk [SEP]
  Output:    single relevance score 0-1
  Loss:      Binary Cross Entropy
  Evaluator: CECorrelationEvaluator (correlation of scores vs labels)
  Output:    models/reranker/

Why CrossEncoder beats bi-encoder for reranking:
  Bi-encoder: encodes query and chunk SEPARATELY → cosine similarity
  CrossEncoder: encodes query+chunk TOGETHER → direct token-level attention
  Result: CrossEncoder sees "optimizer" in query attending to "AdamW" in chunk

Hardware:
  RTX 4050 8GB VRAM — batch_size=16 is safe
  Training time: ~15-20 minutes

Run:
  python -m finetune.train_reranker
"""

import os
import json
import logging
import torch
from typing import List, Tuple
from torch.utils.data import DataLoader
from sentence_transformers import InputExample
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────

CONFIG = {
    # Model
    "base_model":     "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "output_path":    "models/reranker",

    # Data
    "train_path":     "data/processed/pairs/train.jsonl",
    "val_path":       "data/processed/pairs/val.jsonl",

    # Training — tuned for RTX 4050 8GB
    # CrossEncoder is heavier than bi-encoder (joint encoding)
    # batch_size=16 keeps VRAM under 6GB
    "batch_size":     16,
    "epochs":         4,
    "warmup_ratio":   0.1,
    "lr":             2e-5,
    "max_train":      None,      # set e.g. 6000 for quick test
    "max_length":     512,       # max tokens for query+chunk combined

    # W&B
    "wandb_project":  "researchlens",
    "wandb_run":      "reranker-finetune",
}


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_pairs(path: str, max_samples: int = None) -> List[InputExample]:
    """
    Load (query, chunk, label) pairs.
    CrossEncoder InputExample takes texts=[query, chunk] and label=float.
    """
    examples = []
    pos_count = 0
    neg_count = 0
    
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            item = json.loads(line)
            label = float(item["label"])
            examples.append(InputExample(
                texts=[item["query"], item["chunk"]],
                label=label
            ))
            if label == 1.0:
                pos_count += 1
            else:
                neg_count += 1
    
    log.info(f"Loaded {len(examples)} pairs from {path}")
    log.info(f"  Positive: {pos_count} | Negative: {neg_count} | Ratio: 1:{neg_count//max(pos_count,1)}")
    return examples


def build_evaluator(val_path: str, max_samples: int = 1000) -> CECorrelationEvaluator:
    """
    CECorrelationEvaluator measures Pearson/Spearman correlation between
    predicted scores and ground truth labels.
    
    This tells us: does the model rank relevant chunks higher than irrelevant ones?
    """
    sentence_pairs = []
    scores = []
    
    with open(val_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            item = json.loads(line)
            sentence_pairs.append([item["query"], item["chunk"]])
            scores.append(float(item["label"]))
    
    evaluator = CECorrelationEvaluator(
        sentence_pairs=sentence_pairs,
        scores=scores,
        name="academic-reranker"
    )
    log.info(f"Built evaluator: {len(sentence_pairs)} pairs")
    return evaluator


# ─── Evaluation helpers ───────────────────────────────────────────────────────

def evaluate_precision_recall(model: CrossEncoder, val_path: str, max_samples: int = 500):
    """
    Compute precision and recall at various thresholds.
    More intuitive than correlation for retrieval tasks.
    """
    pairs  = []
    labels = []
    
    with open(val_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            item = json.loads(line)
            pairs.append([item["query"], item["chunk"]])
            labels.append(item["label"])
    
    scores = model.predict(pairs, show_progress_bar=False)
    
    # Compute metrics at threshold 0.5
    threshold = 0.5
    tp = sum(1 for s, l in zip(scores, labels) if s >= threshold and l == 1)
    fp = sum(1 for s, l in zip(scores, labels) if s >= threshold and l == 0)
    fn = sum(1 for s, l in zip(scores, labels) if s < threshold and l == 1)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    log.info(f"Precision@0.5: {precision:.4f} | Recall@0.5: {recall:.4f} | F1: {f1:.4f}")
    return {"precision": precision, "recall": recall, "f1": f1}


# ─── Training ─────────────────────────────────────────────────────────────────

def train():
    log.info("="*60)
    log.info("RERANKER FINE-TUNING")
    log.info("="*60)
    log.info(f"Base model: {CONFIG['base_model']}")
    log.info(f"Device: {'CUDA — ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # ── W&B init ─────────────────────────────────────────────────────────────
    wandb.init(
        project=CONFIG["wandb_project"],
        name=CONFIG["wandb_run"],
        config=CONFIG
    )

    # ── Load CrossEncoder ────────────────────────────────────────────────────
    log.info(f"\nLoading base model: {CONFIG['base_model']}")
    model = CrossEncoder(
        CONFIG["base_model"],
        num_labels=1,                      # single relevance score
        max_length=CONFIG["max_length"],
        default_activation_function=torch.nn.Sigmoid()  # output 0-1
    )

    # ── Load data ────────────────────────────────────────────────────────────
    train_examples = load_pairs(CONFIG["train_path"], CONFIG["max_train"])
    evaluator      = build_evaluator(CONFIG["val_path"])

    # ── Warmup ───────────────────────────────────────────────────────────────
    steps_per_epoch = len(train_examples) // CONFIG["batch_size"]
    total_steps     = steps_per_epoch * CONFIG["epochs"]
    warmup_steps    = int(total_steps * CONFIG["warmup_ratio"])
    
    log.info(f"\nTraining: {len(train_examples)} pairs | {CONFIG['epochs']} epochs")
    log.info(f"Steps: {total_steps} total | {warmup_steps} warmup")
    log.info(f"Batch size: {CONFIG['batch_size']} | LR: {CONFIG['lr']}")

    # ── Baseline evaluation ──────────────────────────────────────────────────
    log.info("\nBaseline metrics (before fine-tuning):")
    baseline_metrics = evaluate_precision_recall(model, CONFIG["val_path"])
    wandb.log({f"baseline/{k}": v for k, v in baseline_metrics.items()})

    # ── Train ────────────────────────────────────────────────────────────────
    os.makedirs(CONFIG["output_path"], exist_ok=True)
    
    model.fit(
        train_dataloader=DataLoader(
            train_examples,
            shuffle=True,
            batch_size=CONFIG["batch_size"]
        ),
        evaluator=evaluator,
        epochs=CONFIG["epochs"],
        warmup_steps=warmup_steps,
        optimizer_params={"lr": CONFIG["lr"]},
        output_path=CONFIG["output_path"],
        evaluation_steps=300,
        save_best_model=True,
        show_progress_bar=True,
        loss_fct=torch.nn.BCELoss()   # Binary Cross Entropy for 0/1 labels
    )

    # ── Post-training evaluation ─────────────────────────────────────────────
    log.info("\nPost-training metrics:")
    
    # Reload best saved model
    best_model = CrossEncoder(CONFIG["output_path"])
    final_metrics = evaluate_precision_recall(best_model, CONFIG["val_path"])
    wandb.log({f"final/{k}": v for k, v in final_metrics.items()})
    
    # Show improvement
    log.info("\n" + "="*60)
    log.info("IMPROVEMENT SUMMARY")
    log.info("="*60)
    for metric in ["precision", "recall", "f1"]:
        before = baseline_metrics[metric]
        after  = final_metrics[metric]
        delta  = after - before
        log.info(f"{metric.capitalize():12} | Before: {before:.4f} | After: {after:.4f} | Δ {delta:+.4f}")
    
    log.info(f"\nBest model saved → {CONFIG['output_path']}")
    log.info("Next step: run finetune/train_generator.py")
    
    wandb.finish()


if __name__ == "__main__":
    if not os.path.exists(CONFIG["train_path"]):
        log.error(f"Training data not found: {CONFIG['train_path']}")
        log.error("Run: python -m finetune.prep_data first")
        exit(1)
    train()
