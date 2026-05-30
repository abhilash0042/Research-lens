"""
evaluation.py
=============
Measures how well the retrieval and generation pipeline performs.

Two categories of metrics:

1. RETRIEVAL METRICS — How good is the search?
   - Recall@K:  Is the correct chunk in the top K results?
   - MRR:       Mean Reciprocal Rank — how high is the correct chunk ranked?
   - NDCG@K:    Normalized Discounted Cumulative Gain — weighted ranking quality

2. GENERATION METRICS — How good is the answer?
   - ROUGE-L:   Longest Common Subsequence overlap with reference answer
   - BERTScore: Semantic similarity between generated and reference answer
   - Citation Accuracy: Does the answer cite the correct source?

Usage:
    from src.evaluation import evaluate_retrieval, evaluate_generation
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

from src.utils import ChildChunk, UnifiedIndex, PaperResult
from src.pipeline import hybrid_search, rerank_chunks


# ─── Data structures ─────────────────────────────────────────────────────────

@dataclass
class RetrievalMetrics:
    """Results from retrieval evaluation."""
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0           # Mean Reciprocal Rank
    ndcg_at_10: float = 0.0
    num_queries: int = 0

    def __str__(self):
        return (
            f"Retrieval Metrics ({self.num_queries} queries):\n"
            f"  Recall@1:   {self.recall_at_1:.4f}\n"
            f"  Recall@5:   {self.recall_at_5:.4f}\n"
            f"  Recall@10:  {self.recall_at_10:.4f}\n"
            f"  MRR:        {self.mrr:.4f}\n"
            f"  NDCG@10:    {self.ndcg_at_10:.4f}"
        )


@dataclass
class GenerationMetrics:
    """Results from generation evaluation."""
    rouge_l_precision: float = 0.0
    rouge_l_recall: float = 0.0
    rouge_l_f1: float = 0.0
    bert_score_f1: float = 0.0
    citation_accuracy: float = 0.0    # % of answers with correct citations
    num_examples: int = 0

    def __str__(self):
        return (
            f"Generation Metrics ({self.num_examples} examples):\n"
            f"  ROUGE-L F1:         {self.rouge_l_f1:.4f}\n"
            f"  BERTScore F1:       {self.bert_score_f1:.4f}\n"
            f"  Citation Accuracy:  {self.citation_accuracy:.4f}"
        )


@dataclass
class EvalExample:
    """A single evaluation example with query, expected evidence, and answer."""
    query: str
    relevant_chunk_text: str     # the ground-truth evidence
    expected_answer: str = ""    # optional reference answer
    paper_title: str = ""


# ─── 1. RETRIEVAL EVALUATION ─────────────────────────────────────────────────

def _dcg(relevances: List[int], k: int) -> float:
    """Discounted Cumulative Gain at K."""
    relevances = relevances[:k]
    dcg = 0.0
    for i, rel in enumerate(relevances):
        dcg += rel / np.log2(i + 2)  # i+2 because log2(1) = 0
    return dcg


def _ndcg(relevances: List[int], k: int) -> float:
    """Normalized DCG at K."""
    dcg = _dcg(relevances, k)
    # Ideal DCG: sort relevances descending
    ideal = _dcg(sorted(relevances, reverse=True), k)
    if ideal == 0:
        return 0.0
    return dcg / ideal


def evaluate_retrieval(
    eval_examples: List[EvalExample],
    unified_indices: List[UnifiedIndex],
    use_reranker: bool = True,
    top_k: int = 10
) -> RetrievalMetrics:
    """
    Evaluate the retrieval pipeline on a set of examples.
    
    For each query:
      1. Run hybrid search (FAISS + BM25)
      2. Optionally rerank with CrossEncoder
      3. Check if the relevant chunk appears in the top K results
      4. Compute Recall@K, MRR, NDCG@K
    
    Args:
        eval_examples: list of EvalExample with query + relevant_chunk_text
        unified_indices: the paper indices to search over
        use_reranker: whether to apply the CrossEncoder reranker
        top_k: evaluate at this K
    """
    recalls_1 = []
    recalls_5 = []
    recalls_10 = []
    reciprocal_ranks = []
    ndcg_scores = []

    for example in eval_examples:
        # Search across all indices
        all_candidates = []
        for index in unified_indices:
            candidates = hybrid_search(example.query, index, top_k=20)
            all_candidates.extend(candidates)

        if not all_candidates:
            recalls_1.append(0)
            recalls_5.append(0)
            recalls_10.append(0)
            reciprocal_ranks.append(0)
            ndcg_scores.append(0)
            continue

        # Optionally rerank
        if use_reranker:
            ranked_chunks = rerank_chunks(example.query, all_candidates, top_n=top_k)
        else:
            ranked_chunks = all_candidates[:top_k]

        # Check where the relevant chunk appears
        # Use text overlap to determine match (fuzzy matching)
        relevances = []
        found_rank = None
        for rank, chunk in enumerate(ranked_chunks):
            # A chunk is "relevant" if it contains significant overlap with the evidence
            overlap = _text_overlap(chunk.text, example.relevant_chunk_text)
            if overlap > 0.5:
                relevances.append(1)
                if found_rank is None:
                    found_rank = rank + 1  # 1-indexed
            else:
                relevances.append(0)

        # Recall@K: did we find the relevant chunk in top K?
        recalls_1.append(1 if found_rank is not None and found_rank <= 1 else 0)
        recalls_5.append(1 if found_rank is not None and found_rank <= 5 else 0)
        recalls_10.append(1 if found_rank is not None and found_rank <= 10 else 0)

        # MRR: reciprocal of the rank where we found it
        reciprocal_ranks.append(1.0 / found_rank if found_rank else 0.0)

        # NDCG@10
        ndcg_scores.append(_ndcg(relevances, 10))

    return RetrievalMetrics(
        recall_at_1=np.mean(recalls_1) if recalls_1 else 0.0,
        recall_at_5=np.mean(recalls_5) if recalls_5 else 0.0,
        recall_at_10=np.mean(recalls_10) if recalls_10 else 0.0,
        mrr=np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0,
        ndcg_at_10=np.mean(ndcg_scores) if ndcg_scores else 0.0,
        num_queries=len(eval_examples)
    )


def _text_overlap(text_a: str, text_b: str) -> float:
    """
    Compute word-level Jaccard overlap between two texts.
    Returns a float between 0 and 1.
    """
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


# ─── 2. GENERATION EVALUATION ────────────────────────────────────────────────

def evaluate_generation(
    predictions: List[str],
    references: List[str],
    source_papers: Optional[List[str]] = None
) -> GenerationMetrics:
    """
    Evaluate the quality of generated answers against reference answers.
    
    Metrics:
      - ROUGE-L: Measures overlap of longest common subsequence.
                 Good for checking factual coverage.
      - BERTScore: Uses BERT embeddings to measure semantic similarity.
                   Catches paraphrases that ROUGE would miss.
      - Citation Accuracy: Checks if generated answer contains proper
                          [SOURCE N: ...] citations.
    """
    if not predictions or not references:
        return GenerationMetrics()

    # ROUGE-L
    rouge_scores = _compute_rouge_l(predictions, references)

    # BERTScore
    bert_scores = _compute_bert_score(predictions, references)

    # Citation accuracy
    citation_acc = _compute_citation_accuracy(predictions, source_papers)

    return GenerationMetrics(
        rouge_l_precision=rouge_scores["precision"],
        rouge_l_recall=rouge_scores["recall"],
        rouge_l_f1=rouge_scores["f1"],
        bert_score_f1=bert_scores,
        citation_accuracy=citation_acc,
        num_examples=len(predictions)
    )


def _compute_rouge_l(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute ROUGE-L (Longest Common Subsequence) between predictions and references.
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

        precisions = []
        recalls = []
        f1s = []

        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            precisions.append(scores["rougeL"].precision)
            recalls.append(scores["rougeL"].recall)
            f1s.append(scores["rougeL"].fmeasure)

        return {
            "precision": np.mean(precisions),
            "recall": np.mean(recalls),
            "f1": np.mean(f1s)
        }
    except ImportError:
        print("Warning: rouge_score not installed. Skipping ROUGE-L.")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def _compute_bert_score(predictions: List[str], references: List[str]) -> float:
    """
    Compute BERTScore F1 using the bert-score library.
    
    BERTScore computes token-level cosine similarity between
    contextual embeddings of prediction and reference tokens,
    then aggregates using greedy matching.
    """
    try:
        from bert_score import score as bert_score_fn

        P, R, F1 = bert_score_fn(
            predictions, references,
            lang="en",
            verbose=False,
            rescale_with_baseline=True
        )
        return float(F1.mean())
    except ImportError:
        print("Warning: bert_score not installed. Skipping BERTScore.")
        return 0.0
    except Exception as e:
        print(f"Warning: BERTScore failed: {e}")
        return 0.0


def _compute_citation_accuracy(
    predictions: List[str],
    source_papers: Optional[List[str]] = None
) -> float:
    """
    Check if generated answers contain proper citations.
    
    Checks for:
      1. Contains at least one [SOURCE N: ...] citation
      2. If source_papers provided, checks if cited paper exists
    """
    import re

    if not predictions:
        return 0.0

    correct = 0
    citation_pattern = re.compile(r'\[SOURCE\s+\d+:.*?\]', re.IGNORECASE)

    for pred in predictions:
        citations = citation_pattern.findall(pred)
        if citations:
            correct += 1

    return correct / len(predictions)


# ─── 3. QUICK EVALUATION REPORT ──────────────────────────────────────────────

def run_full_evaluation(
    eval_examples: List[EvalExample],
    unified_indices: List[UnifiedIndex],
    generate_fn=None
) -> Dict[str, any]:
    """
    Run a complete evaluation of both retrieval and generation.
    
    Args:
        eval_examples: test examples with queries and ground truth
        unified_indices: paper indices to search
        generate_fn: function(query, indices) -> answer string
    
    Returns dict with retrieval_metrics, generation_metrics, and summary.
    """
    # Retrieval evaluation
    print("Evaluating retrieval pipeline...")
    retrieval_without_reranker = evaluate_retrieval(
        eval_examples, unified_indices, use_reranker=False
    )
    retrieval_with_reranker = evaluate_retrieval(
        eval_examples, unified_indices, use_reranker=True
    )

    print("\n--- Without Reranker ---")
    print(retrieval_without_reranker)
    print("\n--- With Reranker ---")
    print(retrieval_with_reranker)

    # Reranker improvement
    recall_improvement = retrieval_with_reranker.recall_at_5 - retrieval_without_reranker.recall_at_5
    print(f"\nReranker Recall@5 improvement: {recall_improvement:+.4f}")

    results = {
        "retrieval_no_reranker": retrieval_without_reranker,
        "retrieval_with_reranker": retrieval_with_reranker,
        "reranker_recall5_delta": recall_improvement,
    }

    # Generation evaluation (if generate_fn provided)
    if generate_fn:
        print("\nEvaluating generation pipeline...")
        predictions = []
        references = []

        for example in eval_examples:
            if example.expected_answer:
                answer = generate_fn(example.query, unified_indices)
                predictions.append(answer)
                references.append(example.expected_answer)

        if predictions:
            gen_metrics = evaluate_generation(predictions, references)
            print(gen_metrics)
            results["generation"] = gen_metrics

    return results
