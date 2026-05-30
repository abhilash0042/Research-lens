"""
pipeline.py
===========
Core RAG Pipeline: Hybrid Search (FAISS + BM25) → RRF → CrossEncoder Reranking → Groq Generation.
"""

import faiss
import numpy as np
import nltk
import logging
from typing import List, Dict, Any, Tuple
from src.utils import ChildChunk, UnifiedIndex
from src.models import get_embedder, get_reranker, generate_cited_answer
from src.indexer import reciprocal_rank_fusion

log = logging.getLogger(__name__)


def hybrid_search(query: str, unified_index: UnifiedIndex, top_k: int = 20) -> List[ChildChunk]:
    """
    Performs hybrid search (Dense FAISS + Sparse BM25) and merges with RRF.
    """
    if not unified_index.children:
        return []

    # 1. FAISS Dense Search
    embedder = get_embedder()
    query_vector = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vector)
    
    k_search = min(top_k, len(unified_index.children))
    scores, faiss_indices = unified_index.faiss_index.search(query_vector, k_search)
    # Filter out -1 indices (FAISS returns -1 when fewer results available)
    faiss_ranks = [idx for idx in faiss_indices[0].tolist() if idx >= 0]

    # 2. BM25 Sparse Search
    tokenized_query = nltk.word_tokenize(query.lower())
    bm25_scores = unified_index.bm25_index.get_scores(tokenized_query)
    bm25_ranks = np.argsort(bm25_scores)[::-1][:k_search].tolist()

    # 3. Reciprocal Rank Fusion (RRF)
    fused_indices = reciprocal_rank_fusion([faiss_ranks, bm25_ranks])
    
    # Safely retrieve chunks (guard against out-of-bounds)
    results = []
    for idx in fused_indices[:top_k]:
        if 0 <= idx < len(unified_index.children):
            results.append(unified_index.children[idx])
    return results


def rerank_chunks(query: str, chunks: List[ChildChunk], top_n: int = 5) -> List[ChildChunk]:
    """
    Reranks the candidate chunks using the CrossEncoder.
    """
    if not chunks:
        return []
        
    reranker = get_reranker()
    
    pairs = [[query, chunk.enriched_text] for chunk in chunks]
    
    scores = reranker.predict(pairs)
    
    scored_chunks = list(zip(chunks, scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    
    return [chunk for chunk, score in scored_chunks[:top_n]]


def format_context(chunks: List[ChildChunk]) -> str:
    """
    Formats the top chunks into the context string for the Generator.
    """
    context = ""
    for i, chunk in enumerate(chunks):
        title = chunk.metadata.get("title", "Unknown Paper")
        section = chunk.metadata.get("section", "Unknown Section")
        context += f"[SOURCE {i+1}: {title}, {section}]\n{chunk.text}\n\n"
    return context.strip()


def ask_question(query: str, unified_indices: List[UnifiedIndex], chat_history: List[Dict[str, str]] = None) -> str:
    """
    Full RAG Pipeline with Conversational Memory:
    1. Hybrid Search across all indices
    2. Rerank top results
    3. Generate cited answer via Groq, passing prior chat context
    """
    if not unified_indices:
        return "No papers have been loaded. Please upload PDFs first."
    
    # 1. Search across all loaded papers
    all_candidates = []
    for index in unified_indices:
        candidates = hybrid_search(query, index, top_k=10)
        all_candidates.extend(candidates)
        
    if not all_candidates:
        return "No relevant information found in the provided papers."

    # 2. Rerank to find the absolute best across all papers
    best_chunks = rerank_chunks(query, all_candidates, top_n=5)
    
    if not best_chunks:
        return "No relevant information found after reranking."
    
    # 3. Format the context block
    context_str = format_context(best_chunks)
    
    # 4. Ask Groq (Llama-3.1)
    log.info(f"Sending {len(best_chunks)} chunks to Groq for answer generation")
    
    # Inject chat history into the generation logic
    # We will modify generate_cited_answer or we can just prepend history to the user prompt.
    # The cleaner way is to let generate_cited_answer accept chat_history.
    
    answer = generate_cited_answer(query, context_str, chat_history=chat_history)
    
    return answer
