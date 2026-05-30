"""
models.py
=========
Central model manager for ResearchLens.
Handles loading of all local models and the Groq API client.

Local models:
  - Embedder (MiniLM-L6-v2) — sentence embeddings for FAISS search
  - Reranker (ms-marco CrossEncoder) — relevance scoring
  - Summarizer (BART-large-cnn) — extractive summarization

Cloud API:
  - Groq (Llama-3-8B) — cited answer generation
"""

import os
import logging
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Local Models
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from transformers import pipeline as hf_pipeline

# Remote Generator
from groq import Groq

load_dotenv()
log = logging.getLogger(__name__)

# Global instances to prevent reloading
_embedder = None
_reranker = None
_summarizer = None
_groq_client = None


def get_embedder(model_path: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Loads the embedder model (single shared instance).
    Checks if a fine-tuned version exists in models/embedder, otherwise uses base.
    """
    global _embedder
    if _embedder is None:
        ft_path = "models/embedder"
        path_to_load = ft_path if os.path.exists(ft_path) else model_path
        log.info(f"Loading embedder from: {path_to_load}")
        _embedder = SentenceTransformer(path_to_load)
    return _embedder


def get_reranker(model_path: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> CrossEncoder:
    """
    Loads the cross-encoder reranker.
    Checks if a fine-tuned version exists in models/reranker, otherwise uses base.
    """
    global _reranker
    if _reranker is None:
        ft_path = "models/reranker"
        path_to_load = ft_path if os.path.exists(ft_path) else model_path
        log.info(f"Loading reranker from: {path_to_load}")
        _reranker = CrossEncoder(path_to_load, max_length=512)
    return _reranker




# ─── Groq Generator ─────────────────────────────────────────────────────────

def get_groq_client() -> Groq:
    """Returns the Groq client, initialized with the API key from .env"""
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found in environment variables.\n"
                "Create a .env file with: GROQ_API_KEY=your_key_here\n"
                "Get a free key at: https://console.groq.com"
            )
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def generate_cited_answer(question: str, context: str, model: str = "llama-3.1-8b-instant", chat_history: List[Dict[str, str]] = None) -> str:
    """
    Uses Groq (Llama-3) to generate an answer based purely on the retrieved context.
    Uses Few-Shot Prompting to enforce exact citation formatting.
    
    Includes error handling for network failures and rate limits.
    """
    client = get_groq_client()
    
    system_prompt = """You are ResearchLens, an expert academic research assistant.
Your task is to answer the user's question ONLY using the provided SOURCE chunks.

CRITICAL INSTRUCTIONS:
1. Do not use outside knowledge. If the answer is not in the sources, say: "Not found in the provided papers."
2. Every factual claim MUST include a citation using the exact format: [SOURCE N: paper_title, section].
3. Be precise, specific, and concise.

EXAMPLES:
Question: How many patients were in the study?
Sources:
[SOURCE 1: Clinical Trial V1, Methods] We enrolled 542 patients across 3 sites.

Answer: The study enrolled a total of 542 patients [SOURCE 1: Clinical Trial V1, Methods]."""

    user_prompt = f"""SOURCES:
{context}

QUESTION: {question}"""

    messages = [{"role": "system", "content": system_prompt}]
    if chat_history:
        for msg in chat_history[-6:]:  # Only include last 3 turns (6 messages) to save context window
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_prompt})

    try:
        response = client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=0.1,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        log.error(f"Groq API error: {e}")
        return f"Error generating answer: {str(e)}. Please check your internet connection and API key."
