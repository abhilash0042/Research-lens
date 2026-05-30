"""
intelligence.py
===============
Advanced multi-paper intelligence features powered by
semantic similarity (multi-head self-attention embeddings) + Groq reasoning.
"""

import json
import logging
import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from src.utils import ChildChunk, PaperResult
from src.models import get_embedder, get_groq_client

log = logging.getLogger(__name__)


# ─── Data structures ─────────────────────────────────────────────────────────

@dataclass
class Contradiction:
    """A detected contradiction between two paper claims."""
    claim_a: str
    claim_b: str
    paper_a: str           # paper title
    paper_b: str
    section_a: str
    section_b: str
    similarity: float      # how semantically similar the claims are (0-1)
    explanation: str       # LLM-generated explanation of the contradiction


@dataclass
class ComparisonRow:
    """One row in a cross-paper comparison table."""
    dimension: str         # e.g. "Dataset", "Method", "Sample Size"
    values: Dict[str, str] # paper_title -> value for that dimension


@dataclass
class PaperSummary:
    """Structured summary of a single paper."""
    title: str
    contribution: str
    methodology: str
    results: str
    datasets: str
    limitations: str


# ─── 1. CONTRADICTION DETECTION ──────────────────────────────────────────────

def _embed_chunks(chunks: List[ChildChunk]) -> np.ndarray:
    """
    Embed chunks using the multi-head self-attention embedder.
    Returns normalized 384-dim vectors for cosine similarity.
    """
    embedder = get_embedder()
    texts = [c.enriched_text for c in chunks]
    if not texts:
        return np.array([])
    embeddings = embedder.encode(texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)
    return embeddings


def _find_cross_paper_similar_pairs(
    chunks_a: List[ChildChunk],
    chunks_b: List[ChildChunk],
    threshold: float = 0.75,
    top_k: int = 20
) -> List[Tuple[ChildChunk, ChildChunk, float]]:
    """
    Find pairs of chunks from two different papers that discuss
    the same topic (high cosine similarity between attention vectors).
    """
    if not chunks_a or not chunks_b:
        return []

    emb_a = _embed_chunks(chunks_a)
    emb_b = _embed_chunks(chunks_b)

    dim = emb_b.shape[1]
    index_b = faiss.IndexFlatIP(dim)
    index_b.add(emb_b)

    k_search = min(3, len(chunks_b))
    scores, indices = index_b.search(emb_a, k_search)

    pairs = []
    seen = set()
    for i in range(len(chunks_a)):
        for j_rank in range(k_search):
            j = indices[i][j_rank]
            if j < 0:
                continue
            sim = float(scores[i][j_rank])
            if sim >= threshold and (i, j) not in seen:
                seen.add((i, j))
                pairs.append((chunks_a[i], chunks_b[j], sim))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_k]


def _classify_contradiction(claim_a: str, paper_a: str, claim_b: str, paper_b: str) -> Dict:
    """
    Use Groq (Llama-3) to determine if two similar claims actually contradict.
    """
    client = get_groq_client()

    system_prompt = """You are an expert scientific reviewer analyzing claims from academic papers.
Given two claims from different papers, determine their relationship.

Respond in EXACTLY this JSON format (no markdown, no code fences):
{"verdict": "contradiction" or "agreement" or "unrelated", "explanation": "brief 1-2 sentence explanation"}

Guidelines:
- "contradiction": The claims make opposing or incompatible statements about the same topic.
- "agreement": The claims support or reinforce each other.
- "unrelated": The claims discuss different aspects."""

    user_prompt = f"""CLAIM FROM "{paper_a}":\n{claim_a}\n\nCLAIM FROM "{paper_b}":\n{claim_b}"""

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.0,
            max_tokens=200
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        log.warning(f"Contradiction classification failed: {e}")
        return {"verdict": "unrelated", "explanation": "Failed to parse LLM response."}


def detect_contradictions(
    paper_results: List[PaperResult],
    similarity_threshold: float = 0.75,
    max_pairs_per_comparison: int = 10
) -> List[Contradiction]:
    """Detect contradictions across all pairs of papers."""
    contradictions = []

    for i in range(len(paper_results)):
        for j in range(i + 1, len(paper_results)):
            paper_a = paper_results[i]
            paper_b = paper_results[j]

            similar_pairs = _find_cross_paper_similar_pairs(
                paper_a.children, paper_b.children,
                threshold=similarity_threshold, top_k=max_pairs_per_comparison
            )

            for chunk_a, chunk_b, sim in similar_pairs:
                result = _classify_contradiction(
                    chunk_a.text, paper_a.metadata.title,
                    chunk_b.text, paper_b.metadata.title
                )

                if result.get("verdict") == "contradiction":
                    contradictions.append(Contradiction(
                        claim_a=chunk_a.text,
                        claim_b=chunk_b.text,
                        paper_a=paper_a.metadata.title,
                        paper_b=paper_b.metadata.title,
                        section_a=chunk_a.metadata.get("section", "Unknown"),
                        section_b=chunk_b.metadata.get("section", "Unknown"),
                        similarity=sim,
                        explanation=result.get("explanation", "")
                    ))

    return contradictions


# ─── 2. CROSS-PAPER COMPARISON ───────────────────────────────────────────────

def generate_comparison_table(paper_results: List[PaperResult]) -> List[ComparisonRow]:
    """Generate a structured comparison across all loaded papers."""
    if not paper_results:
        return []
        
    client = get_groq_client()

    papers_block = ""
    for pr in paper_results:
        title = pr.metadata.title
        text_sample = "\n".join([c.text for c in pr.children[:15]])[:3000]
        papers_block += f"\n--- PAPER: {title} ---\n{text_sample}\n"

    system_prompt = """You are a research analyst creating a structured comparison table across academic papers.

Extract information for EACH paper along these dimensions. Respond in EXACTLY this JSON format (no markdown, no code fences):
[
  {"dimension": "Research Objective", "values": {"Paper Title 1": "...", "Paper Title 2": "..."}},
  {"dimension": "Dataset", "values": {"Paper Title 1": "...", "Paper Title 2": "..."}},
  {"dimension": "Methodology", "values": {"Paper Title 1": "...", "Paper Title 2": "..."}},
  {"dimension": "Key Results", "values": {"Paper Title 1": "...", "Paper Title 2": "..."}},
  {"dimension": "Limitations", "values": {"Paper Title 1": "...", "Paper Title 2": "..."}}
]"""

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Compare these papers:\n{papers_block}"}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=1500
        )
        rows_data = json.loads(response.choices[0].message.content)
        return [ComparisonRow(dimension=r["dimension"], values=r["values"]) for r in rows_data]
    except Exception as e:
        log.warning(f"Comparison table generation failed: {e}")
        return []


# ─── 3. LITERATURE REVIEW GENERATION ─────────────────────────────────────────

def generate_literature_review(
    paper_results: List[PaperResult],
    focus_topic: Optional[str] = None
) -> str:
    """Generate a coherent, multi-cited literature review paragraph."""
    if not paper_results:
        return "No papers provided for literature review."
        
    client = get_groq_client()

    papers_block = ""
    for pr in paper_results:
        title = pr.metadata.title
        authors = pr.metadata.authors
        year = pr.metadata.year
        key_text = "\n".join([c.text for c in pr.children[:12]])[:2500]
        papers_block += f"\n--- [{title}] by {authors} ({year}) ---\n{key_text}\n"

    focus_instruction = f"\nFocus specifically on: {focus_topic}" if focus_topic else ""

    system_prompt = f"""You are an academic researcher writing a literature review section for a paper.

INSTRUCTIONS:
1. Write a cohesive, well-structured literature review paragraph (200-350 words).
2. Synthesize findings across ALL provided papers — do not summarize them one by one.
3. Every factual claim MUST cite the source paper using the format (Author, Year).
4. Highlight agreements, differences, and gaps in the literature.{focus_instruction}"""

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Write a literature review based on these papers:\n{papers_block}"}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.3,
            max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e:
        log.warning(f"Literature review generation failed: {e}")
        return "Failed to generate literature review."


# ─── 4. PAPER SUMMARIZATION ──────────────────────────────────────────────────

def summarize_paper(paper_result: PaperResult) -> PaperSummary:
    """Generate a detailed, structured summary of a single paper using Groq."""
    # Expand to use a much larger portion of the paper (up to ~24k chars / ~6k tokens)
    full_text = "\n".join([c.text for c in paper_result.children[:50]])[:24000]

    client = get_groq_client()
    system_prompt = """You are an expert at summarizing academic papers.
Given the text of a paper, generate a highly detailed and comprehensive structured summary. 
Do not be brief. Provide enough technical depth that another researcher can fully understand the paper without reading it.

Respond in EXACTLY this JSON format (no markdown, no code fences):
{
  "contribution": "1-2 detailed paragraphs explaining the core problem and the main contribution.",
  "methodology": "1-2 detailed paragraphs explaining the specific approach, algorithms, or experiments.",
  "results": "1-2 detailed paragraphs explaining the exact findings, metrics, and outcomes.",
  "datasets": "Detailed list of datasets used (or 'Not specified')",
  "limitations": "Detailed explanation of limitations and future work (or 'Not explicitly stated')"
}"""

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Paper title: {paper_result.metadata.title}\n\nPaper Content:\n{full_text}"}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.2,
            max_tokens=2000  # Increased heavily to allow for long summaries
        )
        content = response.choices[0].message.content
        
        # Robust JSON parsing (strip markdown fences if the model hallucinates them)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].strip()
            
        data = json.loads(content)
        return PaperSummary(
            title=paper_result.metadata.title,
            contribution=data.get("contribution", "Not available"),
            methodology=data.get("methodology", "Not available"),
            results=data.get("results", "Not available"),
            datasets=data.get("datasets", "Not specified"),
            limitations=data.get("limitations", "Not explicitly stated")
        )
    except Exception as e:
        log.warning(f"Summarization failed: {e}")
        return PaperSummary(
            title=paper_result.metadata.title,
            contribution="Failed to generate structured summary.",
            methodology="N/A", results="N/A", datasets="N/A", limitations="N/A"
        )


# ─── 5. KEY FINDINGS EXTRACTION ──────────────────────────────────────────────

def extract_key_findings(paper_results: List[PaperResult]) -> Dict[str, List[str]]:
    """Extract the top 3-5 key findings from each paper."""
    client = get_groq_client()
    findings = {}

    for pr in paper_results:
        title = pr.metadata.title
        relevant_chunks = [
            c for c in pr.children
            if any(kw in c.metadata.get("section", "").lower()
                   for kw in ["result", "conclusion", "discussion", "finding"])
        ]
        if not relevant_chunks:
            relevant_chunks = pr.children[:10]

        text = "\n".join([c.text for c in relevant_chunks[:10]])[:2500]

        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Extract top 3-5 key findings. Return as JSON list of strings. No markdown formatting."},
                    {"role": "user", "content": f"Paper: {title}\n\n{text}"}
                ],
                model="llama-3.1-8b-instant",
                temperature=0.1,
                max_tokens=400
            )
            findings[title] = json.loads(response.choices[0].message.content)
        except Exception:
            findings[title] = ["Could not extract findings."]

    return findings


# ─── 6. AUTO-HYPOTHESIS GENERATION ───────────────────────────────────────────

def generate_hypotheses(paper_results: List[PaperResult]) -> str:
    """Generate 3 novel research hypotheses based on the gaps/limitations of the papers."""
    if not paper_results:
        return "Please upload papers first."
        
    client = get_groq_client()
    
    # Extract limitations or discussion sections
    context = ""
    for pr in paper_results:
        title = pr.metadata.title
        limitations = [
            c.text for c in pr.children
            if any(kw in c.metadata.get("section", "").lower() for kw in ["limit", "future", "discussion", "conclusion"])
        ][:5]
        
        if not limitations:
            limitations = [c.text for c in pr.children[-5:]] # fallback to end of paper
            
        context += f"\n--- Paper: {title} ---\n"
        context += "\n".join(limitations) + "\n"

    system_prompt = """You are a brilliant AI Research Scientist.
Analyze the limitations, future work, and conclusions of the provided papers.
Synthesize this information to propose 3 NOVEL, highly specific research hypotheses or experiments that have NOT been done yet, but logically follow from the gaps in these papers.

Format your response as a numbered list with bold titles. For each hypothesis, explain:
1. The Core Idea
2. Why it is novel (based on the provided papers)
3. A brief experimental design to test it."""

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Based on the following excerpts, propose 3 novel research hypotheses:\n\n{context[:15000]}"}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.4,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        log.warning(f"Hypothesis generation failed: {e}")
        return "Failed to generate hypotheses."
