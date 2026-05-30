# src module
from .utils import (
    ValidationResult, PageData, PaperMetadata, 
    ChildChunk, ParentChunk, PaperResult, UnifiedIndex,
    snap_to_sentence, generate_paper_id
)
from .pdf_validator import validate_pdf
from .text_extractor import extract_text, extract_metadata, detect_sections
from .chunker import parent_child_chunk, deduplicate_children
from .indexer import build_faiss_index, build_bm25_index, build_unified_index, save_index, reciprocal_rank_fusion, load_index
from .models import get_embedder, get_reranker, generate_cited_answer
from .pipeline import hybrid_search, rerank_chunks, ask_question
from .intelligence import (
    detect_contradictions, generate_comparison_table, 
    generate_literature_review, summarize_paper, extract_key_findings
)
from .ingestion import ingest_paper, ingest_papers
from .evaluation import evaluate_retrieval, evaluate_generation, run_full_evaluation
