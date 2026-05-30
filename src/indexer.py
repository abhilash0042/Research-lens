"""
indexer.py
==========
Builds and manages FAISS (dense vector) and BM25 (sparse keyword) indices
for the ingested paper chunks.
"""

import os
import faiss
import pickle
import json
import numpy as np
import nltk
from typing import List, Dict, Tuple, Any
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from src.utils import ChildChunk, ParentChunk, PaperMetadata, UnifiedIndex, ensure_data_dirs

# Ensure nltk tokenizer data is available
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')


def build_faiss_index(
    children: List[ChildChunk], 
    model_path: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> Tuple[faiss.Index, List[ChildChunk]]:
    """Build a FAISS inner-product index from child chunk embeddings."""
    if not children:
        return faiss.IndexFlatIP(384), []
    
    # Import from models.py to use the single shared embedder instance
    from src.models import get_embedder
    embedder = get_embedder(model_path)
    
    texts = [c.enriched_text for c in children]
    
    embeddings = embedder.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True
    )
    
    faiss.normalize_L2(embeddings)
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    return index, children

def build_bm25_index(children: List[ChildChunk]) -> BM25Okapi:
    """Build a BM25 keyword index from child chunk text."""
    if not children:
        return BM25Okapi([[""]])
        
    tokenized = [
        nltk.word_tokenize(c.text.lower())
        for c in children
    ]
    return BM25Okapi(tokenized)

def save_index(
    paper_id: str, 
    index: faiss.Index, 
    children: List[ChildChunk], 
    parents: Dict[str, ParentChunk], 
    bm25: BM25Okapi, 
    metadata: PaperMetadata
) -> None:
    """Persist all index data and metadata to disk."""
    ensure_data_dirs()
    base = f"data/indices/{paper_id}"
    os.makedirs(base, exist_ok=True)
    
    faiss.write_index(index, f"{base}/faiss.index")
    
    with open(f"{base}/children.pkl", "wb") as f:
        pickle.dump(children, f)
        
    with open(f"{base}/parents.pkl", "wb") as f:
        pickle.dump(parents, f)
        
    with open(f"{base}/bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
        
    with open(f"{base}/metadata.json", "w") as f:
        json.dump({
            "title": metadata.title,
            "authors": metadata.authors,
            "year": metadata.year,
            "doi": metadata.doi,
            "n_pages": metadata.n_pages,
            "filepath": metadata.filepath
        }, f, indent=2)

def load_index(paper_id: str) -> Tuple[UnifiedIndex, Dict[str, ParentChunk], PaperMetadata]:
    """Load saved indices and metadata for a paper from disk."""
    base = f"data/indices/{paper_id}"
    
    if not os.path.exists(base):
        raise FileNotFoundError(f"Index not found for paper: {paper_id}")
    
    faiss_index = faiss.read_index(f"{base}/faiss.index")
    
    with open(f"{base}/children.pkl", "rb") as f:
        children = pickle.load(f)
        
    with open(f"{base}/parents.pkl", "rb") as f:
        parents = pickle.load(f)
        
    with open(f"{base}/bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)
        
    with open(f"{base}/metadata.json", "r") as f:
        meta_dict = json.load(f)
        metadata = PaperMetadata(**meta_dict)
        
    unified = UnifiedIndex(
        faiss_index=faiss_index,
        bm25_index=bm25,
        children=children
    )
    
    return unified, parents, metadata

def build_unified_index(all_children: List[ChildChunk], model_path: str = "sentence-transformers/all-MiniLM-L6-v2") -> UnifiedIndex:
    """Build both FAISS and BM25 indices and return a unified index."""
    faiss_index, children_ordered = build_faiss_index(all_children, model_path)
    bm25_index = build_bm25_index(children_ordered)
    return UnifiedIndex(
        faiss_index=faiss_index,
        bm25_index=bm25_index,
        children=children_ordered
    )

def reciprocal_rank_fusion(rankings_list: List[List[int]], k: int = 60) -> List[int]:
    """Fuse multiple ranked lists of indices using Reciprocal Rank Fusion."""
    scores = {}
    for rankings in rankings_list:
        for rank, doc_id in enumerate(rankings):
            if doc_id < 0:
                continue  # Skip invalid FAISS indices (-1)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            
    reranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, score in reranked]
