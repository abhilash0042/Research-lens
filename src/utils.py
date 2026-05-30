import re
import os
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    is_scanned: bool = False

@dataclass
class PageData:
    page_num: int
    text: str
    width: float
    height: float
    section: str = "Abstract"
    font_sizes: Dict[str, float] = field(default_factory=dict)

@dataclass
class PaperMetadata:
    title: str = "Unknown Title"
    authors: str = "Unknown Authors"
    year: str = "Unknown Year"
    doi: str = "Unknown DOI"
    n_pages: int = 0
    filepath: str = ""

@dataclass
class ChildChunk:
    text: str
    display_text: str
    enriched_text: str
    parent_id: str
    metadata: Dict[str, Any]
    chunk_index: int = 0

@dataclass
class ParentChunk:
    text: str
    parent_id: str
    children: List[ChildChunk]
    metadata: Dict[str, Any]

@dataclass
class PaperResult:
    metadata: PaperMetadata
    parent_store: Dict[str, ParentChunk]
    children: List[ChildChunk]
    faiss_index: Any  # faiss.Index
    bm25_index: Any   # BM25Okapi
    paper_id: str

@dataclass
class UnifiedIndex:
    faiss_index: Any
    bm25_index: Any
    children: List[ChildChunk]

def snap_to_sentence(text: str, direction: str = "end") -> str:
    """Snap to nearest sentence boundary."""
    if direction == "end":
        # Find last sentence-ending punctuation followed by space
        match = list(re.finditer(r'[.!?]\s+', text))
        if match:
            snapped = text[:match[-1].end()]
            if len(snapped.strip()) > 20:
                return snapped
    elif direction == "start":
        # Find first sentence-ending punctuation followed by space
        match = re.search(r'[.!?]\s+', text)
        if match:
            snapped = text[match.end():]
            if len(snapped.strip()) > 20:
                return snapped
    return text

def generate_paper_id(filepath: str) -> str:
    """Generate a deterministic ID from the filename."""
    basename = os.path.basename(filepath)
    hash_obj = hashlib.md5(basename.encode("utf-8"))
    return f"{basename.replace('.pdf', '')}_{hash_obj.hexdigest()[:6]}"

def ensure_data_dirs(base_dir: str = "data/indices") -> None:
    """Ensure that the data directory exists."""
    os.makedirs(base_dir, exist_ok=True)
