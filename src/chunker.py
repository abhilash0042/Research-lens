from typing import List, Dict, Any
from transformers import AutoTokenizer
from src.utils import ParentChunk, ChildChunk, snap_to_sentence

# Initialize tokenizer (miniLM-L6-v2)
try:
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
except Exception:
    # Fallback to basic tokenizer if model not downloaded yet
    from transformers import BertTokenizerFast
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def _jaccard_similarity(text1: str, text2: str) -> float:
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

def parent_child_chunk(
    section_text: str, 
    section_meta: Dict[str, Any], 
    paper_id: str,
    parent_size: int = 1024, 
    child_size: int = 384, 
    overlap: int = 128,
    start_chunk_index: int = 0
) -> tuple[List[ParentChunk], int]:
    
    tokens = tokenizer.encode(section_text, add_special_tokens=False)
    if not tokens:
        return [], start_chunk_index
        
    chunks = []
    parent_id_counter = 0
    current_chunk_idx = start_chunk_index
    
    # Slide through text in parent-sized windows
    p_start = 0
    while p_start < len(tokens):
        p_end = min(p_start + parent_size, len(tokens))
        
        # Snap parent text to sentence
        raw_parent_text = tokenizer.decode(tokens[p_start:p_end])
        parent_text = snap_to_sentence(raw_parent_text, direction="end")
        
        # If snapped text is significantly smaller, adjust p_end
        snapped_tokens = tokenizer.encode(parent_text, add_special_tokens=False)
        actual_p_end = p_start + len(snapped_tokens) if len(snapped_tokens) > 0 else p_end
        
        parent_id = f"{paper_id}_{section_meta.get('section', 'unknown')}_{parent_id_counter}"
        
        # Create children from this parent
        children = []
        c_start = p_start
        while c_start < actual_p_end:
            c_end = min(c_start + child_size, actual_p_end)
            
            raw_child_text = tokenizer.decode(tokens[c_start:c_end])
            child_text = snap_to_sentence(raw_child_text, direction="end")
            
            snapped_child_tokens = tokenizer.encode(child_text, add_special_tokens=False)
            actual_c_end = c_start + len(snapped_child_tokens) if len(snapped_child_tokens) > 0 else c_end
            
            if child_text.strip():
                # Improvement 2: Enriched text with metadata
                header = f"Paper: {section_meta.get('title', '')} | Section: {section_meta.get('section', '')}\n"
                enriched_text = header + child_text
                
                children.append(ChildChunk(
                    text=child_text,
                    display_text=child_text,
                    enriched_text=enriched_text,
                    parent_id=parent_id,
                    metadata={
                        **section_meta,
                        "paper_id": paper_id,
                        "token_start": c_start,
                        "token_end": actual_c_end
                    },
                    chunk_index=current_chunk_idx
                ))
                current_chunk_idx += 1
            
            c_start += (child_size - overlap)
            if actual_c_end == c_end and actual_c_end >= actual_p_end:
                break # Avoid infinite loop if snapping fails
                
        if parent_text.strip() and children:
            chunks.append(ParentChunk(
                text=parent_text,
                parent_id=parent_id,
                children=children,
                metadata={
                    **section_meta,
                    "paper_id": paper_id,
                    "token_start": p_start,
                    "token_end": actual_p_end
                }
            ))
            
        parent_id_counter += 1
        p_start += (parent_size - overlap)
        if actual_p_end == p_end and actual_p_end >= len(tokens):
             break
    
    return chunks, current_chunk_idx

def deduplicate_children(children: List[ChildChunk], threshold: float = 0.8) -> List[ChildChunk]:
    """Remove near-duplicate child chunks."""
    unique_children = []
    
    for current in children:
        is_duplicate = False
        for existing in unique_children:
            if _jaccard_similarity(current.text, existing.text) > threshold:
                is_duplicate = True
                # Keep the one with more tokens
                if len(current.text) > len(existing.text):
                    existing.text = current.text
                    existing.enriched_text = current.enriched_text
                    existing.metadata = current.metadata
                break
                
        if not is_duplicate:
            unique_children.append(current)
            
    return unique_children
