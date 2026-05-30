from typing import List, Callable, Optional
from src.utils import PaperResult, generate_paper_id
from src.pdf_validator import validate_pdf
from src.text_extractor import extract_text, extract_metadata, detect_sections
from src.chunker import parent_child_chunk, deduplicate_children
from src.indexer import build_faiss_index, build_bm25_index, save_index

def ingest_paper(filepath: str, progress_callback: Optional[Callable[[float, str], None]] = None) -> Optional[PaperResult]:
    """
    Ingest a single PDF file through the full pipeline.
    """
    try:
        paper_id = generate_paper_id(filepath)
        
        # Step 1: Validate
        if progress_callback:
            progress_callback(0.1, f"Validating PDF: {filepath}")
            
        val_result = validate_pdf(filepath)
        if not val_result.is_valid:
            error_msg = "; ".join(val_result.errors)
            if progress_callback:
                progress_callback(0.1, f"Validation failed: {error_msg}")
            print(f"Skipping {filepath}: {error_msg}")
            return None
            
        # Step 2: Extract text
        if progress_callback:
            progress_callback(0.3, "Extracting text and structure...")
            
        pages = extract_text(filepath)
        
        # Step 3: Metadata
        metadata = extract_metadata(filepath, pages)
        
        # Step 4: Section detection
        if progress_callback:
            progress_callback(0.4, "Detecting sections...")
            
        pages = detect_sections(pages, filepath)
        
        # Step 5: Chunking
        if progress_callback:
            progress_callback(0.5, "Chunking document...")
            
        all_chunks = []
        current_chunk_index = 0
        # Group text by section
        current_section = None
        section_text = ""
        section_start_page = 1
        
        for page in pages:
            if current_section != page.section:
                # Process previous section
                if current_section and section_text.strip():
                    section_meta = {
                        "title": metadata.title,
                        "section": current_section,
                        "page_start": section_start_page,
                        "page_end": page.page_num - 1
                    }
                    section_chunks, current_chunk_index = parent_child_chunk(
                        section_text, section_meta, paper_id, start_chunk_index=current_chunk_index
                    )
                    all_chunks.extend(section_chunks)
                    
                # Start new section
                current_section = page.section
                section_text = page.text + "\n"
                section_start_page = page.page_num
            else:
                section_text += page.text + "\n"
                
        # Process final section
        if current_section and section_text.strip():
            section_meta = {
                "title": metadata.title,
                "section": current_section,
                "page_start": section_start_page,
                "page_end": pages[-1].page_num
            }
            section_chunks, current_chunk_index = parent_child_chunk(
                section_text, section_meta, paper_id, start_chunk_index=current_chunk_index
            )
            all_chunks.extend(section_chunks)
            
        # Separate children and parents
        all_children = [c for chunk in all_chunks for c in chunk.children]
        parents = {chunk.parent_id: chunk for chunk in all_chunks}
        
        # Deduplicate children
        all_children = deduplicate_children(all_children)
        
        # Step 6 & 7: Indexing
        if progress_callback:
            progress_callback(0.7, "Building vector and keyword indices...")
            
        faiss_index, children_ordered = build_faiss_index(all_children)
        bm25_index = build_bm25_index(children_ordered)
        
        # Step 8: Persistence
        if progress_callback:
            progress_callback(0.9, "Saving to disk...")
            
        save_index(paper_id, faiss_index, children_ordered, parents, bm25_index, metadata)
        
        if progress_callback:
            progress_callback(1.0, "Ingestion complete.")
            
        return PaperResult(
            metadata=metadata,
            parent_store=parents,
            children=children_ordered,
            faiss_index=faiss_index,
            bm25_index=bm25_index,
            paper_id=paper_id
        )
        
    except Exception as e:
        if progress_callback:
            progress_callback(1.0, f"Error processing {filepath}: {str(e)}")
        print(f"Error processing {filepath}: {e}")
        return None

def ingest_papers(filepaths: List[str], progress_callback: Optional[Callable[[float, str], None]] = None) -> List[PaperResult]:
    """Batch ingest multiple papers."""
    results = []
    total = len(filepaths)
    
    for i, filepath in enumerate(filepaths):
        def local_progress(p: float, msg: str):
            if progress_callback:
                # Scale progress to the overall batch
                overall_p = (i + p) / total
                progress_callback(overall_p, f"[{i+1}/{total}] {msg}")
                
        result = ingest_paper(filepath, local_progress)
        if result:
            results.append(result)
            
    return results
