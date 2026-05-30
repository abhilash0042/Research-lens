import os
import fitz
import logging
from src.ingestion import ingest_paper
from src.indexer import load_index
from src.pipeline import ask_question
from src.intelligence import generate_comparison_table, detect_contradictions, generate_literature_review

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

PDFS_TO_CREATE = [
    ("Study 1", "Caffeine increases programming speed by 20% but decreases code quality by 10%."),
    ("Study 2", "Caffeine has no effect on programming speed, but increases anxiety."),
    ("Study 3", "Caffeine actually decreases programming speed by 5% due to jitteriness."),
    ("Study 4", "Tea is better than coffee for programming because it provides stable energy."),
    ("Study 5", "Programmers who drink coffee type 25% faster and have 15% fewer bugs.")
]

def create_synthetic_pdfs():
    filenames = []
    for i, (title, claim) in enumerate(PDFS_TO_CREATE):
        filename = f"test_paper_{i+1}.pdf"
        filenames.append(filename)
        doc = fitz.open()
        
        page = doc.new_page()
        page.insert_text((50, 50), title, fontsize=20)
        
        y = 100
        for _ in range(15):
            page.insert_text((50, y), claim, fontsize=11)
            y += 20
            
        doc.set_metadata({"title": title, "author": f"Author {i+1}"})
        doc.save(filename)
        doc.close()
    return filenames

def main():
    pdf_files = []
    try:
        # 1. Create 5 PDFs
        log.info("--- 1. Creating 5 Synthetic PDFs ---")
        pdf_files = create_synthetic_pdfs()

        # 2. Ingest all 5 PDFs
        log.info("--- 2. Ingesting 5 PDFs ---")
        paper_results = []
        unified_indices = []
        
        for pdf in pdf_files:
            log.info(f"Ingesting {pdf}...")
            result = ingest_paper(pdf)
            paper_results.append(result)
            
            # Load index to memory
            unified, _, _ = load_index(result.paper_id)
            unified_indices.append(unified)
            
        log.info(f"Successfully ingested {len(paper_results)} PDFs!")

        # 3. Question Answering across all 5
        log.info("--- 3. Testing QA Across 5 Papers ---")
        query = "What is the effect of caffeine on programming speed across these studies?"
        answer = ask_question(query, unified_indices)
        log.info(f"Groq Aggregate Answer:\n{answer}")

        # 4. Intelligence: Contradictions
        log.info("--- 4. Detecting Contradictions Between the 5 Papers ---")
        contradictions = detect_contradictions(paper_results)
        log.info(f"Found {len(contradictions)} contradictions.")
        for i, c in enumerate(contradictions):
            log.info(f"Contradiction {i+1}: {c.paper_a} vs {c.paper_b} -> {c.explanation}")

        # 5. Intelligence: Comparison Table
        log.info("--- 5. Generating Comparison Table for 5 Papers ---")
        table = generate_comparison_table(paper_results)
        log.info(f"Generated comparison table with {len(table)} dimensions.")

        # 6. Intelligence: Literature Review
        log.info("--- 6. Generating Unified Literature Review ---")
        lit_review = generate_literature_review(paper_results, focus_topic="Caffeine and coding")
        log.info(f"Literature Review:\n{lit_review}")

        log.info("ALL TESTS PASSED WITH 5 PDFS!")

    except Exception as e:
        log.error(f"Test failed: {e}", exc_info=True)
    finally:
        for pdf in pdf_files:
            if os.path.exists(pdf):
                os.remove(pdf)
        log.info("Cleaned up test PDFs.")

if __name__ == "__main__":
    main()
