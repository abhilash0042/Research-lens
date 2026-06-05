"""
test_api.py
===========
End-to-end API tests for every ResearchLens endpoint using FastAPI TestClient.
Creates synthetic PDFs, uploads them, and exercises all features.
"""

import os
import shutil
import fitz
import pytest
from fastapi.testclient import TestClient


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """Create a TestClient for the FastAPI app."""
    from src.server import app, GLOBAL_STATE
    # Start with a clean state
    GLOBAL_STATE["unified_indices"].clear()
    GLOBAL_STATE["paper_results"].clear()
    with TestClient(app) as c:
        yield c
    # Cleanup after all tests
    GLOBAL_STATE["unified_indices"].clear()
    GLOBAL_STATE["paper_results"].clear()


@pytest.fixture(scope="module")
def synthetic_pdfs():
    """Create 2 synthetic academic PDFs for testing."""
    papers = [
        {
            "filename": "test_api_paper_1.pdf",
            "title": "Deep Learning for Image Classification",
            "author": "Alice Researcher",
            "content": (
                "Abstract\n"
                "This paper proposes a novel convolutional neural network architecture for image classification. "
                "We achieve 95.2% accuracy on CIFAR-10 using a residual attention mechanism. "
                "Our method outperforms existing baselines by 3.1 percentage points.\n\n"
                "1. Introduction\n"
                "Image classification is a fundamental task in computer vision. "
                "Recent advances in deep learning have significantly improved accuracy. "
                "However, most methods require extensive computational resources.\n\n"
                "2. Methodology\n"
                "We propose ResAttNet, which combines residual connections with self-attention layers. "
                "The model uses 12 convolutional blocks followed by 4 attention heads. "
                "Training uses AdamW optimizer with cosine annealing schedule.\n\n"
                "3. Results\n"
                "On CIFAR-10, our model achieves 95.2% test accuracy. "
                "On ImageNet, we reach 78.4% top-1 accuracy. "
                "The model trains in 8 hours on a single A100 GPU.\n\n"
                "4. Conclusion\n"
                "ResAttNet demonstrates that attention mechanisms can improve CNN performance. "
                "Future work includes extending to video classification tasks."
            )
        },
        {
            "filename": "test_api_paper_2.pdf",
            "title": "Transformer Models for Natural Language Processing",
            "author": "Bob Scientist",
            "content": (
                "Abstract\n"
                "We present a comprehensive study of transformer architectures for NLP tasks. "
                "Our fine-tuned model achieves state-of-the-art results on GLUE benchmark with 89.7% average score.\n\n"
                "1. Introduction\n"
                "Natural language processing has been revolutionized by transformer models. "
                "Self-attention allows capturing long-range dependencies in text. "
                "However, computational cost scales quadratically with sequence length.\n\n"
                "2. Methodology\n"
                "We fine-tune a 350M parameter transformer on GLUE tasks. "
                "We use mixed-precision training and gradient accumulation. "
                "The learning rate follows a linear warmup schedule.\n\n"
                "3. Results\n"
                "Our model achieves 89.7% average score on GLUE. "
                "On SQuAD 2.0, we reach 88.3% F1 score. "
                "Inference takes 15ms per query on V100.\n\n"
                "4. Discussion\n"
                "Transformers are computationally expensive but highly effective. "
                "Our results confirm that scale matters for NLP performance.\n\n"
                "5. Conclusion\n"
                "We recommend transformer architectures for production NLP systems. "
                "Future work includes efficient attention mechanisms to reduce compute costs."
            )
        }
    ]

    filenames = []
    for paper in papers:
        doc = fitz.open()
        # Create multiple pages with enough text for proper chunking
        page = doc.new_page()
        page.insert_text((50, 50), paper["title"], fontsize=18)
        page.insert_text((50, 80), f"By {paper['author']}", fontsize=12)

        y = 120
        for line in paper["content"].split("\n"):
            if y > 750:
                page = doc.new_page()
                y = 50
            page.insert_text((50, y), line, fontsize=11)
            y += 16

        doc.set_metadata({"title": paper["title"], "author": paper["author"]})
        doc.save(paper["filename"])
        doc.close()
        filenames.append(paper["filename"])

    yield filenames

    # Cleanup
    for f in filenames:
        if os.path.exists(f):
            os.remove(f)
    if os.path.exists("data/indices"):
        shutil.rmtree("data/indices", ignore_errors=True)


# ─── Tests ───────────────────────────────────────────────────────────────────

class TestFrontendServing:
    """Test that the frontend is served correctly."""

    def test_serve_frontend(self, client):
        res = client.get("/")
        assert res.status_code == 200
        assert "ResearchLens" in res.text


class TestUpload:
    """Test PDF upload functionality."""

    def test_upload_pdf(self, client, synthetic_pdfs):
        for pdf_path in synthetic_pdfs:
            with open(pdf_path, "rb") as f:
                res = client.post("/api/upload", files={"file": (pdf_path, f, "application/pdf")})
            assert res.status_code == 200, f"Upload failed: {res.json()}"
            data = res.json()
            assert data["success"] is True
            assert "paper_id" in data
            assert "title" in data
            assert len(data["title"]) > 0

    def test_upload_non_pdf(self, client):
        """Should reject non-PDF files."""
        res = client.post(
            "/api/upload",
            files={"file": ("test.txt", b"This is not a PDF", "text/plain")}
        )
        assert res.status_code == 400
        assert "PDF" in res.json()["detail"]


class TestPapers:
    """Test paper listing."""

    def test_list_papers(self, client, synthetic_pdfs):
        # Ensure papers are uploaded first
        for pdf_path in synthetic_pdfs:
            with open(pdf_path, "rb") as f:
                client.post("/api/upload", files={"file": (pdf_path, f, "application/pdf")})

        res = client.get("/api/papers")
        assert res.status_code == 200
        data = res.json()
        assert len(data["papers"]) >= 2
        for paper in data["papers"]:
            assert "paper_id" in paper
            assert "title" in paper
            assert "year" in paper


class TestChat:
    """Test the chat/QA functionality."""

    def test_chat(self, client, synthetic_pdfs):
        # Ensure papers are uploaded
        for pdf_path in synthetic_pdfs:
            with open(pdf_path, "rb") as f:
                client.post("/api/upload", files={"file": (pdf_path, f, "application/pdf")})

        res = client.post(
            "/api/chat",
            json={"query": "What accuracy was achieved on CIFAR-10?", "history": []}
        )
        assert res.status_code == 200
        data = res.json()
        assert "answer" in data
        assert len(data["answer"]) > 10  # Should have a substantive answer

    def test_chat_no_papers(self, client):
        """When no papers are loaded, should return a helpful message."""
        from src.server import GLOBAL_STATE
        # Temporarily clear state
        saved_indices = dict(GLOBAL_STATE["unified_indices"])
        saved_results = dict(GLOBAL_STATE["paper_results"])
        GLOBAL_STATE["unified_indices"].clear()
        GLOBAL_STATE["paper_results"].clear()

        res = client.post(
            "/api/chat",
            json={"query": "What is the methodology?", "history": []}
        )
        assert res.status_code == 200
        assert "upload" in res.json()["answer"].lower()

        # Restore state
        GLOBAL_STATE["unified_indices"].update(saved_indices)
        GLOBAL_STATE["paper_results"].update(saved_results)


class TestSummarize:
    """Test paper summarization."""

    def test_summarize(self, client, synthetic_pdfs):
        # Ensure papers are uploaded and get the paper_id
        for pdf_path in synthetic_pdfs:
            with open(pdf_path, "rb") as f:
                client.post("/api/upload", files={"file": (pdf_path, f, "application/pdf")})

        papers_res = client.get("/api/papers")
        paper_id = papers_res.json()["papers"][0]["paper_id"]

        res = client.post("/api/summarize", json={"paper_id": paper_id})
        assert res.status_code == 200
        data = res.json()
        # All 6 fields should be present
        for field in ["title", "contribution", "methodology", "results", "datasets", "limitations"]:
            assert field in data, f"Missing field: {field}"
            assert len(str(data[field])) > 0

    def test_summarize_not_found(self, client):
        """Should return 404 for an invalid paper_id."""
        res = client.post("/api/summarize", json={"paper_id": "nonexistent_paper_id"})
        assert res.status_code == 404


class TestIntelligence:
    """Test all cross-paper intelligence features."""

    def _ensure_papers_loaded(self, client, synthetic_pdfs):
        for pdf_path in synthetic_pdfs:
            with open(pdf_path, "rb") as f:
                client.post("/api/upload", files={"file": (pdf_path, f, "application/pdf")})

    def test_compare(self, client, synthetic_pdfs):
        self._ensure_papers_loaded(client, synthetic_pdfs)
        res = client.post("/api/intelligence", json={"action": "compare"})
        assert res.status_code == 200
        data = res.json()
        assert data["type"] == "table"
        assert isinstance(data["data"], list)
        if len(data["data"]) > 0:
            assert "dimension" in data["data"][0]
            assert "values" in data["data"][0]

    def test_contradictions(self, client, synthetic_pdfs):
        self._ensure_papers_loaded(client, synthetic_pdfs)
        res = client.post("/api/intelligence", json={"action": "contradictions"})
        assert res.status_code == 200
        data = res.json()
        assert data["type"] == "contradictions"
        assert isinstance(data["data"], list)

    def test_review(self, client, synthetic_pdfs):
        self._ensure_papers_loaded(client, synthetic_pdfs)
        res = client.post("/api/intelligence", json={"action": "review"})
        assert res.status_code == 200
        data = res.json()
        assert data["type"] == "text"
        assert len(data["data"]) > 50

    def test_hypotheses(self, client, synthetic_pdfs):
        self._ensure_papers_loaded(client, synthetic_pdfs)
        res = client.post("/api/intelligence", json={"action": "hypotheses"})
        assert res.status_code == 200
        data = res.json()
        assert data["type"] == "text"
        assert len(data["data"]) > 50

    def test_intelligence_no_papers(self, client):
        """Should return 400 when no papers are loaded."""
        from src.server import GLOBAL_STATE
        saved_indices = dict(GLOBAL_STATE["unified_indices"])
        saved_results = dict(GLOBAL_STATE["paper_results"])
        GLOBAL_STATE["unified_indices"].clear()
        GLOBAL_STATE["paper_results"].clear()

        res = client.post("/api/intelligence", json={"action": "compare"})
        assert res.status_code == 400

        GLOBAL_STATE["unified_indices"].update(saved_indices)
        GLOBAL_STATE["paper_results"].update(saved_results)

    def test_unknown_action(self, client, synthetic_pdfs):
        self._ensure_papers_loaded(client, synthetic_pdfs)
        res = client.post("/api/intelligence", json={"action": "invalid_action"})
        assert res.status_code == 400


class TestClearMemory:
    """Test memory clearing."""

    def test_clear_memory(self, client, synthetic_pdfs):
        # Ensure papers are loaded first
        for pdf_path in synthetic_pdfs:
            with open(pdf_path, "rb") as f:
                client.post("/api/upload", files={"file": (pdf_path, f, "application/pdf")})

        # Clear
        res = client.post("/api/clear")
        assert res.status_code == 200
        assert res.json()["success"] is True

        # Verify papers list is now empty
        papers_res = client.get("/api/papers")
        assert len(papers_res.json()["papers"]) == 0
