import os
import shutil
import tempfile
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

from src.ingestion import ingest_paper
from src.indexer import load_index
from src.pipeline import ask_question
from src.intelligence import (
    summarize_paper, 
    detect_contradictions, 
    generate_comparison_table, 
    generate_literature_review,
    generate_hypotheses
)
from src.utils import UnifiedIndex, PaperResult

app = FastAPI(title="ResearchLens Premium API")

# --- Global State (For a local single-user app) ---
GLOBAL_STATE = {
    "unified_indices": {},   # paper_id -> UnifiedIndex
    "paper_results": {},     # paper_id -> PaperResult
}

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    query: str
    history: List[Dict[str, str]] = []

class SummarizeRequest(BaseModel):
    paper_id: str

class IntelligenceRequest(BaseModel):
    action: str  # "compare", "contradictions", "review", "hypotheses"

# --- API Endpoints ---

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Uploads a PDF, processes it, and adds it to the knowledge base."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")
        
    try:
        # Save to temp and ensure it is fully closed before ingestion
        tmp_path = ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
            
        # Ingest
        result = ingest_paper(tmp_path)
        os.unlink(tmp_path)
        
        if not result:
            raise HTTPException(status_code=400, detail="Could not extract meaningful text from PDF.")
            
        # Load index
        unified, _, _ = load_index(result.paper_id)
        
        # Store in global state
        GLOBAL_STATE["unified_indices"][result.paper_id] = unified
        GLOBAL_STATE["paper_results"][result.paper_id] = result
        
        return {
            "success": True,
            "paper_id": result.paper_id,
            "title": result.metadata.title,
            "year": result.metadata.year,
            "n_pages": result.metadata.n_pages
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/papers")
def list_papers():
    """List all loaded papers."""
    papers = []
    for pid, pr in GLOBAL_STATE["paper_results"].items():
        papers.append({
            "paper_id": pid,
            "title": pr.metadata.title,
            "year": pr.metadata.year
        })
    return {"papers": papers}

@app.post("/api/chat")
def chat(req: ChatRequest):
    """Ask a question across all papers."""
    indices = list(GLOBAL_STATE["unified_indices"].values())
    if not indices:
        return {"answer": "Please upload papers first."}
        
    try:
        answer = ask_question(req.query, indices, req.history)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/summarize")
def summarize(req: SummarizeRequest):
    """Generate a detailed summary for a specific paper."""
    if req.paper_id not in GLOBAL_STATE["paper_results"]:
        raise HTTPException(status_code=404, detail="Paper not found.")
        
    pr = GLOBAL_STATE["paper_results"][req.paper_id]
    try:
        summary = summarize_paper(pr)
        return {
            "title": summary.title,
            "contribution": summary.contribution,
            "methodology": summary.methodology,
            "results": summary.results,
            "datasets": summary.datasets,
            "limitations": summary.limitations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/intelligence")
def run_intelligence(req: IntelligenceRequest):
    """Run cross-paper analytics."""
    prs = list(GLOBAL_STATE["paper_results"].values())
    if len(prs) < 2 and req.action != "hypotheses": # Hypotheses can run on 1 paper technically
        raise HTTPException(status_code=400, detail="Please upload at least 2 papers for cross-paper intelligence.")
        
    try:
        if req.action == "compare":
            rows = generate_comparison_table(prs)
            result = [{"dimension": r.dimension, "values": r.values} for r in rows]
            return {"type": "table", "data": result}
            
        elif req.action == "contradictions":
            contradictions = detect_contradictions(prs)
            result = [{
                "paper_a": c.paper_a,
                "paper_b": c.paper_b,
                "claim_a": c.claim_a,
                "claim_b": c.claim_b,
                "explanation": c.explanation
            } for c in contradictions]
            return {"type": "contradictions", "data": result}
            
        elif req.action == "review":
            review = generate_literature_review(prs)
            return {"type": "text", "data": review}
            
        elif req.action == "hypotheses":
            hypotheses = generate_hypotheses(prs)
            return {"type": "text", "data": hypotheses}
            
        else:
            raise HTTPException(status_code=400, detail="Unknown action")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/clear")
def clear_memory():
    """Clear all loaded papers."""
    GLOBAL_STATE["unified_indices"].clear()
    GLOBAL_STATE["paper_results"].clear()
    return {"success": True}


# --- Frontend Serving ---
# Ensure frontend directory exists
os.makedirs("frontend", exist_ok=True)
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse("frontend/index.html")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    # Disable reload in production (when PORT is set via PaaS)
    reload = os.environ.get("PORT") is None
    uvicorn.run("src.server:app", host="0.0.0.0", port=port, reload=reload)
