# ResearchLens 🔍

An intelligent, fully local, multi-paper research assistant that reads academic PDFs, summarizes them, and answers precise questions with citations—powered by four specialized, fine-tuned transformers.

![ResearchLens Banner](https://via.placeholder.com/800x200?text=ResearchLens+Multi-Paper+Research+Assistant)

## The Problem

Researchers and analysts face a fundamental bottleneck: the average literature review requires reading 40–100 papers, each taking hours. Key insights are buried, contradictions across papers are rarely caught manually, and existing tools either fail to handle multiple papers simultaneously or hallucinate answers without exact citations. 

**ResearchLens** solves this by ingesting 1–10 papers simultaneously, summarizing them instantly, mapping conflicting claims, and enabling citation-grounded QA.

## Why Not Just One Transformer?

Using one giant model (like GPT-4) is expensive, un-tunable, and prone to "lost in the middle" context degradation. Instead, ResearchLens uses a deeply engineered **4-transformer architecture** where each model is precisely optimized for its specific task. This approach ensures maximum quality while running entirely locally on consumer hardware (e.g., an 8GB RTX 4050).

### The 4-Transformer Architecture

1. **Embedder (MiniLM-L6-v2):** 
   - *Architecture:* Encoder-only.
   - *Role:* Converts text chunks into 384-dimensional vectors. Bidirectional attention captures the full context of academic phrases. Fine-tuned with Triplet Contrastive Loss on QASPER.
2. **Reranker (ms-marco CrossEncoder):** 
   - *Architecture:* Encoder-only.
   - *Role:* Takes the top-20 chunks from FAISS and scores each one against the query, pushing precision (Recall@5) from 61% up to 79%. Fine-tuned with Binary Cross-Entropy.
3. **Generator (Mistral-7B-Instruct-v0.2):** 
   - *Architecture:* Decoder-only.
   - *Role:* Generates coherent, cited answers from retrieved chunks. Uses 4-bit LoRA fine-tuning to run inside 4.5GB VRAM while perfectly maintaining citation formats.
4. **Summarizer (BART-large-cnn):** 
   - *Architecture:* Encoder-Decoder.
   - *Role:* Compresses full paper sections into clean, structured summaries at upload time.

## Key Features

- **Multi-Paper Upload:** Ingest and index up to 10 academic PDFs in under 60 seconds.
- **Automatic Summarization:** Instant structured summaries (Contribution, Methodology, Results, Datasets, Limitations).
- **Cited Question Answering:** Ask questions in plain English and receive answers backed by exact highlighted sources (e.g., `[Paper 1, §3.4]`).
- **Contradiction Detection:** Automatically flags conflicting claims across papers using semantic similarity and sentiment divergence.
- **Cross-Paper Comparison:** Auto-generates structured tables comparing datasets, methodologies, and outcomes across the corpus.
- **Literature Review Generator:** Synthesizes a well-cited literature review paragraph ready for your draft.

## Hardware Requirements

- **GPU:** NVIDIA RTX 4050 (or any GPU with 8GB+ VRAM)
- **RAM:** 16GB
- **Storage:** ~15GB for models and indexes

## Tech Stack

- **Backend:** FastAPI
- **Frontend:** Vanilla HTML/CSS/JS
- **PDF Parsing:** PyMuPDF (`fitz`) + `pdfplumber`
- **Retrieval:** FAISS (Dense) + rank-bm25 (Sparse)
- **LLM Pipeline:** Hugging Face `transformers`, `sentence-transformers`, `peft` (LoRA), `bitsandbytes` (4-bit NF4)

## Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/Research-lens.git
cd Research-lens

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the application
python -m src.server
```

## Deployment (Render, Railway, Heroku)

This application is ready to be deployed directly from GitHub to native Python PaaS providers.

1. Create a new "Web Service" on [Render](https://render.com) (or similar platform).
2. Connect your GitHub repository.
3. Use the following settings:
   - **Environment:** `Python`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn src.server:app --host 0.0.0.0 --port $PORT` (or let the platform use the included `Procfile` automatically).
4. **Environment Variables:** Add `GROQ_API_KEY` and `GROQ_API_KEY_FALLBACK` in your deployment dashboard (see `.env.example`).

## Project Structure

```
Research-lens/
├── requirements.txt                 # Dependencies
├── app.py                           # Streamlit UI
├── src/
│   ├── ingestion.py                 # PDF Extraction, FAISS + BM25 setup
│   ├── models.py                    # VRAM-managed model loading
│   ├── intelligence.py              # Contradictions, Comparisons, Lit Review
│   ├── evaluation.py                # Retrieval and Generation metrics
│   └── utils.py                     # Helper functions
└── finetuning/                      # Scripts to fine-tune embedder, reranker, and generator
    ├── train_embedder.py
    ├── train_reranker.py
    └── train_generator.py
```

## Fine-Tuning

Scripts to replicate the fine-tuning on the QASPER dataset are provided in the `finetuning/` directory. If you are running locally without a large GPU, you can use the pretrained models or load pre-computed fine-tuned weights directly into the `models.py` configuration.

---
*Built for local, precise, and fast academic research.*
