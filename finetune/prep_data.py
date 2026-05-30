"""
prep_data.py
============
Builds training data for all 3 fine-tuning jobs from QASPER + SciQ datasets.

Outputs:
  data/processed/triplets/         → embedder training  (triplet loss)
  data/processed/pairs/            → reranker training  (binary relevance)
  data/processed/qa_examples/      → generator training (cited QA)

Run:
  python -m finetune.prep_data
"""

import os
import json
import random
import logging
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass, asdict
from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

random.seed(42)

# ─── Output paths ────────────────────────────────────────────────────────────
TRIPLETS_DIR    = "data/processed/triplets"
PAIRS_DIR       = "data/processed/pairs"
QA_DIR          = "data/processed/qa_examples"
RAW_DIR         = "data/raw"

# ─── Dataclasses ─────────────────────────────────────────────────────────────

@dataclass
class Triplet:
    """For embedder training: anchor=query, positive=evidence, negative=random chunk"""
    anchor:   str      # question
    positive: str      # evidence chunk that answers it
    negative: str      # chunk that does NOT answer it

@dataclass
class RankerPair:
    """For reranker training: (query, chunk) → 1 if relevant, 0 if not"""
    query:    str
    chunk:    str
    label:    int      # 1=relevant, 0=not relevant

@dataclass
class QAExample:
    """For generator training: question + context → cited answer"""
    question:   str
    context:    str    # retrieved chunks formatted with source tags
    answer:     str    # answer with citation format [Source: paper, section]
    paper_title: str

# ─── QASPER Processing ────────────────────────────────────────────────────────

def flatten_qasper_paragraphs(full_text: Dict) -> List[Dict]:
    """
    QASPER full_text structure:
      section_name: List[str]
      paragraphs:   List[List[str]]  ← each section has list of paragraphs
    
    Returns flat list of {"section": str, "text": str}
    """
    sections = full_text.get("section_name", [])
    paragraphs = full_text.get("paragraphs", [])
    flat = []
    for sec, para_list in zip(sections, paragraphs):
        for para in para_list:
            if para.strip():
                flat.append({"section": sec, "text": para.strip()})
    return flat


def extract_evidence_text(answers: Dict) -> Tuple[str, str]:
    """
    Extract the best free-form answer and its evidence from QASPER answer dict.
    
    QASPER answers structure:
      answers: List of annotator dicts, each with:
        answer: List[{free_form_answer, yes_no, extractive_spans, unanswerable}]
        evidence: List[str]
    
    Returns (answer_text, evidence_text) or ("", "") if unanswerable.
    """
    for annotator in answers:
        for ans in annotator.get("answer", []):
            if ans.get("unanswerable", False):
                continue
            # Prefer free-form answer
            free_form = ans.get("free_form_answer", "").strip()
            evidence_list = annotator.get("evidence", [])
            evidence = " ".join([e.strip() for e in evidence_list if e.strip()])
            if free_form and evidence:
                return free_form, evidence
    return "", ""


def build_triplets_from_qasper(dataset) -> List[Triplet]:
    """
    Build (query, positive, negative) triplets for embedder training.
    
    Strategy:
    - anchor   = question
    - positive = evidence paragraph(s) that answer the question
    - negative = random paragraph from SAME paper (hard negative)
                 or random paragraph from different paper (easy negative)
    - Mix 70% hard negatives, 30% easy negatives
    """
    log.info("Building triplets for embedder training...")
    triplets = []
    
    # Collect all paragraphs across all papers for easy negatives
    all_paragraphs = []
    for item in dataset:
        paras = flatten_qasper_paragraphs(item["full_text"])
        all_paragraphs.extend([p["text"] for p in paras])
    
    for item in tqdm(dataset, desc="Triplets"):
        title = item["title"]
        paras = flatten_qasper_paragraphs(item["full_text"])
        para_texts = [p["text"] for p in paras]
        
        if len(para_texts) < 2:
            continue
        
        for qa in item["qas"]:
            question = qa["question"].strip()
            if not question:
                continue
            
            ans_text, evidence = extract_evidence_text(qa["answers"])
            if not evidence or len(evidence) < 30:
                continue
            
            # Hard negative: random para from same paper, not the evidence
            non_evidence = [p for p in para_texts if p != evidence and len(p) > 50]
            if not non_evidence:
                continue
            
            if random.random() < 0.7:
                # Hard negative — same paper
                negative = random.choice(non_evidence)
            else:
                # Easy negative — different paper
                negative = random.choice(all_paragraphs)
                while negative == evidence:
                    negative = random.choice(all_paragraphs)
            
            triplets.append(Triplet(
                anchor=question,
                positive=evidence,
                negative=negative
            ))
    
    log.info(f"Built {len(triplets)} triplets")
    return triplets


def build_ranker_pairs_from_qasper(dataset) -> List[RankerPair]:
    """
    Build (query, chunk, label) pairs for reranker training.
    
    Strategy:
    - Positive pair: (question, evidence) → label=1
    - Negative pair: (question, random non-evidence para) → label=0
    - 1:3 positive:negative ratio (realistic retrieval scenario)
    """
    log.info("Building ranker pairs for reranker training...")
    pairs = []
    
    for item in tqdm(dataset, desc="Ranker pairs"):
        paras = flatten_qasper_paragraphs(item["full_text"])
        para_texts = [p["text"] for p in paras if len(p["text"]) > 50]
        
        if not para_texts:
            continue
        
        for qa in item["qas"]:
            question = qa["question"].strip()
            if not question:
                continue
            
            ans_text, evidence = extract_evidence_text(qa["answers"])
            if not evidence or len(evidence) < 30:
                continue
            
            # 1 positive pair
            pairs.append(RankerPair(
                query=question,
                chunk=evidence,
                label=1
            ))
            
            # 3 negative pairs (hard negatives from same paper)
            non_evidence = [p for p in para_texts if p != evidence]
            negatives = random.sample(non_evidence, min(3, len(non_evidence)))
            for neg in negatives:
                pairs.append(RankerPair(
                    query=question,
                    chunk=neg,
                    label=0
                ))
    
    log.info(f"Built {len(pairs)} ranker pairs ({sum(1 for p in pairs if p.label==1)} positive, {sum(1 for p in pairs if p.label==0)} negative)")
    return pairs


def build_qa_examples_from_qasper(dataset) -> List[QAExample]:
    """
    Build (question, context, cited_answer) examples for Mistral fine-tuning.
    
    This teaches Mistral to:
    1. Answer only from provided context
    2. Always cite the source
    3. Say "Not found" when context doesn't contain the answer
    
    Context format mirrors what retriever will produce at inference time.
    """
    log.info("Building QA examples for generator training...")
    examples = []
    
    for item in tqdm(dataset, desc="QA examples"):
        title  = item["title"]
        paras  = flatten_qasper_paragraphs(item["full_text"])
        
        # Build section lookup: text → section name
        section_lookup = {p["text"]: p["section"] for p in paras}
        para_texts = [p["text"] for p in paras if len(p["text"]) > 50]
        
        if not para_texts:
            continue
        
        for qa in item["qas"]:
            question = qa["question"].strip()
            if not question:
                continue
            
            ans_text, evidence = extract_evidence_text(qa["answers"])
            if not evidence:
                continue
            
            section = section_lookup.get(evidence, "Methods")
            
            # Build context with 1 correct + 2 distractors (simulates retrieval)
            distractors = [p for p in para_texts if p != evidence]
            distractors = random.sample(distractors, min(2, len(distractors)))
            
            # All context chunks with source tags
            context_chunks = [evidence] + distractors
            random.shuffle(context_chunks)  # randomize position
            
            context_str = ""
            correct_source_idx = None
            for i, chunk in enumerate(context_chunks):
                context_str += f"[SOURCE {i+1}: {title}, {section}]\n{chunk}\n\n"
                if chunk == evidence:
                    correct_source_idx = i + 1
            
            # Cited answer — this is what Mistral learns to produce
            cited_answer = f"{ans_text} [SOURCE {correct_source_idx}: {title}, {section}]"
            
            examples.append(QAExample(
                question=question,
                context=context_str.strip(),
                answer=cited_answer,
                paper_title=title
            ))
    
    log.info(f"Built {len(examples)} QA examples")
    return examples


def build_sciq_pairs(sciq_dataset) -> Tuple[List[RankerPair], List[Triplet]]:
    """
    SciQ provides science QA with supporting text.
    Use it to augment reranker pairs and triplets.
    
    SciQ structure:
      question, correct_answer, support (evidence), distractor1/2/3
    """
    log.info("Building SciQ augmentation data...")
    pairs  = []
    triplets = []
    
    for item in tqdm(sciq_dataset, desc="SciQ"):
        question = item["question"].strip()
        support  = item["support"].strip()
        correct  = item["correct_answer"].strip()
        distractors = [
            item.get("distractor1", ""),
            item.get("distractor2", ""),
            item.get("distractor3", "")
        ]
        distractors = [d for d in distractors if d.strip()]
        
        if not support or not question or len(support) < 30:
            continue
        
        # Reranker pair
        pairs.append(RankerPair(query=question, chunk=support, label=1))
        for d in distractors[:2]:
            pairs.append(RankerPair(query=question, chunk=d, label=0))
        
        # Triplet
        if distractors:
            triplets.append(Triplet(
                anchor=question,
                positive=support,
                negative=random.choice(distractors)
            ))
    
    log.info(f"SciQ: {len(pairs)} ranker pairs, {len(triplets)} triplets")
    return pairs, triplets


# ─── Save helpers ─────────────────────────────────────────────────────────────

def save_jsonl(data: List[Any], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
    log.info(f"Saved {len(data)} examples → {path}")


def load_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def split_train_val(data: List, val_ratio: float = 0.1) -> Tuple[List, List]:
    random.shuffle(data)
    split = int(len(data) * (1 - val_ratio))
    return data[:split], data[split:]


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    
    # ── Load QASPER ──────────────────────────────────────────────────────────
    log.info("Loading QASPER dataset...")
    try:
        qasper = load_dataset("allenai/qasper", trust_remote_code=True)
    except Exception:
        qasper = load_dataset("allenai/qasper")
    
    train_data = qasper["train"]
    val_data   = qasper["validation"]
    log.info(f"QASPER train: {len(train_data)} papers | val: {len(val_data)} papers")

    # ── Load SciQ ────────────────────────────────────────────────────────────
    log.info("Loading SciQ dataset...")
    try:
        sciq = load_dataset("sciq")
        sciq_train = sciq["train"]
        log.info(f"SciQ train: {len(sciq_train)} examples")
    except Exception as e:
        log.warning(f"SciQ load failed: {e}. Skipping SciQ augmentation.")
        sciq_train = []

    # ── Build training data ──────────────────────────────────────────────────

    # 1. Triplets for embedder
    train_triplets = build_triplets_from_qasper(train_data)
    val_triplets   = build_triplets_from_qasper(val_data)
    
    if sciq_train:
        sciq_ranker_pairs, sciq_triplets = build_sciq_pairs(sciq_train)
        train_triplets.extend(sciq_triplets)
    
    save_jsonl(train_triplets, f"{TRIPLETS_DIR}/train.jsonl")
    save_jsonl(val_triplets,   f"{TRIPLETS_DIR}/val.jsonl")

    # 2. Ranker pairs for reranker
    train_pairs = build_ranker_pairs_from_qasper(train_data)
    val_pairs   = build_ranker_pairs_from_qasper(val_data)
    
    if sciq_train:
        train_pairs.extend(sciq_ranker_pairs)
    
    # Shuffle to mix positive/negative
    random.shuffle(train_pairs)
    save_jsonl(train_pairs, f"{PAIRS_DIR}/train.jsonl")
    save_jsonl(val_pairs,   f"{PAIRS_DIR}/val.jsonl")

    # 3. QA examples for generator
    train_qa = build_qa_examples_from_qasper(train_data)
    val_qa   = build_qa_examples_from_qasper(val_data)
    
    save_jsonl(train_qa, f"{QA_DIR}/train.jsonl")
    save_jsonl(val_qa,   f"{QA_DIR}/val.jsonl")

    # ── Summary ──────────────────────────────────────────────────────────────
    log.info("\n" + "="*60)
    log.info("DATA PREPARATION COMPLETE")
    log.info("="*60)
    log.info(f"Embedder triplets  | train: {len(train_triplets):,} | val: {len(val_triplets):,}")
    log.info(f"Reranker pairs     | train: {len(train_pairs):,}   | val: {len(val_pairs):,}")
    log.info(f"Generator QA       | train: {len(train_qa):,}      | val: {len(val_qa):,}")
    log.info("="*60)
    log.info("Next: run finetune/train_embedder.py")


if __name__ == "__main__":
    main()
