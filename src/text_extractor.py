import re
import fitz
import pdfplumber
from typing import List, Tuple
from src.utils import PageData, PaperMetadata

SECTION_PATTERNS = [
    r"^abstract$",
    r"^1\.?\s*introduction",
    r"^2\.?\s*(?:related work|background)",
    r"^3\.?\s*(?:method|methodology|our approach)",
    r"^4\.?\s*experiment",
    r"^5\.?\s*(?:result|results|evaluation)",
    r"^6\.?\s*discussion",
    r"^7\.?\s*conclusion",
    r"^references$",
    r"^appendix",
    r"^\d+\.?\s+[A-Z][a-z]+"    # any numbered section
]

def extract_text(filepath: str) -> List[PageData]:
    """
    Extracts text from PDF, preferring PyMuPDF blocks for multi-column.
    Falls back to pdfplumber if text is < 500 chars.
    """
    doc = fitz.open(filepath)
    pages = []
    
    for page_num, page in enumerate(doc):
        # Extract font sizes
        font_sizes = {}
        dict_blocks = page.get_text("dict").get("blocks", [])
        for b in dict_blocks:
            if b.get("type", -1) == 0:
                block_text = "".join([s.get("text", "") for l in b.get("lines", []) for s in l.get("spans", [])]).strip().lower()
                spans = [s.get("size", 0) for l in b.get("lines", []) for s in l.get("spans", [])]
                max_size = max(spans) if spans else 0
                if block_text:
                    font_sizes[block_text] = max_size

        # Use get_text("blocks") for better layout handling
        blocks = page.get_text("blocks")
        # Sort top-to-bottom, left-to-right to reconstruct columns
        # y0 is b[1], x0 is b[0]
        blocks.sort(key=lambda b: (round(b[1] / 30) * 30, b[0]))
        text = "\n".join(b[4].strip() for b in blocks if b[6] == 0) # type 0 is text
        
        pages.append(PageData(
            page_num=page_num + 1,
            text=text,
            width=page.rect.width,
            height=page.rect.height,
            font_sizes=font_sizes
        ))
        
    total_text = " ".join([p.text for p in pages])
    if len(total_text.strip()) < 500:
        pages = _extract_with_pdfplumber(filepath)
        
    doc.close()
    return pages

def _extract_with_pdfplumber(filepath: str) -> List[PageData]:
    pages = []
    with pdfplumber.open(filepath) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append(PageData(
                page_num=page_num + 1,
                text=text,
                width=float(page.width),
                height=float(page.height)
            ))
    return pages

def extract_metadata(filepath: str, pages: List[PageData]) -> PaperMetadata:
    """Extract metadata using a layered heuristic approach."""
    doc = fitz.open(filepath)
    meta = doc.metadata
    
    title = meta.get("title", "").strip()
    authors = meta.get("author", "").strip()
    year = ""
    
    # Try year from creationDate (format: D:YYYYMMDDHHmmSSZ)
    cdate = meta.get("creationDate", "")
    if cdate and cdate.startswith("D:"):
        year = cdate[2:6]
        
    # Heuristic 1: title from first page largest text
    if not title or len(title) < 5 or "Microsoft Word" in title:
        first_page = doc[0]
        blocks = first_page.get_text("dict")["blocks"]
        title_candidates = []
        for b in blocks:
            if b["type"] == 0:
                for l in b["lines"]:
                    for s in l["spans"]:
                        title_candidates.append((s["text"], s["size"]))
        
        if title_candidates:
            # Get largest font size text
            title_candidates.sort(key=lambda x: x[1], reverse=True)
            best_title = " ".join([t[0] for t in title_candidates if t[1] == title_candidates[0][1]])
            title = best_title.strip()
            
    # Heuristic 2: year from regex on first page
    if not year and pages:
        match = re.search(r"(19|20)\d{2}", pages[0].text)
        if match:
            year = match.group(0)
            
    # Heuristic 3: authors from first page text before abstract
    if not authors and pages:
        lines = pages[0].text.split("\n")
        author_lines = []
        for line in lines:
            if re.match(r"^abstract$", line.strip(), re.IGNORECASE):
                break
            if line.strip() and line.strip() != title:
                # Add if looks like author line (commas, university, emails)
                if "," in line or "University" in line or "@" in line:
                    author_lines.append(line.strip())
        if author_lines:
            authors = "; ".join(author_lines)

    doc.close()
    
    return PaperMetadata(
        title=title if title else "Unknown Title",
        authors=authors if authors else "Unknown Authors",
        year=year if year else "Unknown Year",
        doi=meta.get("doi", "Unknown DOI"),
        n_pages=len(pages),
        filepath=filepath
    )

def detect_sections(pages: List[PageData], filepath: str) -> List[PageData]:
    """Detect sections using regex and font size heuristics."""
    current_section = "Abstract"
    
    font_sizes = {}
    all_sizes = []
    
    for page in pages:
        font_sizes.update(page.font_sizes)
        all_sizes.extend(page.font_sizes.values())
        
    median_size = sorted(all_sizes)[len(all_sizes)//2] if all_sizes else 10
    
    for page in pages:
        lines = page.text.split("\n")
        for line in lines:
            line_clean = line.strip().lower()
            if not line_clean:
                continue
                
            # Regex match
            matched = False
            for pattern in SECTION_PATTERNS:
                if re.match(pattern, line_clean, re.IGNORECASE):
                    current_section = line.strip()
                    matched = True
                    break
            
            # Font size heuristic
            if not matched and line_clean in font_sizes:
                size = font_sizes[line_clean]
                # If short, single line, and larger font -> likely header
                if size > median_size + 1.5 and len(line_clean) < 60:
                    current_section = line.strip()
                    
        page.section = current_section
        
    return pages
