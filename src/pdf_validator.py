import os
import fitz
from src.utils import ValidationResult

def validate_pdf(filepath: str) -> ValidationResult:
    """
    Validates a PDF file.
    Checks: file exists, .pdf extension, size < 50MB, readable by PyMuPDF,
    has embedded fonts (not scanned), has meaningful text (>= 200 chars).
    """
    errors = []
    warnings = []
    is_valid = True
    is_scanned = False
    
    # Check if exists
    if not os.path.exists(filepath):
        errors.append(f"File not found: {filepath}")
        return ValidationResult(is_valid=False, errors=errors)
        
    # Check extension
    if not filepath.lower().endswith(".pdf"):
        errors.append(f"Not a PDF file: {filepath}")
        return ValidationResult(is_valid=False, errors=errors)
        
    # Check size < 50MB
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    if size_mb > 50:
        errors.append(f"File too large: {size_mb:.2f}MB (max 50MB)")
        return ValidationResult(is_valid=False, errors=errors)
        
    try:
        doc = fitz.open(filepath)
    except Exception as e:
        errors.append(f"Cannot open PDF: {str(e)}")
        return ValidationResult(is_valid=False, errors=errors)
        
    if doc.page_count < 1:
        errors.append("PDF has no pages.")
        return ValidationResult(is_valid=False, errors=errors)
        
    # Check for embedded fonts and meaningful text
    has_fonts = False
    total_text = ""
    for page in doc:
        fonts = page.get_fonts()
        if fonts:
            has_fonts = True
        total_text += page.get_text("text")
        
    if not has_fonts:
        is_scanned = True
        errors.append("No embedded fonts detected. This appears to be a scanned PDF and OCR is not supported.")
        is_valid = False
        
    if len(total_text.strip()) < 200:
        errors.append(f"Not enough meaningful text found (only {len(total_text.strip())} chars).")
        is_valid = False
        
    doc.close()
    
    return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings, is_scanned=is_scanned)
