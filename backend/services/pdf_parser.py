"""
PDF parsing service using PyMuPDF (fitz).
Handles messy Indian court document formatting.
"""

import re
import fitz   # PyMuPDF


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract and clean text from a PDF file.
    Handles multi-column layouts and header/footer noise common in Indian courts.
    """
    doc = fitz.open(pdf_path)
    pages_text = []

    for page_num, page in enumerate(doc):
        # Use "text" mode which preserves reading order
        text = page.get_text("text")
        cleaned = _clean_page_text(text, page_num)
        if cleaned.strip():
            pages_text.append(cleaned)

    doc.close()
    full_text = "\n\n".join(pages_text)
    return _post_process(full_text)


def _clean_page_text(text: str, page_num: int) -> str:
    """Per-page cleaning."""
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        line = line.strip()

        # Skip blank lines but preserve paragraph breaks
        if not line:
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue

        # Skip pure page numbers (e.g. "1", "Page 1 of 23")
        if re.match(r"^(Page\s+)?\d+(\s+of\s+\d+)?$", line, re.IGNORECASE):
            continue

        # Skip very short lines that are likely artifacts
        if len(line) < 3:
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def _post_process(text: str) -> str:
    """Global text cleanup for Indian legal documents."""
    # Normalize whitespace
    text = re.sub(r" {2,}", " ", text)

    # Collapse more than 3 consecutive blank lines to 2
    text = re.sub(r"\n{4,}", "\n\n\n", text)

    # Fix common OCR issues in Indian legal docs
    # e.g. "Pe t it io n er" → "Petitioner"
    text = re.sub(r"\b([A-Z])\s([a-z])\s([a-z])\b", r"\1\2\3", text)

    # Fix "Vs." / "v/s" / "Versus" normalization
    text = re.sub(r"\bv/s\b", "vs.", text, flags=re.IGNORECASE)
    text = re.sub(r"\bversus\b", "vs.", text, flags=re.IGNORECASE)

    # Remove form-feed characters
    text = text.replace("\x0c", "\n\n")

    return text.strip()
