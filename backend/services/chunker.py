"""
Intelligent legal text chunking.
Splits text at legal-structural boundaries rather than arbitrary character limits.
"""

import re
from typing import List, Dict, Any

# Structural section headers common in Indian court documents
LEGAL_SECTION_PATTERNS = [
    r"^JUDGMENT\b",
    r"^ORDER\b",
    r"^FACTS\b",
    r"^BACKGROUND\b",
    r"^SUBMISSIONS?\b",
    r"^ARGUMENTS?\b",
    r"^ISSUES?\b",
    r"^HELD\b",
    r"^REASONING\b",
    r"^DECISION\b",
    r"^CONCLUSION\b",
    r"^RELIEF\b",
    r"^PETITION\b",
    r"^WRIT PETITION\b",
    r"^CIVIL APPEAL\b",
    r"^CRIMINAL APPEAL\b",
    r"^SPECIAL LEAVE PETITION\b",
    r"^\d+\.\s+[A-Z]",          # numbered sections: "1. Facts"
    r"^[A-Z]{2,}(\s+[A-Z]+)+$", # ALL CAPS headings
]

COMPILED_PATTERNS = [re.compile(p, re.MULTILINE | re.IGNORECASE) for p in LEGAL_SECTION_PATTERNS]

MAX_CHUNK_CHARS  = 1200
MIN_CHUNK_CHARS  = 200
OVERLAP_CHARS    = 150


def chunk_legal_text(
    text: str,
    case_id: str,
    metadata: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Split legal text into semantically meaningful chunks.
    Priority: legal section boundaries → paragraph boundaries → sentence boundaries.
    """
    # 1. Try to split on legal section headers
    sections = _split_on_legal_sections(text)

    chunks = []
    chunk_idx = 0

    for section_title, section_text in sections:
        # 2. Within each section, split on paragraph boundaries
        paragraphs = _split_paragraphs(section_text)

        current_chunk = ""
        current_section = section_title

        for para in paragraphs:
            if not para.strip():
                continue

            if len(current_chunk) + len(para) < MAX_CHUNK_CHARS:
                current_chunk += (" " if current_chunk else "") + para
            else:
                if current_chunk and len(current_chunk) >= MIN_CHUNK_CHARS:
                    chunks.append(_make_chunk(
                        text=current_chunk,
                        section=current_section,
                        idx=chunk_idx,
                        case_id=case_id,
                        metadata=metadata,
                    ))
                    chunk_idx += 1
                    # Overlap: keep last sentence of previous chunk
                    current_chunk = _last_sentences(current_chunk, OVERLAP_CHARS) + " " + para
                else:
                    current_chunk = para

        # Flush remaining
        if current_chunk and len(current_chunk) >= MIN_CHUNK_CHARS:
            chunks.append(_make_chunk(
                text=current_chunk,
                section=current_section,
                idx=chunk_idx,
                case_id=case_id,
                metadata=metadata,
            ))
            chunk_idx += 1

    # Fallback: if no chunks produced, do a naive split
    if not chunks:
        chunks = _naive_split(text, case_id, metadata)

    return chunks


def _split_on_legal_sections(text: str):
    """Return list of (section_title, section_text) tuples."""
    lines = text.split("\n")
    sections = []
    current_title = "PREAMBLE"
    current_lines: List[str] = []

    for line in lines:
        stripped = line.strip()
        is_header = any(p.match(stripped) for p in COMPILED_PATTERNS) and len(stripped) < 80

        if is_header and current_lines:
            sections.append((current_title, "\n".join(current_lines)))
            current_title = stripped
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_title, "\n".join(current_lines)))

    return sections if sections else [("FULL TEXT", text)]


def _split_paragraphs(text: str) -> List[str]:
    """Split on blank lines (paragraph breaks)."""
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _last_sentences(text: str, max_chars: int) -> str:
    """Return the last ~max_chars characters, cut at sentence boundary."""
    if len(text) <= max_chars:
        return text
    tail = text[-max_chars:]
    # Find first sentence boundary in tail
    m = re.search(r"[.!?]\s+", tail)
    if m:
        return tail[m.end():]
    return tail


def _make_chunk(
    text: str,
    section: str,
    idx: int,
    case_id: str,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "id": f"{case_id}_{idx}",
        "case_id": case_id,
        "section": section,
        "text": text.strip(),
        "metadata": {
            **metadata,
            "section": section,
            "chunk_index": idx,
        },
    }


def _naive_split(text: str, case_id: str, metadata: Dict[str, Any]) -> List[Dict]:
    """Last-resort: split every MAX_CHUNK_CHARS characters at sentence boundary."""
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = start + MAX_CHUNK_CHARS
        if end < len(text):
            # Try to cut at sentence boundary
            boundary = text.rfind(". ", start, end)
            if boundary > start:
                end = boundary + 1
        chunk_text = text[start:end].strip()
        if len(chunk_text) >= MIN_CHUNK_CHARS:
            chunks.append(_make_chunk(chunk_text, "TEXT", idx, case_id, metadata))
            idx += 1
        start = end - OVERLAP_CHARS
    return chunks
