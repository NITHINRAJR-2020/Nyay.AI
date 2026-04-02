"""
Metadata extraction from Indian legal documents.
Uses the Claude API (or falls back to regex heuristics) to extract:
- Case Name, Court, Date, Judges, Parties
"""

import os
import re
import json
import requests
from typing import Dict, Any

# ── LLM call ──────────────────────────────────────────────────────────────────

def _call_llm(prompt: str) -> str:
    """Call available LLM API. Priority: Anthropic → OpenAI → Gemini."""
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key    = os.getenv("OPENAI_API_KEY")
    gemini_key    = os.getenv("GEMINI_API_KEY")

    if anthropic_key:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": anthropic_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 512,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["content"][0]["text"]

    elif openai_key:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"},
            json={
                "model": "gpt-3.5-turbo",
                "max_tokens": 512,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    elif gemini_key:
        import google.generativeai as genai
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text

    else:
        raise EnvironmentError(
            "No LLM API key found. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY."
        )


# ── main extraction ────────────────────────────────────────────────────────────

METADATA_PROMPT = """You are an expert in Indian law. Extract structured metadata from the following excerpt of an Indian court document.

Return ONLY a valid JSON object (no markdown, no explanation) with these keys:
- case_name: Full case title (e.g. "State of Maharashtra vs. Ram Lal Gupta")
- court: Court name (e.g. "Supreme Court of India", "Bombay High Court")
- date: Date of judgment in DD/MM/YYYY or Month DD, YYYY format. Use "Unknown" if not found.
- judges: List of judge names (e.g. ["Justice D.Y. Chandrachud", "Justice A.S. Bopanna"])
- parties: Object with "petitioner" and "respondent" keys
- case_number: Case number if found (e.g. "W.P. (C) 123/2021"), else null
- subject_matter: 2-5 word topic (e.g. "Land acquisition compensation", "Bail under NDPS Act")

DOCUMENT EXCERPT:
{text}

JSON:"""


def extract_metadata(text: str) -> Dict[str, Any]:
    """
    Extract metadata from the first portion of a legal document.
    Falls back to regex heuristics if LLM unavailable.
    """
    try:
        prompt = METADATA_PROMPT.format(text=text[:3500])
        raw = _call_llm(prompt).strip()

        # Strip markdown fences if present
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        data = json.loads(raw)
        return _validate_metadata(data)

    except Exception as e:
        print(f"[metadata_extractor] LLM failed ({e}), using regex fallback")
        return _regex_fallback(text)


def _validate_metadata(data: dict) -> dict:
    """Ensure all keys exist with sensible defaults."""
    return {
        "case_name":      data.get("case_name") or "Unknown Case",
        "court":          data.get("court") or "Unknown Court",
        "date":           data.get("date") or "Unknown",
        "judges":         data.get("judges") or [],
        "parties": {
            "petitioner": (data.get("parties") or {}).get("petitioner") or "Unknown",
            "respondent": (data.get("parties") or {}).get("respondent") or "Unknown",
        },
        "case_number":    data.get("case_number"),
        "subject_matter": data.get("subject_matter") or "General",
    }


# ── regex fallback ─────────────────────────────────────────────────────────────

def _regex_fallback(text: str) -> Dict[str, Any]:
    """Best-effort regex extraction for Indian legal documents."""
    meta = {
        "case_name": "Unknown Case",
        "court": "Unknown Court",
        "date": "Unknown",
        "judges": [],
        "parties": {"petitioner": "Unknown", "respondent": "Unknown"},
        "case_number": None,
        "subject_matter": "General",
    }

    # Court detection
    court_patterns = {
        "Supreme Court of India": r"supreme court of india",
        "Delhi High Court": r"(delhi|high court of delhi)",
        "Bombay High Court": r"(bombay|high court.*bombay)",
        "Calcutta High Court": r"(calcutta|high court.*calcutta)",
        "Madras High Court": r"(madras|high court.*madras)",
        "Allahabad High Court": r"(allahabad|high court.*allahabad)",
        "High Court": r"high court",
        "District Court": r"district court",
        "Sessions Court": r"sessions court",
        "National Company Law Tribunal": r"nclt|national company law tribunal",
    }
    text_lower = text.lower()
    for court_name, pattern in court_patterns.items():
        if re.search(pattern, text_lower):
            meta["court"] = court_name
            break

    # Date extraction (DD/MM/YYYY or Month DD, YYYY)
    date_match = re.search(
        r"\b(\d{1,2}[/\-]\d{1,2}[/\-]\d{4})\b|"
        r"\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December),?\s+\d{4})\b",
        text, re.IGNORECASE
    )
    if date_match:
        meta["date"] = (date_match.group(1) or date_match.group(2)).strip()

    # Parties: look for "X vs. Y" or "X v. Y"
    party_match = re.search(
        r"([A-Z][A-Za-z\s\.\,]{3,60}?)\s+(?:vs?\.?|versus)\s+([A-Z][A-Za-z\s\.\,]{3,60})",
        text
    )
    if party_match:
        petitioner = party_match.group(1).strip().rstrip(",")
        respondent = party_match.group(2).strip().rstrip(",")
        meta["parties"] = {"petitioner": petitioner, "respondent": respondent}
        meta["case_name"] = f"{petitioner} vs. {respondent}"

    # Judges: "Justice XYZ" or "Hon'ble XYZ"
    judge_matches = re.findall(
        r"(?:Justice|Hon'ble|HON'BLE|JUSTICE)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})",
        text
    )
    meta["judges"] = list(dict.fromkeys(judge_matches))[:5]  # deduplicate, max 5

    # Case number
    case_num = re.search(
        r"\b(W\.?P\.?\s*\(?\s*[A-Z]\s*\)?\s*\d+[/\-]\d{4}|"
        r"C\.?A\.?\s*No\.?\s*\d+[/\-]\d{4}|"
        r"Crl\.?\s*A\.?\s*No\.?\s*\d+[/\-]\d{4}|"
        r"S\.?L\.?P\.?\s*\(?\s*[A-Z]\s*\)?\s*No\.?\s*\d+[/\-]\d{4})\b",
        text, re.IGNORECASE
    )
    if case_num:
        meta["case_number"] = case_num.group(1)

    return meta