"""
QA Engine: question answering, case summarization, similar-case finder.
Uses LLM over retrieved chunks.

Upgrades:
  - Structured context formatting (CASE / COURT / SECTION blocks)
  - Improved system prompt: Issue → Analysis → Conclusion structure
  - Citations returned alongside answer
  - LLM token budget capped to avoid oversized prompts
"""

import os
import re
import json
import requests
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

# ── constants ──────────────────────────────────────────────────────────────────

# Maximum characters of chunk text sent to LLM per chunk (avoids bloated prompts)
MAX_CHUNK_CHARS_FOR_LLM = 600

# Maximum total context characters sent to LLM
MAX_CONTEXT_CHARS = 6000


# ── LLM call ──────────────────────────────────────────────────────────────────

def _call_llm(system: str, user: str, max_tokens: int = 800) -> str:
    """Call available LLM API. Priority: Anthropic -> OpenAI -> Gemini."""
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
                "max_tokens": max_tokens,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["content"][0]["text"]

    elif openai_key:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"},
            json={
                "model": "gpt-3.5-turbo",
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    elif gemini_key:
        import google.generativeai as genai
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=system,
        )
        response = model.generate_content(
            user,
            generation_config=genai.GenerationConfig(max_output_tokens=max_tokens),
        )
        return response.text

    else:
        return (
            "No LLM API key configured. "
            "Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY to enable AI answers.\n\n"
            "**Relevant excerpts found:**\n\n"
        )


# ── QA ─────────────────────────────────────────────────────────────────────────

QA_SYSTEM = """You are an expert Indian legal assistant with deep knowledge of Indian constitutional law, civil and criminal procedure, and landmark Supreme Court and High Court judgments.

When answering questions, always structure your response under these three headings:

**Issue**
State the precise legal question raised.

**Analysis**
- Cite the legal reasoning drawn from the provided excerpts.
- Reference relevant sections of Indian law (IPC, CPC, CrPC, Constitution, etc.) where applicable.
- Mention court hierarchy where relevant (Supreme Court > High Courts > District Courts).
- Quote or paraphrase specific passages from the excerpts with attribution (e.g., "As held in [Case Name]...").

**Conclusion**
Provide a clear, direct answer to the question.

If uncertain about any point, explicitly say so. Use formal but accessible language."""


def answer_question(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Generate a legally-informed answer with source citations.

    Returns:
        {
            "answer":    str,           # structured LLM response
            "citations": List[dict]     # source references used
        }
    """
    context, citations = _format_context_and_citations(retrieved_chunks)

    user_prompt = f"""Using the excerpts below from Indian court documents, answer the question.

{context}

QUESTION: {question}

Structure your answer under: **Issue**, **Analysis**, **Conclusion**."""

    try:
        answer = _call_llm(QA_SYSTEM, user_prompt, max_tokens=900)
    except Exception as e:
        # Graceful degradation — surface raw excerpts if LLM is down
        answer = f"LLM unavailable ({e}). Here are the most relevant excerpts:\n\n"
        for i, c in enumerate(retrieved_chunks[:3], 1):
            answer += (
                f"**Excerpt {i}** (Section: {c.get('section','?')}, Score: {c.get('score',0):.2f})\n"
                + c["text"][:400] + "...\n\n"
            )

    return {"answer": answer, "citations": citations}


# ── Summarization ──────────────────────────────────────────────────────────────

SUMMARY_SYSTEM = """You are an expert Indian legal analyst. Generate comprehensive, structured case summaries that are useful to practising lawyers."""

SUMMARY_PROMPT = """Summarize the following Indian court case based on these document excerpts.
Structure the summary under these exact headings (use markdown):

## Case Overview
(Case name, court, date, bench)

## Facts
(Key facts in numbered list)

## Legal Issues
(Issues framed by the court)

## Arguments
### Petitioner/Appellant
### Respondent

## Judgment & Ratio Decidendi
(The court's decision and legal reasoning)

## Key Legal Principles
(Principles established or affirmed)

## Practical Significance
(Why this case matters for practitioners)

---
CASE METADATA:
{metadata}

DOCUMENT EXCERPTS:
{context}

Provide a comprehensive summary suitable for inclusion in a legal brief."""


def summarize_case(chunks: List[Dict[str, Any]], metadata: Dict[str, Any]) -> str:
    """Generate structured case summary using top chunks."""
    context, _ = _format_context_and_citations(chunks[:8])
    meta_str    = json.dumps(metadata, indent=2, ensure_ascii=False)
    user_msg    = SUMMARY_PROMPT.format(metadata=meta_str, context=context)

    try:
        return _call_llm(SUMMARY_SYSTEM, user_msg, max_tokens=1200)
    except Exception as e:
        return f"Summary generation failed: {e}\n\nMetadata: {meta_str}"


# ── Similar cases ──────────────────────────────────────────────────────────────

def find_similar_cases(
    target_case_id: str,
    vector_store,
    all_cases: Dict[str, Any],
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """Find cases most similar to target using mean embedding cosine similarity."""
    target_vec = vector_store.get_case_embedding(target_case_id)
    if target_vec is None:
        return []

    scores = []
    for case_id, case_data in all_cases.items():
        if case_id == target_case_id:
            continue
        vec = vector_store.get_case_embedding(case_id)
        if vec is None:
            continue
        similarity = float(np.dot(target_vec, vec))
        scores.append((similarity, case_id, case_data))

    scores.sort(reverse=True)

    return [
        {
            "case_id":        cid,
            "filename":       cdata.get("filename"),
            "similarity":     round(sim, 4),
            "similarity_pct": f"{sim * 100:.1f}%",
            "metadata":       cdata.get("metadata", {}),
        }
        for sim, cid, cdata in scores[:top_k]
    ]


# ── helpers ────────────────────────────────────────────────────────────────────

def _format_context_and_citations(
    chunks: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Build a structured context string for the LLM and a citations list for the caller.

    Context block format per chunk:
        -- [N] ------------------------------------------
        CASE:    <case_name>
        COURT:   <court>
        SECTION: <section>

        <chunk text (capped at MAX_CHUNK_CHARS_FOR_LLM)>

    Citations list entry:
        {"case_name": ..., "court": ..., "section": ..., "text": <300-char preview>}
    """
    parts: List[str] = []
    citations: List[Dict[str, Any]] = []
    total_chars = 0

    for i, chunk in enumerate(chunks, 1):
        meta      = chunk.get("metadata", {})
        case_name = meta.get("case_name") or chunk.get("case_id", "Unknown Case")
        court     = meta.get("court", "")
        section   = chunk.get("section", "")
        score     = chunk.get("score")
        score_str = f" | Score: {score:.2f}" if score is not None else ""

        # Cap individual chunk length to control total prompt size
        text = chunk["text"][:MAX_CHUNK_CHARS_FOR_LLM]
        if len(chunk["text"]) > MAX_CHUNK_CHARS_FOR_LLM:
            text += "..."

        block = (
            f"-- [{i}]{score_str} ------------------------------------------\n"
            f"CASE:    {case_name}\n"
            f"COURT:   {court}\n"
            f"SECTION: {section}\n\n"
            f"{text}"
        )

        # Stop adding chunks once total context budget is exhausted
        if total_chars + len(block) > MAX_CONTEXT_CHARS:
            break

        parts.append(block)
        total_chars += len(block)

        citations.append({
            "case_name": case_name,
            "court":     court,
            "section":   section,
            "text":      chunk["text"][:300],   # short preview for frontend display
        })

    return "\n\n".join(parts), citations