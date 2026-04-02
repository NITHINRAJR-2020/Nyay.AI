"""
QA Engine: question answering, case summarization, similar-case finder.
Uses LLM over retrieved chunks.
"""

import os
import re
import json
import requests
import numpy as np
from typing import List, Dict, Any, Optional

# ── LLM call ──────────────────────────────────────────────────────────────────

def _call_llm(system: str, user: str, max_tokens: int = 800) -> str:
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
        # Gemini doesn't have a separate system role — prepend it to the user prompt
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
        # No LLM — return a descriptive fallback
        return (
            "⚠️ No LLM API key configured. "
            "Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY to enable AI answers.\n\n"
            "**Relevant excerpts found:**\n\n"
        )


# ── QA ─────────────────────────────────────────────────────────────────────────

QA_SYSTEM = """You are an expert Indian legal assistant with deep knowledge of Indian constitutional law, civil and criminal procedure, and landmark Supreme Court and High Court judgments.

When answering questions:
- Be precise and cite specific legal principles
- Reference court hierarchy (Supreme Court > High Courts > District Courts)
- Mention relevant sections of Indian laws (IPC, CPC, CrPC, Constitution, etc.) when applicable
- Use formal but accessible language
- If uncertain, clearly state so
- Structure longer answers with clear headings"""

def answer_question(question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    """Generate a legally-informed answer using retrieved chunks as context."""
    context = _format_chunks_as_context(retrieved_chunks)

    user_prompt = f"""Based on the following excerpts from Indian court documents, answer the question below.

RETRIEVED EXCERPTS:
{context}

QUESTION: {question}

Provide a structured, legally precise answer. If the excerpts directly address the question, quote or paraphrase the relevant parts. If asking about similar cases or arguments, synthesize patterns across the excerpts."""

    try:
        return _call_llm(QA_SYSTEM, user_prompt, max_tokens=900)
    except Exception as e:
        # Graceful degradation: return raw chunks
        fallback = f"⚠️ LLM unavailable ({e}). Here are the most relevant excerpts:\n\n"
        for i, c in enumerate(retrieved_chunks[:3], 1):
            fallback += f"**Excerpt {i}** (Section: {c.get('section','?')}, Score: {c['score']:.2f})\n"
            fallback += c["text"][:400] + "...\n\n"
        return fallback


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
    """Generate structured case summary."""
    # Use top chunks (first 8 most representative)
    context   = _format_chunks_as_context(chunks[:8])
    meta_str  = json.dumps(metadata, indent=2, ensure_ascii=False)
    user_msg  = SUMMARY_PROMPT.format(metadata=meta_str, context=context)

    try:
        return _call_llm(SUMMARY_SYSTEM, user_msg, max_tokens=1200)
    except Exception as e:
        return f"⚠️ Summary generation failed: {e}\n\nMetadata: {meta_str}"


# ── Similar cases ──────────────────────────────────────────────────────────────

def find_similar_cases(
    target_case_id: str,
    vector_store,
    all_cases: Dict[str, Any],
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """
    Find cases most similar to target_case_id using mean embedding cosine similarity.
    """
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
        similarity = float(np.dot(target_vec, vec))   # both normalized → cosine sim
        scores.append((similarity, case_id, case_data))

    scores.sort(reverse=True)

    results = []
    for sim, cid, cdata in scores[:top_k]:
        results.append({
            "case_id":        cid,
            "filename":       cdata.get("filename"),
            "similarity":     round(sim, 4),
            "similarity_pct": f"{sim * 100:.1f}%",
            "metadata":       cdata.get("metadata", {}),
        })

    return results


# ── helpers ────────────────────────────────────────────────────────────────────

def _format_chunks_as_context(chunks: List[Dict[str, Any]]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        case_name   = meta.get("case_name", "Unknown Case")
        court       = meta.get("court", "")
        section     = chunk.get("section", "")
        score       = chunk.get("score")
        score_str   = f" | Relevance: {score:.2f}" if score is not None else ""

        header = f"[{i}] {case_name} | {court} | {section}{score_str}"
        parts.append(f"{header}\n{chunk['text']}")

    return "\n\n---\n\n".join(parts)