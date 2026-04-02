# ⚖️ Nyay.AI — Legal Case Intelligence System

A local AI-powered system for Indian law firms to upload, index, search, and analyze court judgments using semantic search and LLM-powered Q&A.

---

## 🗂️ Project Structure

```
legal-case-intelligence/
├── backend/
│   ├── main.py                    # FastAPI app + all routes
│   └── services/
│       ├── pdf_parser.py          # PyMuPDF text extraction
│       ├── chunker.py             # Legal-aware text chunking
│       ├── metadata_extractor.py  # LLM/regex metadata extraction
│       ├── vector_store.py        # FAISS embeddings + search
│       └── qa_engine.py           # QA, summarization, similar cases
├── frontend/
│   └── index.html                 # Single-file UI (no build step)
├── data/
│   └── uploads/                   # Uploaded PDFs stored here
├── embeddings/                    # FAISS index + chunk store (auto-created)
├── requirements.txt
├── .env.example
├── start.sh
└── README.md
```

---

## ⚡ Quick Setup (5 minutes)

### Step 1: Clone / extract the project

```bash
cd legal-case-intelligence
```

### Step 2: Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `torch` and `sentence-transformers` are ~1-2 GB. If on a slow connection, you can skip local embeddings and use OpenAI instead (set `OPENAI_API_KEY`).

### Step 4: Configure API key

```bash
cp .env.example .env
# Edit .env and add your Anthropic or OpenAI key
```

**API key options:**

| Option | Key | Cost |
|--------|-----|------|
| **Anthropic** (recommended) | `ANTHROPIC_API_KEY` | ~$0.001/case |
| **OpenAI** | `OPENAI_API_KEY` | ~$0.002/case |
| **No key** | — | Embeddings work, no AI answers |

### Step 5: Run the server

```bash
bash start.sh
# OR manually:
cd backend && uvicorn main:app --reload --port 8000
```

### Step 6: Open the UI

```
http://localhost:8000
```

---

## 🇮🇳 Usage Guide

### Uploading Cases
1. Drag and drop a PDF (Supreme Court/High Court judgment, petition, order)
2. The system automatically:
   - Extracts text (handles messy Indian PDF formatting)
   - Splits into legal-structure-aware chunks
   - Extracts metadata (case name, court, date, judges, parties)
   - Generates embeddings and indexes into FAISS
3. You'll see the case appear in the sidebar with metadata

### Search & Q&A
- Type any natural language question in the search box
- Examples:
  - *"What were the grounds for granting bail?"*
  - *"Summarize the key arguments of the petitioner"*
  - *"What sections of IPC were cited?"*
  - *"What is the ratio decidendi?"*
- Use the **Scope** dropdown to restrict search to one case
- Results show AI answer + source excerpts with relevance scores

### Case Summarization
1. Select a case from the sidebar
2. Click the **Summarize** tab
3. Click **Generate Summary**
4. Get structured output: Facts → Issues → Arguments → Judgment → Key Principles

### Similar Case Finder
1. Select a case from the sidebar
2. Click **Similar Cases** tab
3. Click **Find Similar Cases**
4. See semantically similar cases ranked by cosine similarity score

---

## 🔧 API Reference

All endpoints available at `http://localhost:8000`:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload` | Upload PDF file |
| `GET`  | `/cases` | List all indexed cases |
| `GET`  | `/cases/{id}` | Get case metadata |
| `DELETE` | `/cases/{id}` | Delete case + embeddings |
| `POST` | `/query` | Semantic search + QA |
| `POST` | `/summarize` | Generate case summary |
| `POST` | `/similar` | Find similar cases |

### Example API calls (curl)

```bash
# Upload a PDF
curl -X POST http://localhost:8000/upload \
  -F "file=@judgment.pdf"

# Ask a question (all cases)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What was the bail condition imposed?", "top_k": 5}'

# Ask about specific case
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What was the outcome?", "case_id": "abc123"}'

# Summarize a case
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"case_id": "abc123"}'

# Find similar cases
curl -X POST http://localhost:8000/similar \
  -H "Content-Type: application/json" \
  -d '{"case_id": "abc123", "top_k": 3}'
```

---

## 🏗️ Architecture

```
PDF Upload
    │
    ▼
PyMuPDF (text extraction + cleaning)
    │
    ▼
Legal Chunker (section-aware splitting, 1200 char chunks w/ 150 char overlap)
    │
    ▼
Metadata Extractor (LLM → case name, court, date, judges, parties)
    │
    ▼
Embedder (sentence-transformers all-MiniLM-L6-v2 OR OpenAI)
    │
    ▼
FAISS Index (cosine similarity via normalized inner product)
    │
    ▼ (on query)
Semantic Search → Top-K chunks
    │
    ▼
LLM (Claude Haiku / GPT-3.5) → Structured Answer
```

---

## 🐛 Troubleshooting

**"No extractable text" error**
→ The PDF is image-based (scanned). Install `pytesseract` + `pdf2image` for OCR support (not included by default).

**Slow first upload**
→ sentence-transformers downloads the model (~80MB) on first use. Subsequent uploads are fast.

**LLM answers not working**
→ Check your API key in `.env`. Run `echo $ANTHROPIC_API_KEY` to verify it's loaded.

**FAISS installation issues**
→ Try `pip install faiss-cpu --no-cache-dir`. On Apple Silicon: `conda install -c conda-forge faiss-cpu`.

**Port already in use**
→ Change port: `uvicorn main:app --port 8001`

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| FastAPI | 0.111 | REST API framework |
| PyMuPDF | 1.24 | PDF text extraction |
| faiss-cpu | 1.8 | Vector similarity search |
| sentence-transformers | 3.0 | Local embeddings |
| torch | 2.3 | ML backend for ST |
| requests | 2.32 | LLM API calls |

---

## 🔮 Extending the System

**Add OCR for scanned PDFs:**
```bash
pip install pytesseract pdf2image
```
Then modify `pdf_parser.py` to fall back to OCR when text extraction yields < 100 chars.

**Add persistent PostgreSQL storage:**
Replace `cases.json` with SQLAlchemy + PostgreSQL for production use.

**Add Hindi/regional language support:**
Use `paraphrase-multilingual-MiniLM-L12-v2` model in `vector_store.py` for multilingual embeddings.

**Add authentication:**
Add FastAPI security middleware with JWT tokens.

---

*Built for Indian legal practitioners. Handles Supreme Court of India and High Court judgments.*
