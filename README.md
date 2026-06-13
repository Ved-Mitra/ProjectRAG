# 🤖 ProjectRAG — Portfolio Project Chatbot

> A RAG-based AI chatbot embedded in a portfolio website that lets visitors ask natural-language questions about the developer's projects and receive accurate, context-aware answers.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![Gemini](https://img.shields.io/badge/Gemini-Embedding%20%26%20LLM-orange)](https://ai.google.dev/)

---

## 📌 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Tech Stack](#3-tech-stack)
4. [RAG Strategy — Hybrid AI RAG](#4-rag-strategy--hybrid-ai-rag)
5. [Project Structure](#5-project-structure)
6. [Data Preparation](#6-data-preparation)
7. [Embedding Pipeline](#7-embedding-pipeline)
8. [Retrieval Pipeline](#8-retrieval-pipeline)
9. [Generation Pipeline](#9-generation-pipeline)
10. [Memory & Conversation History](#10-memory--conversation-history)
11. [API Layer](#11-api-layer)
12. [Frontend Widget](#12-frontend-widget)
13. [Environment Variables](#13-environment-variables)
14. [Setup & Installation](#14-setup--installation)
15. [Running the Project](#15-running-the-project)
16. [Testing Strategy](#16-testing-strategy)
17. [Deployment](#17-deployment)
18. [Future Improvements](#18-future-improvements)

---

## 1. Project Overview

**ProjectRAG** is a portfolio-integrated AI chatbot that answers questions about any project a developer has built. Instead of static project descriptions, visitors can have a real conversation like:

- *"What tech stack did you use in your last project?"*
- *"How did you handle authentication in your full-stack app?"*
- *"Which project was the hardest to build and why?"*

The chatbot uses **Retrieval-Augmented Generation (RAG)** so it only answers from grounded, factual knowledge about actual projects — it never hallucinates project details.

### Core Goals
- Accurate, grounded answers about projects (no hallucination)
- Conversational memory across a session
- Embeddable in any portfolio website as a floating chat widget
- Low latency responses (< 2s)
- Easy to update when new projects are added

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PORTFOLIO WEBSITE                                   │
│  ┌───────────────────┐                                                        │
│  │  Chat Widget (JS) │ ──── HTTP POST /chat ────►  FastAPI Backend            │
│  └───────────────────┘                              │                         │
└─────────────────────────────────────────────────────┼─────────────────────────┘
                                                       │
                              ┌────────────────────────▼──────────────────────┐
                              │               RAG PIPELINE                     │
                              │                                                 │
                              │  User Query                                     │
                              │      │                                          │
                              │      ▼                                          │
                              │  [Query Analyzer]  ◄── Adaptive RAG Router     │
                              │      │                                          │
                              │      ├──► [Dense Retriever]  (ChromaDB/Pinecone)│
                              │      ├──► [Sparse Retriever] (BM25)            │
                              │      └──► [Memory Store]     (Conversation Hist)│
                              │      │                                          │
                              │      ▼                                          │
                              │  [Re-ranker / Context Fusion]                   │
                              │      │                                          │
                              │      ▼                                          │
                              │  [Gemini LLM] ──► Final Answer                 │
                              └───────────────────────────────────────────────┘
```

---

## 3. Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **LLM** | Google Gemini 1.5 Flash / Pro | Answer generation |
| **Embeddings** | `models/text-embedding-004` (Gemini) | Semantic vector creation |
| **Vector Store** | ChromaDB (local) / Pinecone (cloud) | Dense retrieval |
| **Sparse Search** | BM25 (via `rank_bm25`) | Keyword-based retrieval |
| **Backend API** | FastAPI + Uvicorn | REST endpoint for chat |
| **Memory** | LangChain `ConversationBufferWindowMemory` | Short-term session memory |
| **Orchestration** | LangChain / LangGraph | Chain & agent orchestration |
| **Data Format** | Markdown / JSON | Project knowledge base |
| **Frontend** | React JS + CSS (floating widget) | Embeddable chat UI |
| **Deployment** | Docker + Railway / Render / Google Cloud Run | Backend hosting |

---

## 4. RAG Strategy — Hybrid AI RAG

This project uses **Hybrid AI RAG** combining three complementary retrieval strategies and adaptive routing.

### 4.1 Dense Retrieval (Semantic Search)
- Each project is embedded using **Gemini `text-embedding-004`**.
- Embeddings are stored in **ChromaDB** (local dev) or **Pinecone** (production).
- On each query, the query is also embedded and top-K semantically similar chunks are retrieved.

### 4.2 Sparse Retrieval (Keyword / BM25)
- A BM25 index is built over all project documents.
- Catches exact keyword matches that semantic search may miss (e.g., specific library names, version numbers).

### 4.3 Hybrid Fusion
- Results from dense + sparse are merged using **Reciprocal Rank Fusion (RRF)**.
- Avoids bias toward either retrieval method.

### 4.4 Adaptive RAG Router
- A lightweight classifier (or Gemini prompt) decides the retrieval strategy per query:
  - **"What projects use React?"** → Dense retrieval sufficient
  - **"What version of PyTorch did you use?"** → Sparse retrieval preferred
  - **"Tell me more about that"** → Memory-only retrieval (follow-up question)
- The router can also decide to skip retrieval entirely for meta-questions (e.g., *"How are you?"*).

### 4.5 Conversation Memory (RAG with Memory)
- Each session maintains a **sliding window conversation history** (last N turns).
- Memory is injected into the prompt as context for follow-up questions.
- Memory is stored in-process (per session), not persisted to DB (for MVP).

### 4.6 Re-ranking
- Retrieved chunks are re-ranked using a **cross-encoder** model or a Gemini-powered relevance scorer before being passed to the LLM.
- Ensures the most relevant context goes in the prompt window.

---

## 5. Project Structure

```
ProjectRAG/
│
├── data/                          # Knowledge base
│   ├── raw/                       # Raw project descriptions (Markdown files)
│   │   ├── project_1.md
│   │   ├── project_2.md
│   │   └── ...
│   └── processed/                 # Chunked & cleaned text ready for embedding
│       └── chunks.json
│
├── embeddings/                    # Embedding & vector store scripts
│   ├── embed.py                   # Embeds processed chunks via Gemini API
│   └── chroma_store/              # Local ChromaDB persistent store (gitignored)
│
├── retrieval/
│   ├── dense_retriever.py         # ChromaDB / Pinecone dense retrieval
│   ├── sparse_retriever.py        # BM25 keyword retrieval
│   ├── hybrid_retriever.py        # RRF fusion of dense + sparse results
│   └── reranker.py                # Re-ranking retrieved chunks
│
├── memory/
│   └── conversation_memory.py     # Session-based sliding window memory
│
├── adaptive_router/
│   └── router.py                  # Classifies query type and routes retrieval
│
├── pipeline/
│   └── rag_chain.py               # Main RAG chain: query → retrieve → generate
│
├── api/
│   ├── main.py                    # FastAPI app entry point
│   ├── routes/
│   │   └── chat.py                # POST /chat endpoint
│   └── schemas.py                 # Pydantic request/response models
│
├── frontend/
│   ├── widget.html                # Standalone demo page
│   ├── chatbot.js                 # Embeddable chat widget script
│   └── chatbot.css                # Widget styles
│
├── tests/
│   ├── test_retrieval.py
│   ├── test_pipeline.py
│   └── test_api.py
│
├── scripts/
│   ├── ingest.py                  # One-shot: chunk + embed + store all projects
│   └── evaluate.py                # Evaluate RAG quality (precision, faithfulness)
│
├── .env.example                   # Template for environment variables
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 6. Data Preparation

### 6.1 Writing Project Descriptions
Each project gets its own Markdown file in `data/raw/`. The file should cover:

```markdown
# Project Name

## Overview
One paragraph summary of what the project does and its purpose.

## Problem It Solves
What problem motivated this project.

## Tech Stack
- Language: Python 3.11
- Framework: FastAPI
- Database: PostgreSQL
- Deployment: Docker + Railway

## Key Features
- Feature 1
- Feature 2

## Architecture & Design Decisions
Explain why you made certain technical choices.

## Challenges & Learnings
What was hard, what you learned.

## Links
- GitHub: https://github.com/...
- Live Demo: https://...
```

### 6.2 Chunking Strategy
- Use **recursive character text splitter** with:
  - `chunk_size = 512` tokens
  - `chunk_overlap = 64` tokens
- Preserve section headers as chunk metadata for better retrieval context.
- Each chunk stores metadata: `{ project_name, section, source_file }`.

### 6.3 Ingestion Script
Run `python scripts/ingest.py` to:
1. Read all `.md` files from `data/raw/`
2. Split into chunks with metadata
3. Save processed chunks to `data/processed/chunks.json`
4. Embed chunks via Gemini API
5. Store embeddings in ChromaDB / Pinecone

---

## 7. Embedding Pipeline

### Model
- **Model**: `models/text-embedding-004` (Google Gemini)
- **Dimension**: 768
- **Task type**: `RETRIEVAL_DOCUMENT` for ingestion, `RETRIEVAL_QUERY` for queries

### Key Implementation Notes
- Batch embed chunks (max 100 per API call) to avoid rate limits.
- Store embeddings with their metadata and original text in the vector store.
- Rebuild embeddings whenever project data changes (run `ingest.py`).

```python
# embeddings/embed.py — high-level flow
import google.generativeai as genai

genai.configure(api_key=GEMINI_API_KEY)

def embed_chunks(chunks: list[dict]) -> list[list[float]]:
    texts = [c["text"] for c in chunks]
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=texts,
        task_type="RETRIEVAL_DOCUMENT"
    )
    return result["embedding"]
```

---

## 8. Retrieval Pipeline

### Dense Retrieval (`retrieval/dense_retriever.py`)
1. Embed user query with `task_type="RETRIEVAL_QUERY"`.
2. Query ChromaDB / Pinecone for top-K nearest neighbors (K=10).
3. Return chunks with similarity scores.

### Sparse Retrieval (`retrieval/sparse_retriever.py`)
1. Build BM25 index from all chunk texts at startup.
2. On query, tokenize and score all chunks.
3. Return top-K results by BM25 score.

### Hybrid Fusion (`retrieval/hybrid_retriever.py`)
```
RRF score = Σ 1 / (rank_i + k)   where k = 60
```
- Combine ranked lists from dense and sparse retriever.
- Re-sort by RRF score.
- Return final top-5 chunks as context.

---

## 9. Generation Pipeline

### Prompt Template
```
You are a helpful assistant answering questions about {developer_name}'s projects.
Use ONLY the context provided below. If the answer is not in the context, say
"I don't have information about that specific detail."

CONVERSATION HISTORY:
{history}

RETRIEVED CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
```

### LLM Settings
- **Model**: `gemini-1.5-flash` (fast) or `gemini-1.5-pro` (for complex questions)
- **Temperature**: `0.2` (factual, low creativity)
- **Max output tokens**: `512`

---

## 10. Memory & Conversation History

- **Type**: Sliding window buffer — last **6 turns** (3 user + 3 assistant).
- **Scope**: Per user session (keyed by `session_id` UUID).
- **Storage**: In-memory dictionary on the server (MVP). Can upgrade to Redis for multi-instance deployments.
- **Follow-up resolution**: Before retrieval, the Adaptive Router detects follow-up queries (e.g., "Tell me more") and rewrites them into standalone queries using conversation history + Gemini.

```python
# memory/conversation_memory.py
sessions: dict[str, list] = {}   # session_id → list of {role, content}

def get_history(session_id: str) -> list:
    return sessions.get(session_id, [])

def update_history(session_id: str, user_msg: str, bot_msg: str):
    history = sessions.setdefault(session_id, [])
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": bot_msg})
    sessions[session_id] = history[-12:]  # keep last 6 turns (12 messages)
```

---

## 11. API Layer

### Endpoint

```
POST /chat
Content-Type: application/json

{
  "session_id": "uuid-string",    // unique per browser session
  "message": "What projects use React?"
}
```

```json
// Response
{
  "answer": "You've used React in two projects: ...",
  "sources": ["project_portfolio.md", "project_ecommerce.md"]
}
```

### FastAPI Setup (`api/main.py`)
- CORS enabled for portfolio domain.
- Rate limiting: max 20 requests / minute / IP (use `slowapi`).
- Health check endpoint: `GET /health`.

---

## 12. Frontend Widget

### Embedding in Portfolio
Add this to any HTML page:
```html
<!-- At the end of <body> -->
<link rel="stylesheet" href="https://your-backend.com/static/chatbot.css" />
<script
  src="https://your-backend.com/static/chatbot.js"
  data-api-url="https://your-backend.com"
  data-developer-name="Ved Mitra"
></script>
```

### Widget Behavior
- Floating chat button (bottom-right corner).
- Opens a chat panel on click.
- Sends messages to `/chat` with a `session_id` stored in `sessionStorage`.
- Displays typing indicator while waiting for response.
- Shows source filenames below each answer (for transparency).
- Mobile responsive.

---

## 13. Environment Variables

Copy `.env.example` to `.env` and fill in:

```env
# Required
GEMINI_API_KEY=your_gemini_api_key_here
DEVELOPER_NAME=Ved Mitra

# Vector Store — choose one
VECTOR_STORE=chroma           # "chroma" (local) or "pinecone" (cloud)

# Pinecone (if using Pinecone)
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=projectrag

# API Settings
ALLOWED_ORIGINS=https://your-portfolio.com,http://localhost:3000
RATE_LIMIT=20/minute

# Memory
SESSION_WINDOW_SIZE=6
```

---

## 14. Setup & Installation

### Prerequisites
- Python 3.11+
- A Google AI Studio account for Gemini API key
- (Optional) Pinecone account for cloud vector storage

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/Ved-Mitra/ProjectRAG.git
cd ProjectRAG

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and fill in your API keys

# 5. Add your project descriptions
# Create .md files in data/raw/ following the format in Section 6.1

# 6. Ingest and embed project data
python scripts/ingest.py

# 7. Start the backend server
uvicorn api.main:app --reload --port 8000
```

### `requirements.txt` (Key Dependencies)
```
fastapi
uvicorn[standard]
python-dotenv
google-generativeai
langchain
langchain-google-genai
chromadb
rank_bm25
slowapi
pydantic
```

---

## 15. Running the Project

```bash
# Development (with hot reload)
uvicorn api.main:app --reload --port 8000

# Test the API
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test-123", "message": "What projects have you built?"}'

# Open the widget demo
open frontend/widget.html
```

---

## 16. Testing Strategy

### Unit Tests
- `test_retrieval.py` — Verify dense/sparse/hybrid retrieval returns correct chunks.
- `test_pipeline.py` — Verify end-to-end RAG chain produces grounded answers.
- `test_api.py` — Verify API request/response schemas and status codes.

### RAG Quality Evaluation (`scripts/evaluate.py`)
Use **RAGAS** metrics:
| Metric | Description | Target |
|---|---|---|
| **Faithfulness** | Answer is grounded in retrieved context | > 0.85 |
| **Answer Relevancy** | Answer is relevant to the question | > 0.80 |
| **Context Precision** | Retrieved chunks are precise | > 0.75 |
| **Context Recall** | All relevant info is retrieved | > 0.70 |

```bash
# Run evaluation
python scripts/evaluate.py --test-set tests/eval_questions.json
```

---

## 17. Deployment

### Docker

```dockerfile
# Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run
docker build -t projectrag .
docker run -p 8000:8000 --env-file .env projectrag
```

### Recommended Hosting Platforms
| Platform | Free Tier | Notes |
|---|---|---|
| **Railway** | ✅ | Easy Docker deploy, great for MVP |
| **Render** | ✅ | Free tier spins down after inactivity |
| **Google Cloud Run** | ✅ | Scales to zero, pay-per-request |
| **Fly.io** | ✅ | Low latency, persistent volumes |

> **Note**: Use **Pinecone** (instead of ChromaDB) in production so the vector store persists across container restarts.

---

## 18. Future Improvements

- [ ] **Multi-modal**: Support answering from project screenshots/diagrams using Gemini's vision capabilities.
- [ ] **Streaming responses**: Stream token-by-token output using FastAPI `StreamingResponse`.
- [ ] **Persistent memory**: Use Redis to persist conversation history across sessions.
- [ ] **Admin panel**: Web UI to add/update/delete projects without touching files.
- [ ] **Analytics**: Track which questions are asked most frequently.
- [ ] **Feedback loop**: Thumbs up/down on answers to collect fine-tuning data.
- [ ] **Self-hosted embeddings**: Explore `nomic-embed-text` as a free, local embedding model.
- [ ] **LangGraph Agent**: Replace linear RAG chain with a LangGraph agent for multi-hop reasoning.

---

## 📄 License

MIT License — see [LICENSE](./LICENSE) for details.

---

*Built by [Ved Mitra Verma](https://github.com/Ved-Mitra)*