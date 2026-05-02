# HR Policy RAG Assistant

A Streamlit application that answers HR policy questions using Retrieval-Augmented Generation (RAG), with optional personalized answers from HRIS employee data.

## What The App Does

- Answers questions about HR policies (leave, remote work, overtime, security, etc.).
- Uses vector search over policy/HRIS documents before generation.
- Supports personalized answers when an `employee_id` is provided in the UI sidebar.
- Uses OpenRouter (OpenAI-compatible API) for both chat and embeddings.

## Project Structure

```text
rag_tdd/
├── app.py                      # Root Streamlit entrypoint
├── .env.example                # Environment variable template
├── requirements.txt
├── pytest.ini
├── streamlit/
│   ├── app.py                  # Alternative Streamlit entrypoint
│   ├── data/
│   │   ├── policies.json       # HR policy source data
│   │   └── hris.json           # Employee HRIS source data
│   ├── src/
│   │   ├── config.py           # OpenRouter model/base-url/env config
│   │   ├── document_loader.py  # JSON loading + Document creation
│   │   ├── vector_store.py     # FAISS + embedding + similarity search
│   │   └── rag_chain.py        # Retrieval + prompt + response generation
│   └── tests/
│       └── test_rag.py         # Unit tests
└── README.md
```

## Architecture

### 1) Ingestion

- `document_loader.py` loads:
  - `streamlit/data/policies.json`
  - `streamlit/data/hris.json`
- Converts each record into LangChain `Document` objects with metadata.

### 2) Indexing

- `vector_store.py` creates a FAISS vector index from documents.
- Embeddings are generated via OpenRouter-compatible OpenAI embeddings.
- Search includes duplicate filtering by title.

### 3) Retrieval + Generation

- `rag_chain.py` retrieves top relevant docs from FAISS.
- For personal HR questions:
  - removes unrelated HRIS retrieval leakage,
  - injects only the selected employee context,
  - can return direct deterministic PTO answer for PTO-day questions.
- Sends final context + question to chat model through OpenRouter.

### 4) UI Layer

- `app.py` provides Streamlit chat UI:
  - employee ID input (sidebar),
  - chat history,
  - response rendering.

## Data Model

### Policy record example

```json
{
  "id": "P-001",
  "title": "Remote Work Policy",
  "category": "Workplace",
  "description": "Employees can work remotely up to two days per week with manager approval.",
  "status": "active",
  "effective_year": 2026
}
```

### HRIS record example

```json
{
  "employee_id": "E110",
  "name": "James Harris",
  "pto_balance": 5,
  "pto_details": [
    { "type": "Annual Leave", "days": 5 }
  ]
}
```

## Setup

## 1) Install dependencies

From project root:

```powershell
python -m pip install -r requirements.txt
```

If your system uses `py` launcher:

```powershell
py -m pip install -r requirements.txt
```

## 2) Configure environment

Copy `.env.example` to `.env` and set your OpenRouter key:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_CHAT_MODEL=openai/gpt-4o-mini
OPENROUTER_EMBEDDING_MODEL=openai/text-embedding-3-small
```

## 3) Launch the app

From project root:

```powershell
python -m streamlit run app.py
```

Then open the local URL shown in terminal (usually `http://localhost:8501`).

## 4) Run tests

```powershell
python -m pytest -q
```

## Usage Tips

- Ask policy questions directly:
  - "What does the Overtime Policy say?"
  - "Summarize the IT Security Policy."
- For personal PTO answers:
  1. Set employee ID in sidebar (e.g. `E110`)
  2. Ask: "How many PTO days do I have?"

## Current Limitations

- Policy documents are currently indexed at record level (no advanced chunking yet).
- Retrieval is dense FAISS similarity with duplicate filtering; hybrid BM25/reranker is not yet implemented.
- Best suited for demo/small internal datasets.

## Next Improvements

- Add policy-only text chunking with overlap and richer chunk metadata.
- Add hybrid retrieval (vector + keyword) and reranking.
- Add retrieval evaluation metrics (precision@k, recall@k, MRR).
- Add persistent index versioning and incremental re-index workflow.
