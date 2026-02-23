<p align="center">
  <h1 align="center">🤖 Agentic RAG Chat</h1>
  <p align="center"><strong>Self-hosted AI chat platform with RAG, guardrails, and streaming</strong></p>
  <p align="center">
    <img src="https://github.com/teamauresta/agentic-rag-chat/actions/workflows/ci.yml/badge.svg" alt="CI">
    <img src="https://img.shields.io/badge/python-3.12-blue?logo=python" alt="Python">
    <img src="https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi" alt="FastAPI">
    <img src="https://img.shields.io/badge/pgvector-PostgreSQL-336791?logo=postgresql" alt="pgvector">
    <img src="https://img.shields.io/badge/Redis-7-DC382D?logo=redis" alt="Redis">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
  </p>
</p>

---

Deploy your own AI assistant in minutes. Connect any OpenAI-compatible LLM (vLLM, Ollama, OpenAI, Together, etc.), upload documents for RAG, and get a production-ready chat API with guardrails, session management, and a beautiful widget.

## ✨ Features

- 🔌 **Any LLM Backend** — Works with vLLM, Ollama, OpenAI, Together, or any OpenAI-compatible API
- 📄 **RAG Pipeline** — Upload PDFs, DOCX, CSV, TXT, MD → auto-chunked, embedded, and searchable via pgvector
- 🛡️ **3-Layer Guardrails** — Input filtering, streaming sanitisation, output validation
- ⚡ **SSE Streaming** — Real-time token streaming to the client
- 💬 **Session Management** — Redis-backed conversation history with automatic summarisation
- 🔑 **API Key Auth** — Simple bearer token authentication
- 🚦 **Rate Limiting** — Per-IP and per-session rate limits
- 📊 **Token Management** — Automatic history trimming with LLM-powered summarisation
- 📁 **File Upload API** — Upload and index documents via REST API
- 🎨 **Chat Widget** — Beautiful, configurable HTML widget (dark mode, markdown, file upload)
- 🐳 **Docker Ready** — `docker compose up` and you're running
- 🔒 **Self-Hosted** — Everything runs on your infrastructure. No data leaves your network.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Chat Widget (HTML)                     │
│              or any HTTP client / frontend                │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTPS / SSE
                       ▼
┌─────────────────────────────────────────────────────────┐
│                   Agentic RAG Chat API                     │
│                     (FastAPI + Python)                    │
│                                                           │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌─────────┐│
│  │   Auth   │  │ Guardrails│  │  Tokens  │  │  Rate   ││
│  │  (API    │  │ (3-layer) │  │ (tiktoken│  │ Limiter ││
│  │   keys)  │  │           │  │  + trim) │  │         ││
│  └──────────┘  └───────────┘  └──────────┘  └─────────┘│
│                                                           │
│  ┌──────────────────┐  ┌─────────────────────────────┐  │
│  │   RAG Engine     │  │   Session Manager (Redis)    │  │
│  │ (embed + search) │  │   (history + rate limits)    │  │
│  └────────┬─────────┘  └─────────────┬───────────────┘  │
└───────────┼───────────────────────────┼──────────────────┘
            │                           │
            ▼                           ▼
┌───────────────────┐        ┌──────────────────┐
│  PostgreSQL +     │        │     Redis        │
│  pgvector         │        │                  │
│  (embeddings)     │        │  (sessions)      │
└───────────────────┘        └──────────────────┘
            │
            │ SSE Stream
            ▼
┌───────────────────────────────────────┐
│    Any OpenAI-Compatible LLM          │
│  (vLLM, Ollama, OpenAI, Together...)  │
└───────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Clone and configure

```bash
git clone https://github.com/sotastack/agent.git
cd agent
cp .env.example .env
# Edit .env with your LLM endpoint and API key
```

### 2. Start services

```bash
docker compose up -d
```

### 3. Ingest sample documents and chat

```bash
# Ingest the sample docs
docker compose exec agent python ingest.py --path docs/ --source "Sample Docs"

# Test the API
curl http://localhost:8083/api/v1/health

# Open the widget
open widget/index.html
```

That's it. You're running a self-hosted AI assistant with RAG.

## 📖 Configuration

All configuration is via environment variables. See [`.env.example`](.env.example) for the full list.

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_URL` | `http://localhost:8000/v1` | OpenAI-compatible API endpoint |
| `LLM_API_KEY` | - | API key for your LLM backend |
| `LLM_MODEL` | `default` | Model name to use |
| `CLIENT_API_KEYS` | - | Comma-separated API keys for client auth |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `RAG_DB_HOST` | `localhost` | PostgreSQL host |
| `RAG_ENABLED` | `true` | Enable/disable RAG |
| `RAG_TOP_K` | `5` | Number of RAG results to inject |
| `RAG_MIN_SIMILARITY` | `0.3` | Minimum cosine similarity threshold |
| `MAX_TOKENS_CONTEXT` | `28000` | Max tokens in context window |
| `RATE_LIMIT_PER_MIN` | `20` | Per-IP rate limit |

## 🔌 LLM Backend Examples

**vLLM (local GPU):**
```env
LLM_URL=http://localhost:8000/v1
LLM_API_KEY=token-abc123
LLM_MODEL=meta-llama/Llama-3-8B-Instruct
```

**Ollama:**
```env
LLM_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=llama3
```

**OpenAI:**
```env
LLM_URL=https://api.openai.com/v1
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4o
```

## 📡 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/health` | Health check |
| `POST` | `/api/v1/chat` | Send a message (SSE streaming response) |
| `POST` | `/api/v1/upload` | Upload a document for RAG indexing |
| `GET` | `/api/v1/files` | List indexed documents |
| `GET` | `/api/v1/session/{id}` | Get session info |
| `DELETE` | `/api/v1/session/{id}` | Delete a session |

### Chat Request

```bash
curl -X POST http://localhost:8083/api/v1/chat \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"message": "What is in the knowledge base?"}' \
  --no-buffer
```

### Upload a Document

```bash
curl -X POST http://localhost:8083/api/v1/upload \
  -H "Authorization: Bearer your-api-key" \
  -F "file=@document.pdf" \
  -F "source=Product Manual"
```

## 🛡️ Guardrails

Agentic RAG Chat includes three layers of protection:

1. **Input Guardrails** — Blocks prompt injection, jailbreak attempts, and model probing
2. **Streaming Sanitisation** — Strips unwanted characters (e.g., CJK from English-only models) in real-time
3. **Output Validation** — Checks completed responses for system prompt leaks

Customise blocked patterns in `guardrails.py`.

## 🎨 Widget

The included chat widget (`widget/index.html`) is a single HTML file with zero dependencies. Configure it via URL parameters:

```
widget/index.html?api=http://localhost:8083/api/v1&key=your-key&title=My+Assistant
```

| Param | Description |
|-------|-------------|
| `api` | Agent API base URL |
| `key` | API key for authentication |
| `title` | Custom title in the header |
| `subtitle` | Custom subtitle |

## 🛠️ Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run in dev mode
make dev

# Ingest sample documents
make ingest

# Health check
make test
```

## 📝 Customisation

- **System Prompt**: Edit `prompts/default.txt` or add client-specific prompts as `prompts/{client}.txt`
- **Guardrails**: Modify `guardrails.py` to add/remove blocked patterns
- **RAG Settings**: Adjust `RAG_TOP_K`, `RAG_MIN_SIMILARITY` in `.env`
- **Widget**: The widget is a single HTML file — fork and customise freely

## 📦 Tech Stack

- **FastAPI** — async Python web framework
- **httpx** — async HTTP client for LLM streaming
- **Redis** — session storage and rate limiting
- **PostgreSQL + pgvector** — vector similarity search for RAG
- **sentence-transformers** — CPU-based embedding (all-MiniLM-L6-v2)
- **tiktoken** — token counting for context management

## 📄 License

MIT — see [LICENSE](LICENSE).

## 🔗 Links

- **Website**: [sotastack.com.au](https://sotastack.com.au)
- **Issues**: [GitHub Issues](https://github.com/sotastack/agent/issues)

---

<p align="center">Built by <a href="https://sotastack.com.au">SOTAStack</a> · Melbourne, Australia 🇦🇺</p>
