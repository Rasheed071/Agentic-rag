# Configuration Guide

All configuration is managed via `.env`, environment variables, or `utils/config_manager.py`.

## .env Example
```
# API Keys
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...

# LLM Model
LLM_MODEL=gpt-4o-mini

# RAG Chunking
CHUNK_SIZE=500
CHUNK_OVERLAP=100
MAX_RETRIEVED_DOCS=5

# Web Search
WEB_SEARCH_MAX_RESULTS=3

# Embeddings
EMBEDDING_MODEL_NAME=text-embedding-3-small
EMBEDDING_DEVICE=cpu
EMBEDDING_PROVIDER=openai
```

## Configuration Precedence
1. `.env` file (recommended)
2. Environment variables
3. config.json (if present)
4. Streamlit session state (for runtime changes)

## Changing Settings
- Edit `.env` and restart the app
- Or set environment variables before running
- Or use the ConfigManager in code

## API Key Management
- OpenAI and Tavily keys are required for full functionality
- Keys can be set in `.env` or via Streamlit sidebar if missing

---

See [utils/config_manager.py](../utils/config_manager.py) for advanced options.