# Agentic RAG Assistant Documentation

Welcome to the in-depth documentation for the Agentic RAG Assistant. This guide covers architecture, configuration, and module details.

## Contents
- [Quickstart](./index.md#quickstart)
- [Architecture](./index.md#architecture)
- [Streamlit App](./app.md)
- [Agentic Workflow](./agentic_rag_logic.md)
- [RAG Components & Utils](./utils.md)
- [Configuration](./configuration.md)

---

## Quickstart
See the [README.md](../README.md) for setup and usage instructions.

---

## Architecture

- **Streamlit UI**: Handles user interaction, file upload, chat, and status.
- **RAG Components**: Modular logic for LLM, embeddings, vector store, retriever, and web search.
- **Agentic Graph**: LangGraph-based workflow for retrieval, grading, web search fallback, and answer generation.
- **Config Manager**: Loads and validates all configuration and API keys.

---

See module docs for more details.