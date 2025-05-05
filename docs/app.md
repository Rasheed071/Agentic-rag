# Streamlit App (app.py)

The main entrypoint for the Agentic RAG Assistant. Provides the user interface, file upload, chat, and system status.

## Key Responsibilities
- Loads and validates API keys from .env
- Initializes LLM, embeddings, vector store, and agentic graph
- Handles file upload and document processing
- Maintains chat history and session state
- Orchestrates the RAG workflow and displays answers

## Main Flow
1. On startup, loads environment and initializes all components
2. Processes the default knowledge base and any uploaded files
3. Updates the retriever (ensemble of default and uploaded docs)
4. Handles user chat input and displays responses
5. Uses LangGraph agentic workflow for retrieval, grading, web search, and answer generation

## Customization
- Modify UI, chat formatting, or sidebar in `app.py`
- Adjust session state or Streamlit caching as needed

See [agentic_rag_logic.md](./agentic_rag_logic.md) for workflow details.