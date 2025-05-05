# Agentic RAG Assistant

A powerful, modular Retrieval-Augmented Generation (RAG) assistant built with Streamlit, LangChain, OpenAI, and Tavily. This app allows you to ask questions based on a default knowledge base, your uploaded documents (PDF/Excel), and optionally web search results.

---

## Features
- **Conversational RAG**: Chat interface powered by OpenAI LLMs.
- **Document Upload**: Ingest and query your own PDFs and Excel files.
- **Web Search Integration**: Uses Tavily for up-to-date answers when local knowledge is insufficient.
- **Vector Store**: Efficient document retrieval using FAISS.
- **Relevance Grading**: LLM-based filtering of retrieved documents.
- **Agentic Workflow**: Modular, stateful graph logic for robust, explainable answers.
- **Configurable**: Easily adjust models, chunking, and retrieval parameters via .env or config.

---

## Project Structure

```
agentic-rag-app claudi/
├── app.py                  # Streamlit UI and main entrypoint
├── agentic_rag_logic.py    # LangGraph agentic workflow logic
├── requirements.txt        # Python dependencies
├── .env                    # API keys and configuration
├── data/
│   ├── default_knowledge.pdf
│   └── default_knowledge_faiss/  # FAISS vector store
├── utils/
│   ├── rag_components.py   # RAG pipeline components (LLM, embeddings, vector store, retriever)
│   ├── file_processor.py   # File parsing, text extraction, chunking
│   └── config_manager.py   # Centralized config management
└── ...
```

---

## Quickstart

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd agentic-rag-app claudi
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API keys**
   - Copy `.env` and fill in your OpenAI and Tavily API keys.

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

5. **Usage**
   - Upload PDF/Excel files in the sidebar.
   - Ask questions in the chat box.
   - The assistant will use your documents, the default knowledge base, and web search as needed.

---

## Configuration
- All settings (API keys, chunk size, model, etc.) are managed via `.env` or `utils/config_manager.py`.
- See `.env` for all available options.

---

## Architecture Overview
- **Streamlit UI**: User interface, file upload, chat, and status display.
- **RAG Components**: Modular logic for LLM, embeddings, vector store, retriever, and web search.
- **Agentic Graph**: LangGraph-based workflow for retrieval, grading, web search fallback, and answer generation.
- **Config Manager**: Loads and validates all configuration and API keys.

---

## Documentation
- See the `docs/` directory for in-depth module and usage documentation.

---

## License
MIT License. See LICENSE file.

---

## Acknowledgements
- [LangChain](https://github.com/langchain-ai/langchain)
- [Streamlit](https://streamlit.io/)
- [OpenAI](https://openai.com/)
- [Tavily](https://tavily.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
