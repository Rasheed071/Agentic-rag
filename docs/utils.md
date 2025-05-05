# RAG Components & Utils

This module contains the core logic for LLM, embeddings, vector store, retriever, file processing, and configuration.

## rag_components.py
- **get_llm**: Initializes the OpenAI LLM
- **get_embedding_model**: Loads the embedding model
- **create_vector_store**: Builds a FAISS vector store from text chunks
- **get_or_create_default_vector_store**: Loads or creates the default vector store
- **create_ensemble_retriever**: Combines default and uploaded retrievers
- **get_relevance_grader**: LLM-based document relevance grading
- **get_rag_chain**: RAG chain for answer generation
- **get_web_search_tool**: Tavily web search integration

## file_processor.py
- **extract_text_from_files**: Extracts and cleans text from PDF, Excel, CSV, and text files
- **get_text_chunks**: Splits text into chunks for embedding
- **create_document_chunks**: Prepares document chunks with metadata

## config_manager.py
- **ConfigManager**: Loads, validates, and manages all configuration and API keys
- Supports .env, environment variables, and config.json

---

See [configuration.md](./configuration.md) for all config options.