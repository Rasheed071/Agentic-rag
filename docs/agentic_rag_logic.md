# Agentic Workflow (agentic_rag_logic.py)

Implements the agentic RAG workflow using LangGraph. Defines the state, nodes, and conditional logic for robust, explainable answers.

## State Definition
- `question`: User's question
- `generation`: Final answer
- `documents`: Retrieved documents
- `web_search_needed`: Flag for web search fallback
- `chat_history`: Conversation memory
- `memory`: ConversationBufferMemory (optional)
- `metadata`: Processing metrics and source info

## Node Functions
- **retrieve**: Retrieves relevant documents from the vector store
- **grade_documents**: Uses LLM to filter relevant docs
- **web_search**: Uses Tavily for web search if needed
- **generate_answer**: Generates answer using RAG chain
- **update_memory**: Updates conversation memory

## Conditional Routing
- If no relevant docs or all are irrelevant, triggers web search
- Otherwise, generates answer from local docs

## Customization
- Add new nodes or modify routing in `agentic_rag_logic.py`
- Adjust memory or metadata tracking as needed

See [utils.md](./utils.md) for RAG components.