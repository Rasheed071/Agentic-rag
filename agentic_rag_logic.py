# agentic_rag_logic.py

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from typing import List, Dict, Optional, TypedDict, Union, Any, Tuple
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langgraph.graph import END, StateGraph
import os
import logging
import time

# --- Load Environment Variables ---
from dotenv import load_dotenv
load_dotenv()
print("Attempted to load environment variables from .env file.")

# Configure logging
logger = logging.getLogger("agentic_rag")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

# --- LangGraph State Definition ---

class GraphState(TypedDict):
    """Represents the state passed between nodes in the LangGraph."""
    question: str
    generation: str
    documents: List[Document]
    web_search_needed: str
    chat_history: List[Dict[str, str]]  # For conversational memory
    memory: Optional[Any]  # Conversational buffer memory
    metadata: Dict[str, Any]  # Track processing metrics and source info

# --- LangGraph Node Functions ---

def retrieve(state: GraphState, config: RunnableConfig) -> GraphState:
    """Retrieves relevant documents from the vector store based on the question."""
    logger.info("---NODE: RETRIEVE---")
    start_time = time.time()
    question = state["question"]
    retriever = config['configurable'].get('retriever')
    metadata = state.get("metadata", {})
    metadata["retrieval_source"] = "none"
    
    if retriever is None:
        logger.warning("---RETRIEVE: Failed - Retriever is None in config.---")
        st.warning("Retriever not configured. Cannot search documents.")
        metadata["retrieval_status"] = "failed"
        metadata["retrieval_reason"] = "retriever_missing"
        return {**state, "documents": [], "metadata": metadata}

    try:
        logger.info(f"---RETRIEVE: Searching documents for: '{question}'---")
        # Use invoke method as per latest LangChain API
        retrieved_docs = retriever.invoke(question)
        
        # Set metadata for documents
        for doc in retrieved_docs:
            if 'source' not in doc.metadata:
                doc.metadata['source'] = 'Knowledge Base'
        
        logger.info(f"---RETRIEVE: Found {len(retrieved_docs)} documents.---")
        
        # Update metadata
        metadata["retrieval_status"] = "success"
        metadata["retrieval_count"] = len(retrieved_docs)
        metadata["retrieval_time"] = f"{time.time() - start_time:.2f}s"
        if retrieved_docs:
            metadata["retrieval_source"] = "knowledge_base"
            # Get sources for logging
            sources = list(set([doc.metadata.get('source', 'Unknown') for doc in retrieved_docs]))
            metadata["retrieval_sources"] = sources
        
        return {**state, "documents": retrieved_docs, "metadata": metadata}
    except Exception as e:
        logger.error(f"---RETRIEVE: Failed - An unexpected error occurred: {str(e)}---")
        st.error(f"Error during document retrieval: {str(e)}")
        metadata["retrieval_status"] = "failed"
        metadata["retrieval_reason"] = str(e)
        return {**state, "documents": [], "metadata": metadata}

def grade_documents(state: GraphState, config: RunnableConfig) -> GraphState:
    """Grades the relevance of retrieved documents using the LLM grader."""
    logger.info("---NODE: GRADE DOCUMENTS---")
    start_time = time.time()
    question = state["question"]
    documents = state["documents"]
    web_search_needed = "No"  # Default assumption
    metadata = state.get("metadata", {})
    doc_grader = config['configurable'].get('doc_grader')
    
    if doc_grader is None:
        logger.warning("---GRADE: Failed - Grader is None. Triggering web search.---")
        st.warning("Document grader not available. Assuming web search is needed.")
        metadata["grading_status"] = "failed"
        metadata["grading_reason"] = "grader_missing"
        return {**state, "web_search_needed": "Yes", "documents": [], "metadata": metadata}

    if not documents:
        logger.warning("---GRADE: No documents to grade. Triggering web search.---")
        web_search_needed = "Yes"
        filtered_docs = []
        metadata["grading_status"] = "skipped"
        metadata["grading_reason"] = "no_documents"
    else:
        logger.info(f"---GRADE: Grading {len(documents)} retrieved documents...---")
        filtered_docs = []
        all_irrelevant = True
        graded_count = 0
        relevant_count = 0
        
        for i, doc in enumerate(documents):
            try:
                # Ensure page_content is not empty or None before grading
                doc_content = getattr(doc, 'page_content', '')
                if not doc_content:
                    logger.warning(f"---GRADE: Document {i+1} has empty content. Skipping grade.---")
                    continue  # Skip grading empty documents

                # Invoke grader with document content and question
                score = doc_grader.invoke({
                    "user_question": question, 
                    "document_content": doc_content
                })
                
                graded_count += 1
                
                # Check if the document is relevant
                grade = score.binary_score.lower() if hasattr(score, 'binary_score') else 'no'
                reasoning = score.reasoning if hasattr(score, 'reasoning') else 'No reasoning provided'
                
                # Add grading info to document metadata
                doc.metadata['relevance_grade'] = grade
                doc.metadata['relevance_reasoning'] = reasoning

                if grade == "yes":
                    logger.info(f"---GRADE: Document {i+1} RELEVANT---")
                    filtered_docs.append(doc)  # Keep relevant document
                    all_irrelevant = False
                    relevant_count += 1
                else:
                    logger.info(f"---GRADE: Document {i+1} NOT RELEVANT---")
            except Exception as e:
                logger.error(f"---GRADE: Failed for doc {i+1} - {str(e)}---")
                st.warning(f"Error grading document {i+1}. Treating as NOT RELEVANT.")

        # Update metadata
        metadata["grading_status"] = "success"
        metadata["grading_time"] = f"{time.time() - start_time:.2f}s"
        metadata["grading_total"] = graded_count
        metadata["grading_relevant"] = relevant_count
        
        if all_irrelevant and documents:
            logger.warning("---GRADE: All retrieved documents were graded as irrelevant. Triggering web search.---")
            web_search_needed = "Yes"
            filtered_docs = []  # Discard all if web search needed
            metadata["grading_result"] = "all_irrelevant"
        elif filtered_docs:
            metadata["grading_result"] = "found_relevant"
        else:
            metadata["grading_result"] = "none_relevant"

    logger.info(f"---GRADE: Completed. Relevant docs kept: {len(filtered_docs)}. Web search needed: {web_search_needed}---")
    return {**state, "documents": filtered_docs, "web_search_needed": web_search_needed, "metadata": metadata}

def web_search(state: GraphState, config: RunnableConfig) -> GraphState:
    """Performs a web search using the Tavily tool if triggered."""
    logger.info("---NODE: WEB SEARCH---")
    start_time = time.time()
    question = state["question"]
    metadata = state.get("metadata", {})
    web_search_tool = config['configurable'].get('web_search_tool')

    if web_search_tool is None:
        logger.warning("---WEB SEARCH: Failed - Tool is None---")
        st.warning("Web search tool not available (check Tavily API key). Cannot perform web search.")
        metadata["web_search_status"] = "failed"
        metadata["web_search_reason"] = "tool_missing"
        return {**state, "documents": [], "metadata": metadata}

    logger.info(f"---WEB SEARCH: Performing search for: '{question}'---")
    try:
        search_results = web_search_tool.invoke({"query": question})
        web_docs = []
        
        if isinstance(search_results, list):
            logger.info(f"---WEB SEARCH: Received {len(search_results)} results.---")
            for i, result in enumerate(search_results):
                content = result.get('content', '')
                url = result.get('url', f'Web Result {i+1}')
                title = result.get('title', 'Untitled')
                if content:  # Only add if there's content
                    page_content = f"Source: {url}\nTitle: {title}\n\n{content}"
                    web_docs.append(Document(
                        page_content=page_content, 
                        metadata={"source": url, "title": title, "origin": "web_search"}
                    ))
        elif isinstance(search_results, str) and search_results:
            logger.info(f"---WEB SEARCH: Received single string result.---")
            web_docs.append(Document(
                page_content=search_results, 
                metadata={"source": "web_search", "origin": "web_search"}
            ))
        else:
            logger.warning(f"---WEB SEARCH: No results or unexpected format received.---")

        logger.info(f"---WEB SEARCH: Prepared {len(web_docs)} documents from web search.---")
        
        # Update metadata
        metadata["web_search_status"] = "success" if web_docs else "no_results"
        metadata["web_search_time"] = f"{time.time() - start_time:.2f}s"
        metadata["web_search_count"] = len(web_docs)
        if web_docs:
            metadata["retrieval_source"] = "web_search"
        
        # These web documents replace any previously held documents
        return {**state, "documents": web_docs, "metadata": metadata}

    except Exception as e:
        logger.error(f"---WEB SEARCH: Failed - {str(e)}---")
        st.error(f"Error during web search: {str(e)}")
        metadata["web_search_status"] = "failed"
        metadata["web_search_reason"] = str(e)
        return {**state, "documents": [], "metadata": metadata}

def generate_answer(state: GraphState, config: RunnableConfig) -> GraphState:
    """Generates the final answer using the RAG chain based on the available documents."""
    logger.info("---NODE: GENERATE ANSWER---")
    start_time = time.time()
    question = state["question"]
    documents = state["documents"]
    chat_history = state.get("chat_history", [])
    memory = state.get("memory")
    metadata = state.get("metadata", {})
    rag_chain = config['configurable'].get('rag_chain')

    if rag_chain is None:
        logger.error("---GENERATE: Failed - RAG Chain is None---")
        st.error("RAG chain not available. Cannot generate answer.")
        metadata["generation_status"] = "failed"
        metadata["generation_reason"] = "chain_missing"
        return {**state, "generation": "Error: Answer generation component failed.", "metadata": metadata}

    try:
        # Format chat history for the RAG chain
        formatted_history = []
        if chat_history:
            for interaction in chat_history:
                if isinstance(interaction, dict) and "question" in interaction and "answer" in interaction:
                    formatted_history.append(f"User: {interaction['question']}\nAssistant: {interaction['answer']}")
        
        history_context = "\n\n".join(formatted_history)
        
        # Prepare documents for generation
        if not documents:
            logger.warning("---GENERATE: No relevant documents found. Generating 'don't know' response.---")
            generation = "I couldn't find relevant information in the available documents or via web search to answer your question. Could you please rephrase or ask a different question?"
            metadata["generation_status"] = "no_documents"
        else:
            logger.info(f"---GENERATE: Generating answer based on {len(documents)} documents and chat history.---")
            # Pass context to the RAG chain
            generation = rag_chain.invoke({
                "context": documents, 
                "question": question,
                "chat_history": formatted_history
            })
            metadata["generation_status"] = "success"
            logger.info(f"---GENERATE: Successfully generated answer.---")
            
            # Update memory if available
            if memory is not None:
                try:
                    memory.save_context(
                        {"input": question},
                        {"output": generation}
                    )
                    logger.info("---MEMORY: Updated conversation memory.---")
                except Exception as e:
                    logger.error(f"---MEMORY: Failed to update - {str(e)}---")
        
        # Update metadata
        metadata["generation_time"] = f"{time.time() - start_time:.2f}s"
        sources = []
        if documents:
            sources = list(set([doc.metadata.get('source', 'Unknown') for doc in documents]))
        metadata["answer_sources"] = sources
        
        return {**state, "generation": generation, "metadata": metadata}
    
    except Exception as e:
        logger.error(f"---GENERATE: Failed - {str(e)}---")
        st.error(f"Error during answer generation: {str(e)}")
        metadata["generation_status"] = "failed"
        metadata["generation_reason"] = str(e)
        generation = "Sorry, an error occurred while generating the answer. Please try again or rephrase your question."
        return {**state, "generation": generation, "metadata": metadata}

def update_memory(state: GraphState, config: RunnableConfig) -> GraphState:
    """Updates the conversation memory with the latest interaction."""
    logger.info("---NODE: UPDATE MEMORY---")
    memory = state.get("memory")
    question = state["question"]
    generation = state["generation"]
    
    if memory is None:
        # Create new memory if not already present
        logger.info("---MEMORY: Initializing new conversation memory---")
        try:
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="output",
                input_key="input",
                k=5  # Keep the last 5 conversation turns
            )
        except Exception as e:
            logger.error(f"---MEMORY: Failed to initialize - {str(e)}---")
            return state
    
    try:
        # Save the current interaction
        memory.save_context(
            {"input": question},
            {"output": generation}
        )
        logger.info("---MEMORY: Updated conversation memory successfully---")
        return {**state, "memory": memory}
    except Exception as e:
        logger.error(f"---MEMORY: Failed to update - {str(e)}---")
        return state

# --- Conditional Edge Logic ---

def decide_to_generate(state: GraphState) -> str:
    """Determines the next step based on the 'web_search_needed' flag."""
    logger.info("---EDGE: DECIDE TO GENERATE---")
    web_search_needed = state.get("web_search_needed", "No")
    documents = state.get("documents", [])
    metadata = state.get("metadata", {})
    
    # Check if web search is explicitly needed
    if web_search_needed == "Yes":
        logger.info("---DECISION: Web search is needed. Routing to 'web_search'.---")
        metadata["routing_decision"] = "web_search_triggered"
        state["metadata"] = metadata
        return "web_search"
    
    # Check if we have relevant documents
    if documents:
        logger.info("---DECISION: Relevant documents found locally. Routing to 'generate_answer'.---")
        metadata["routing_decision"] = "documents_available"
        state["metadata"] = metadata
        return "generate_answer"
    else:
        # No relevant docs and no explicit web search request - try web search as fallback
        logger.warning("---DECISION: No relevant local documents. Routing to 'web_search' as fallback.---")
        metadata["routing_decision"] = "no_documents_fallback"
        state["metadata"] = metadata
        return "web_search"

# --- Initialize Memory ---

def get_conversation_memory(k: int = 5) -> ConversationBufferMemory:
    """Initialize a ConversationBufferMemory to store chat history."""
    try:
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output",
            input_key="input",
            k=k  # Number of conversation turns to keep
        )
        return memory
    except Exception as e:
        logger.error(f"Failed to initialize conversation memory: {str(e)}")
        return None

# --- Build the Graph ---

def build_agentic_rag_graph(doc_grader=None, rag_chain=None, web_search_tool=None):
    """Builds and compiles the LangGraph StateGraph for the Agentic RAG system.
    
    Args:
        doc_grader: The document relevance grader component
        rag_chain: The RAG chain for generating answers
        web_search_tool: The web search tool (optional)
        
    Returns:
        The compiled StateGraph for the Agentic RAG system
    """
    try:
        # Initialize the state graph
        agentic_rag = StateGraph(GraphState)

        # Add nodes
        agentic_rag.add_node("retrieve", retrieve)  # Retriever passed in config
        agentic_rag.add_node("grade_documents", grade_documents)
        agentic_rag.add_node("web_search", web_search)
        agentic_rag.add_node("generate_answer", generate_answer)
        agentic_rag.add_node("update_memory", update_memory)

        # Define graph flow
        agentic_rag.set_entry_point("retrieve")
        agentic_rag.add_edge("retrieve", "grade_documents")
        
        # Conditional routing based on document relevance and web search need
        agentic_rag.add_conditional_edges(
            "grade_documents",
            decide_to_generate,
            {
                "web_search": "web_search",
                "generate_answer": "generate_answer",
            },
        )
        
        # Final edges
        agentic_rag.add_edge("web_search", "generate_answer")
        agentic_rag.add_edge("generate_answer", "update_memory")
        agentic_rag.add_edge("update_memory", END)

        # Compile the graph
        logger.info("Compiling Agentic RAG Graph...")
        compiled_graph = agentic_rag.compile()
        logger.info("Agentic RAG Graph Compiled Successfully")
        return compiled_graph
        
    except Exception as e:
        logger.error(f"ERROR: Error building LangGraph: {str(e)}")
        st.error(f"Error building LangGraph: {str(e)}")
        return None