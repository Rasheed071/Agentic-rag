# app.py

from dotenv import load_dotenv
load_dotenv()
# Import necessary libraries


import os
import streamlit as st
import time
from typing import Any, List, Union

# --- Import logic and components ---
from utils.file_processor import extract_text_from_files, get_text_chunks
from utils.rag_components import (
    get_llm, get_embedding_model, create_vector_store,
    get_relevance_grader, get_rag_chain, get_web_search_tool,
    get_or_create_default_vector_store, create_ensemble_retriever
)
from agentic_rag_logic import build_agentic_rag_graph, GraphState
from langchain_core.runnables import RunnableConfig

# --- Configuration ---
DEFAULT_DATA_DIR = "data"
DEFAULT_KNOWLEDGE_FILE = os.path.join(DEFAULT_DATA_DIR, "default_knowledge.pdf")
DEFAULT_VECTORSTORE_PICKLE = os.path.join(DEFAULT_DATA_DIR, "default_knowledge_faiss.pkl")

# --- Load Environment Variables ---
load_dotenv()
print("Loaded environment variables from .env file.")

# --- Page Configuration ---
st.set_page_config(
    page_title="Agentic RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4F8BF9;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4F8BF9;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .stTextInput > div > div > input {
        border-radius: 10px;
    }
    .stButton > button {
        border-radius: 10px;
        background-color: #4F8BF9;
        color: white;
        font-weight: bold;
    }
    .status-ok {
        color: #00CC66;
        font-weight: bold;
    }
    .status-error {
        color: #FF6B6B;
        font-weight: bold;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #E8F0FE;
    }
    .assistant-message {
        background-color: #F0F2F6;
    }
    .info-box {
        background-color: #F0F7FF;
        border-left: 5px solid #4F8BF9;
        padding: 1rem;
        border-radius: 5px;
    }
    .file-list {
        background-color: #F8F9FA;
        padding: 0.5rem;
        border-radius: 5px;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# --- App Title ---
st.markdown('<div class="main-header">🤖 Agentic RAG Assistant</div>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
This intelligent assistant answers questions based on:
<ul>
    <li>📄 Default knowledge base</li>
    <li>📚 Your uploaded documents (PDF/Excel)</li>
    <li>🌐 Web search (when necessary)</li>
</ul>
</div>
""", unsafe_allow_html=True)

# --- Check API Keys ---
OPENAI_API_KEY_LOADED = "sk-None-iGJamWlxXjXVKw4G9qTFT3BlbkFJctH2m3srePlvIa0vy4fs"                   # bool(os.environ.get("OPENAI_API_KEY") and "None" not in os.environ.get("OPENAI_API_KEY", ""))
TAVILY_API_KEY_LOADED = bool(os.environ.get("TAVILY_API_KEY") and "None" not in os.environ.get("TAVILY_API_KEY", ""))

with st.sidebar:
    st.markdown('<div class="sub-header">System Status</div>', unsafe_allow_html=True)
    
    openai_status = "✅ Connected" if OPENAI_API_KEY_LOADED else "❌ Not Connected"
    tavily_status = "✅ Connected" if TAVILY_API_KEY_LOADED else "❌ Not Connected (Web Search Disabled)"
    
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between;">
        <span>OpenAI API:</span>
        <span class="{'status-ok' if OPENAI_API_KEY_LOADED else 'status-error'}">{openai_status}</span>
    </div>
    <div style="display: flex; justify-content: space-between;">
        <span>Tavily API:</span>
        <span class="{'status-ok' if TAVILY_API_KEY_LOADED else 'status-error'}">{tavily_status}</span>
    </div>
    """, unsafe_allow_html=True)
    
    if not OPENAI_API_KEY_LOADED:
        st.error("⚠️ OpenAI API key is missing or invalid. Please check your .env file.")

# --- Session State Initialization ---
if "default_vector_store" not in st.session_state:
    st.session_state.default_vector_store = None
if "uploaded_vector_store" not in st.session_state:
    st.session_state.uploaded_vector_store = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "rag_graph" not in st.session_state:
    st.session_state.rag_graph = None
if "components_initialized" not in st.session_state:
    st.session_state.components_initialized = False
if "llm" not in st.session_state:
    st.session_state.llm = None
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None
if "doc_grader" not in st.session_state:
    st.session_state.doc_grader = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "web_search_tool" not in st.session_state:
    st.session_state.web_search_tool = None
if "default_data_processed" not in st.session_state:
    st.session_state.default_data_processed = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Initialization Functions ---
@st.cache_resource(show_spinner="Initializing AI components...")
def initialize_components():
    print("Initializing RAG components...")
    if not OPENAI_API_KEY_LOADED:
        st.error("Cannot initialize components: OpenAI API Key is missing or invalid in .env")
        return False

    llm = get_llm()
    embedding_model = get_embedding_model()
    doc_grader = get_relevance_grader(llm)
    rag_chain = get_rag_chain(llm)
    web_search_tool = get_web_search_tool()

    if not all([llm, embedding_model, doc_grader, rag_chain]):
        st.error("Failed to initialize one or more core RAG components. Check console logs.")
        return False

    # Always process default knowledge base on app start
    try:
        # Ensure the data directory exists
        os.makedirs(DEFAULT_DATA_DIR, exist_ok=True)
        
        # Check if default knowledge file exists
        if os.path.exists(DEFAULT_KNOWLEDGE_FILE):
            default_vector_store = get_or_create_default_vector_store(
                DEFAULT_KNOWLEDGE_FILE, embedding_model, DEFAULT_VECTORSTORE_PICKLE
            )
            st.session_state.default_vector_store = default_vector_store
            st.session_state.default_data_processed = True
            st.session_state.processed_files.add(os.path.basename(DEFAULT_KNOWLEDGE_FILE))
        else:
            print(f"Warning: Default knowledge file not found at {DEFAULT_KNOWLEDGE_FILE}")
            st.warning(f"Default knowledge file not found. System will rely on uploaded documents and web search.")
    except Exception as e:
        print(f"Error processing default knowledge base: {e}")
        st.warning(f"Could not process default knowledge base: {e}")

    rag_graph = build_agentic_rag_graph(doc_grader, rag_chain, web_search_tool)
    if not rag_graph:
        st.error("Failed to build the RAG system graph. Check console logs.")
        return False

    st.session_state.llm = llm
    st.session_state.embedding_model = embedding_model
    st.session_state.doc_grader = doc_grader
    st.session_state.rag_chain = rag_chain
    st.session_state.web_search_tool = web_search_tool
    st.session_state.rag_graph = rag_graph
    st.session_state.components_initialized = True
    return True

def process_files(files_to_process: List[Union[Any, str]], source_type: str = "Uploaded") -> bool:
    if not st.session_state.components_initialized:
        st.error("Components not initialized. Cannot process files.")
        return False
    if not files_to_process:
        st.warning(f"No {source_type.lower()} files provided for processing.")
        return False

    with st.spinner(f"Processing {len(files_to_process)} {source_type.lower()} file(s)..."):
        try:
            embedding_model = st.session_state.embedding_model
            if embedding_model is None:
                st.error("Embedding model not available. Cannot process.")
                return False

            # Extract text from files
            raw_text = extract_text_from_files(files_to_process)
            if not raw_text:
                st.warning(f"No text could be extracted from the {source_type.lower()} file(s).")
                return False
                
            # Split text into chunks
            text_chunks = get_text_chunks(raw_text)
            if not text_chunks:
                st.warning("Text extracted, but could not be split into chunks.")
                return False
                
            # Create vector store
            uploaded_vector_store = create_vector_store(text_chunks, embedding_model)
            if not uploaded_vector_store:
                st.error("Failed to create vector store from the processed text.")
                return False
                
            st.session_state.uploaded_vector_store = uploaded_vector_store
            
            # Add processed files to tracking set
            for f in files_to_process:
                if isinstance(f, str):
                    st.session_state.processed_files.add(os.path.basename(f))
                elif hasattr(f, 'name'):
                    st.session_state.processed_files.add(f.name)
                    
            return True
        except Exception as e:
            st.error(f"Error processing {source_type.lower()} files: {e}")
            return False

# --- Update retriever ---
def update_retriever():
    st.session_state.retriever = create_ensemble_retriever(
        st.session_state.default_vector_store,
        st.session_state.uploaded_vector_store
    )

# --- Initialize Components on App Load ---
components_ready = initialize_components()
if components_ready:
    update_retriever()

    
# --- File Upload Interface ---
with st.sidebar:
    st.markdown('<div class="sub-header">Document Upload</div>', unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Upload PDF or Excel files", 
        type=["pdf", "xlsx", "xls"], 
        accept_multiple_files=True,
        help="Upload documents to build or enhance the knowledge base"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Process Files", key="process_files", disabled=not components_ready or not uploaded_files):
            if process_files(uploaded_files, source_type="Uploaded"):
                update_retriever()
                st.success(f"✅ Processed {len(uploaded_files)} file(s)")
    
    with col2:
        if st.button("Clear All", key="clear_data", disabled=not st.session_state.processed_files):
            st.session_state.uploaded_vector_store = None
            update_retriever()
            # Keep default knowledge file if it exists
            default_file = os.path.basename(DEFAULT_KNOWLEDGE_FILE)
            if default_file in st.session_state.processed_files:
                st.session_state.processed_files = {default_file}
            else:
                st.session_state.processed_files = set()
            st.session_state.chat_history = []
            st.success("✅ Knowledge base and chat history cleared")
            time.sleep(1)
            st.rerun()
    
    # Display active knowledge base
    if st.session_state.processed_files:
        st.markdown('<div class="sub-header">Active Knowledge Base</div>', unsafe_allow_html=True)
        st.markdown('<div class="file-list">', unsafe_allow_html=True)
        for filename in sorted(list(st.session_state.processed_files)):
            st.markdown(f"• {filename}", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- Chat Interface ---
st.markdown('<div class="sub-header">💬 Chat</div>', unsafe_allow_html=True)

# Display chat messages
for i, message in enumerate(st.session_state.chat_history):
    role = message.get("role", "user")
    content = message.get("content", "")
    with st.chat_message(role):
        st.markdown(content)

# Accept user input
prompt = st.chat_input("Ask me anything about your documents...", disabled=not components_ready)

if prompt:
    # Add user message to chat
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Check if components are ready
    if not components_ready or not st.session_state.components_initialized:
        with st.chat_message("assistant"):
            st.error("System components are not ready. Please check API keys and logs.")
    elif st.session_state.retriever is None and not st.session_state.web_search_tool:
        with st.chat_message("assistant"):
            st.error("Neither knowledge base nor web search is available. Please check your configuration.")
    else:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            with st.spinner("Thinking..."):
                try:
                    # Format chat history for RAG
                    formatted_chat_history = []
                    for i in range(0, len(st.session_state.chat_history)-1, 2):
                        if i+1 < len(st.session_state.chat_history):
                            formatted_chat_history.append({
                                "question": st.session_state.chat_history[i]["content"],
                                "answer": st.session_state.chat_history[i+1]["content"]
                            })
                    
                    # Limit to last 5 interactions
                    recent_history = formatted_chat_history[-5:] if formatted_chat_history else []
                    
                    # Set initial state for graph
                    initial_state = GraphState(
                        question=prompt,
                        generation="",
                        documents=[],
                        web_search_needed="No",
                        chat_history=recent_history
                    )
                    
                    # Configure and invoke graph
                    config = RunnableConfig(
                                configurable={
                                "retriever": st.session_state.retriever,
                                "doc_grader": st.session_state.doc_grader,
                                "rag_chain": st.session_state.rag_chain,
                                "web_search_tool": st.session_state.web_search_tool
                            },
                            recursion_limit=10,
                            tags=["agentic-rag-v2"] if os.environ.get("LANGSMITH_API_KEY") else []
                        )
                    
                    # final_state = st.session_state.rag_graph.invoke(initial_state, config=config)
                    final_state = st.session_state.rag_graph.invoke(initial_state, config=config)
                    response = final_state.get("generation", "Sorry, I couldn't generate a response with the available information.")
                    
                    # Update placeholder with response
                    message_placeholder.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    error_msg = f"An error occurred: {str(e)}"
                    message_placeholder.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                    print(f"ERROR in RAG process: {e}")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 0.8rem;">
    Agentic RAG Assistant v2.0 | Powered by LangChain, OpenAI, and LangGraph
</div>
""", unsafe_allow_html=True)