# utils/rag_components.py

from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st
import logging
from typing import List, Dict, Any, Optional, Union, Callable

# LLM and Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Vector Store
from langchain_community.vectorstores import FAISS

# Document Handling & Chains
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Pydantic for structured output
from pydantic import BaseModel, Field

# Tools
from langchain_community.tools.tavily_search import TavilySearchResults

# Retrievers
from langchain.retrievers import EnsembleRetriever

# Utilities
import pickle
import traceback
import os
import json
from pathlib import Path

# --- Load Environment Variables ---
from dotenv import load_dotenv
load_dotenv()
print("Loaded environment variables from .env file.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("rag_components")

# --- Model Initialization ---

def get_llm():
    """Initializes and returns the ChatOpenAI LLM from environment variables."""
    api_key = os.environ.get("OPENAI_API_KEY")
    model_name = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    
    if not api_key:
        logger.error("OpenAI API Key not found in environment variables.")
        return None
        
    try:
        logger.info(f"Initializing LLM with model: {model_name}")
        return ChatOpenAI(
            model=model_name, 
            temperature=0, 
            openai_api_key=api_key,
            max_retries=3,
            timeout=30
        )
    except Exception as e:
        logger.error(f"Error initializing LLM (Model: {model_name}): {str(e)}")
        logger.debug(traceback.format_exc())
        return None

def get_embedding_model():
    """Initializes and returns the OpenAI embedding model."""
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        logger.error("OpenAI API Key not found in environment variables. Cannot initialize embeddings.")
        return None
        
    try:
        logger.info("Initializing OpenAI embeddings model")
        embedding_model = OpenAIEmbeddings(
            openai_api_key=api_key,
            model="text-embedding-3-small",  # Use the latest model
            dimensions=1536,  # Standard dimensions
            max_retries=3
        )
        return embedding_model
    except Exception as e:
        logger.error(f"Error initializing OpenAI Embedding Model: {str(e)}")
        logger.debug(traceback.format_exc())
        return None

# --- Vector Store Operations ---

def create_vector_store(text_chunks: List[str], embedding_model) -> Optional[FAISS]:
    """Creates a FAISS vector store from text chunks using the provided embedding model."""
    if not text_chunks:
        logger.warning("No text chunks provided to create vector store.")
        return None
        
    if embedding_model is None:
        logger.error("Embedding model is not available, cannot create vector store.")
        return None
        
    try:
        logger.info(f"Creating vector store from {len(text_chunks)} chunks")
        
        # Convert text chunks to documents with metadata
        documents = [
            Document(page_content=chunk, metadata={"source": f"chunk_{i}", "index": i}) 
            for i, chunk in enumerate(text_chunks)
        ]
        
        # Create vector store
        vectorstore = FAISS.from_documents(documents, embedding_model)
        logger.info("Vector store created successfully")
        
        return vectorstore
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        logger.debug(traceback.format_exc())
        return None

def get_default_vector_store(knowledge_file: str, embedding_model) -> FAISS:
    """Creates a vector store from a default knowledge file."""
    from utils.file_processor import extract_text_from_files, get_text_chunks
    
    if not os.path.exists(knowledge_file):
        logger.error(f"Default knowledge file not found at '{knowledge_file}'")
        raise FileNotFoundError(f"Default knowledge file not found at '{knowledge_file}'")
    
    logger.info(f"Processing default knowledge file: {knowledge_file}")
    raw_text = extract_text_from_files([knowledge_file])
    
    if not raw_text:
        logger.error(f"No text could be extracted from the default knowledge file '{knowledge_file}'")
        raise ValueError(f"No text could be extracted from the default knowledge file '{knowledge_file}'")
    
    text_chunks = get_text_chunks(raw_text)
    
    if not text_chunks:
        logger.error("Text extracted, but could not be split into chunks")
        raise ValueError("Text extracted, but could not be split into chunks")
    
    logger.info(f"Creating vector store from {len(text_chunks)} chunks")
    
    # Convert text chunks to documents with source metadata
    documents = [
        Document(
            page_content=chunk, 
            metadata={"source": os.path.basename(knowledge_file), "index": i}
        ) 
        for i, chunk in enumerate(text_chunks)
    ]
    
    return FAISS.from_documents(documents, embedding_model)

def save_faiss_vectorstore(vectorstore, path):
    """Save a FAISS vector store to disk."""
    logger.info(f"Saving FAISS vector store to {path}")
    try:
        directory = path.replace(".pkl", "")
        os.makedirs(os.path.dirname(directory), exist_ok=True)
        vectorstore.save_local(directory)
        logger.info(f"Vector store saved successfully to {directory}")
        return True
    except Exception as e:
        logger.error(f"Error saving vector store: {str(e)}")
        logger.debug(traceback.format_exc())
        return False

def load_faiss_vectorstore(path, embedding_model):
    """Load a FAISS vector store from disk."""
    logger.info(f"Loading FAISS vector store from {path}")
    try:
        directory = path.replace(".pkl", "")
        if not os.path.exists(directory):
            logger.error(f"Vector store directory not found at {directory}")
            return None
            
        vectorstore = FAISS.load_local(
            folder_path=directory,
            embeddings=embedding_model,
            allow_dangerous_deserialization=True
        )
        logger.info(f"Vector store loaded successfully from {directory}")
        return vectorstore
    except Exception as e:
        logger.error(f"Error loading vector store: {str(e)}")
        logger.debug(traceback.format_exc())
        return None

def get_or_create_default_vector_store(knowledge_file: str, embedding_model, pickle_path: str) -> FAISS:
    """Load FAISS vector store from disk if exists, otherwise create and save it."""
    faiss_dir = pickle_path.replace(".pkl", "")
    
    if os.path.exists(faiss_dir) and os.path.isdir(faiss_dir):
        logger.info(f"Found existing vector store at {faiss_dir}, attempting to load")
        vectorstore = load_faiss_vectorstore(pickle_path, embedding_model)
        if vectorstore:
            return vectorstore
        else:
            logger.warning("Failed to load existing vector store. Creating a new one.")
    
    logger.info(f"Creating new default vector store from {knowledge_file}")
    vectorstore = get_default_vector_store(knowledge_file, embedding_model)
    
    if vectorstore:
        save_faiss_vectorstore(vectorstore, pickle_path)
    
    return vectorstore

def create_ensemble_retriever(default_vs, uploaded_vs=None):
    """Create a merged retriever from default and uploaded vector stores."""
    retrievers = []
    retriever_weights = []
    
    # Add retrievers with weights (giving higher weight to user uploaded content)
    if default_vs is not None:
        default_retriever = default_vs.as_retriever(
            search_type="similarity",
            search_kwargs={"k": int(os.environ.get("MAX_RETRIEVED_DOCS", 5))}
        )
        retrievers.append(default_retriever)
        retriever_weights.append(0.7)  # Default knowledge weight
    
    if uploaded_vs is not None:
        uploaded_retriever = uploaded_vs.as_retriever(
            search_type="similarity",
            search_kwargs={"k": int(os.environ.get("MAX_RETRIEVED_DOCS", 5))}
        )
        retrievers.append(uploaded_retriever)
        retriever_weights.append(1.0)  # User uploaded content receives higher weight
    
    if not retrievers:
        logger.warning("No retrievers available")
        return None
    
    if len(retrievers) == 1:
        logger.info(f"Using single retriever: {type(retrievers[0]).__name__}")
        return retrievers[0]
    
    logger.info(f"Creating ensemble retriever with {len(retrievers)} retrievers")
    return EnsembleRetriever(
        retrievers=retrievers,
        weights=retriever_weights
    )

# --- RAG Components ---

# 1. Relevance Grader (using Pydantic)
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Documents are relevant to the question: 'yes' or 'no'")
    reasoning: str = Field(description="Reasoning behind the relevance judgment")

def get_relevance_grader(llm):
    """Creates the relevance grading chain using the provided LLM."""
    if llm is None:
        logger.error("LLM not available, cannot create relevance grader")
        return None
    
    try:
        structured_llm_grader = llm.with_structured_output(GradeDocuments)
        
        system_prompt = """You are an expert evaluator assessing the relevance of a retrieved document to a user question.
        
        ## Instructions:
        - Base your decision SOLELY on the text provided in the document.
        - If the document contains any information that could help answer the user's question (even partially), mark it as relevant ('yes').
        - If the document is completely unrelated to the user's question, mark it as not relevant ('no').
        - In your reasoning, explain specifically what makes the document relevant or irrelevant.
        - Be generous in your assessment - if there's any chance the document could be useful, mark it as relevant.
        """ 
        
        user_prompt = """## User Question:
        {user_question}
        
        ## Document:
        {document_content}
        
        ## Relevance Evaluation:
        - Is the document relevant to the question? (yes/no)
        - Reasoning:
        """
        
        grader_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_prompt)
        ])
        
        relevance_grader = grader_prompt | structured_llm_grader
        
        return relevance_grader
    except Exception as e:
        logger.error(f"Error creating relevance grader: {str(e)}")
        logger.debug(traceback.format_exc())
        return None

# 2. RAG Chain
def get_rag_chain(llm):
    """Creates the RAG chain for answering questions based on retrieved documents."""
    if llm is None:
        logger.error("LLM not available, cannot create RAG chain")
        return None
    
    try:
        # Formatter for context from retrieved documents
        def format_docs(docs):
            return "\n\n".join(f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs))
        
        # Context + Question template
        system_template = """You are a helpful AI assistant tasked with answering questions based on the provided documents.
        
        ## Instructions:
        - Answer the user's question using ONLY the information from the provided documents.
        - If the documents don't contain enough information to fully answer the question, acknowledge this limitation.
        - If the documents contain conflicting information, point this out and explain the different perspectives.
        - When citing information, mention which document it came from (e.g., "According to Document 2...").
        - Keep your answers comprehensive but concise.
        - Structure your response in a clear, readable format.
                
        ## Documents:
        {context}
        """
        
        user_template = """## Question:
        {question}
        
        Please provide a detailed answer based on the documents provided."""
        
        # Create the prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("user", user_template)
        ])
        
        # Create the RAG chain
        rag_chain = (
            {
                "context": lambda inputs: format_docs(inputs["context"]), 
                "question": lambda inputs: inputs["question"]
            }
            | prompt 
            | llm 
            | StrOutputParser()
        )
        
        return rag_chain
    except Exception as e:
        logger.error(f"Error creating RAG chain: {str(e)}")
        logger.debug(traceback.format_exc())
        return None

# 3. Web Search Tool
def get_web_search_tool():
    """Creates a web search tool using Tavily API if the API key is available."""
    tavily_api_key = os.environ.get("TAVILY_API_KEY")
    
    if not tavily_api_key:
        logger.warning("Tavily API Key not found in environment variables. Web search disabled.")
        return None
    
    try:
        logger.info("Initializing Tavily search tool")
        search_tool = TavilySearchResults(
            tavily_api_key=tavily_api_key,
            k=3,  # Number of results to return
            max_results=5,  # Maximum number of results to retrieve
            include_raw_content=True,
            include_images=False,
            search_depth="advanced"
        )
        return search_tool
    except Exception as e:
        logger.error(f"Error initializing Tavily search tool: {str(e)}")
        logger.debug(traceback.format_exc())
        return None