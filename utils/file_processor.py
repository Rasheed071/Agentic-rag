# utils/file_processor.py

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from typing import List, Any, Union, Dict, Optional
import os
import io
import re
import logging

# --- Load Environment Variables ---
from dotenv import load_dotenv
load_dotenv()
print("Attempted to load environment variables from .env file.")

# Configure logging
logger = logging.getLogger("file_processor")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

# --- Text Extraction Functions ---

def extract_text_from_pdf(pdf_file: Any) -> str:
    """
    Extracts text from a single uploaded PDF file stream.
    
    Args:
        pdf_file: A file-like object containing PDF data
        
    Returns:
        Extracted text as a string
    """
    text = ""
    try:
        # Reset stream position
        pdf_file.seek(0)
        
        # Create PDF reader
        pdf_reader = PdfReader(pdf_file)
        total_pages = len(pdf_reader.pages)
        logger.info(f"Processing PDF with {total_pages} pages")
        
        # Extract text from each page
        for i, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"  # Add double newline between pages
            except Exception as e:
                logger.warning(f"Error extracting text from page {i+1}: {e}")
                
        if not text.strip():
            logger.warning("PDF appears to be empty or contains only scanned images")
            
    except Exception as e:
        file_name = getattr(pdf_file, 'name', 'Unknown')
        logger.error(f"Could not read PDF file {file_name}: {e}")
        st.warning(f"Could not read PDF file {file_name}: {e}")
        
    return text

def extract_text_from_excel(excel_file: Any) -> str:
    """
    Extracts text from a single uploaded Excel file stream.
    
    Args:
        excel_file: A file-like object containing Excel data
        
    Returns:
        Extracted text as a string
    """
    all_text = ""
    try:
        # Reset stream position
        excel_file.seek(0)
        
        # Read all sheets into a dictionary of DataFrames
        excel_data = pd.read_excel(excel_file, sheet_name=None, engine='openpyxl')
        logger.info(f"Processing Excel file with {len(excel_data)} sheets")
        
        for sheet_name, df in excel_data.items():
            # Handle empty sheets
            if df.empty:
                all_text += f"--- Sheet: {sheet_name} ---\n(Empty Sheet)\n\n"
                continue
                
            # Clean the dataframe - replace NaNs with empty strings
            df = df.fillna('')
            
            # Get column headers as a formatted string
            headers = " | ".join(str(col) for col in df.columns)
            
            # Start with sheet name and headers
            sheet_text = f"--- Sheet: {sheet_name} ---\n{headers}\n"
            
            # Convert data rows to formatted strings
            for _, row in df.iterrows():
                # Join row values with separator
                row_text = " | ".join(str(val) for val in row.values)
                sheet_text += f"{row_text}\n"
                
            all_text += sheet_text + "\n\n"
            
    except Exception as e:
        file_name = getattr(excel_file, 'name', 'Unknown')
        logger.error(f"Could not read Excel file {file_name}: {e}")
        st.warning(f"Could not read Excel file {file_name}: {e}")
        
    return all_text

def extract_text_from_csv(csv_file: Any) -> str:
    """
    Extracts text from a single uploaded CSV file stream.
    
    Args:
        csv_file: A file-like object containing CSV data
        
    Returns:
        Extracted text as a string
    """
    try:
        # Reset stream position
        csv_file.seek(0)
        
        # Try to detect encoding and delimiter
        df = pd.read_csv(csv_file, encoding='utf-8', on_bad_lines='skip')
        
        # Handle empty CSV
        if df.empty:
            return "(Empty CSV file)"
            
        # Clean the dataframe - replace NaNs with empty strings
        df = df.fillna('')
        
        # Get column headers
        headers = " | ".join(str(col) for col in df.columns)
        
        # Start with headers
        all_text = f"{headers}\n"
        
        # Add data rows
        for _, row in df.iterrows():
            row_text = " | ".join(str(val) for val in row.values)
            all_text += f"{row_text}\n"
            
        return all_text
        
    except Exception as e:
        file_name = getattr(csv_file, 'name', 'Unknown')
        logger.error(f"Could not read CSV file {file_name}: {e}")
        st.warning(f"Could not read CSV file {file_name}: {e}")
        return ""

def extract_text_from_text_file(text_file: Any) -> str:
    """
    Extracts text from a single uploaded text file stream.
    
    Args:
        text_file: A file-like object containing text data
        
    Returns:
        Extracted text as a string
    """
    try:
        # Reset stream position
        text_file.seek(0)
        
        # Read the file content
        content = text_file.read()
        
        # Handle binary vs text
        if isinstance(content, bytes):
            # Try utf-8 first, then latin-1 as fallback
            try:
                return content.decode('utf-8')
            except UnicodeDecodeError:
                return content.decode('latin-1')
        else:
            return content
            
    except Exception as e:
        file_name = getattr(text_file, 'name', 'Unknown')
        logger.error(f"Could not read text file {file_name}: {e}")
        st.warning(f"Could not read text file {file_name}: {e}")
        return ""

def detect_file_type(file_name: str) -> str:
    """
    Detects the file type based on the file extension.
    
    Args:
        file_name: Name of the file
        
    Returns:
        File type as a string ('pdf', 'excel', 'csv', 'text', or 'unknown')
    """
    file_name = file_name.lower()
    
    if file_name.endswith('.pdf'):
        return 'pdf'
    elif file_name.endswith(('.xlsx', '.xls')):
        return 'excel'
    elif file_name.endswith('.csv'):
        return 'csv'
    elif file_name.endswith(('.txt', '.md', '.json', '.xml', '.html', '.htm')):
        return 'text'
    else:
        return 'unknown'

def extract_text_from_file(file: Union[Any, str]) -> str:
    """
    Extracts text from a single file (uploaded stream or file path).
    Detects file type based on name/path.

    Args:
        file: Either a file-like object (from st.file_uploader) or a string path.

    Returns:
        Extracted text as a single string.
    """
    file_path = None
    file_stream = None
    file_name = "N/A"

    # Determine if input is a file path or a file stream
    if isinstance(file, str):
        file_path = file
        file_name = os.path.basename(file_path)
    elif hasattr(file, 'name') and hasattr(file, 'seek'): 
        file_stream = file
        file_name = file.name
    else:
        logger.error(f"Unsupported file input type: {type(file)}")
        st.error(f"Unsupported file input type: {type(file)}")
        return ""

    logger.info(f"Processing file: {file_name}")
    file_type = detect_file_type(file_name)
    
    # Extract text based on file type
    extracted_text = ""
    
    # Handle PDF files
    if file_type == 'pdf':
        if file_stream:
            extracted_text = extract_text_from_pdf(file_stream)
        elif file_path and os.path.exists(file_path):
            with open(file_path, "rb") as f:
                extracted_text = extract_text_from_pdf(f)
        else:
            logger.warning(f"PDF file not found at path: {file_path}")
            
    # Handle Excel files
    elif file_type == 'excel':
        if file_stream:
            extracted_text = extract_text_from_excel(file_stream)
        elif file_path and os.path.exists(file_path):
            with open(file_path, "rb") as f:
                extracted_text = extract_text_from_excel(f)
        else:
            logger.warning(f"Excel file not found at path: {file_path}")
            
    # Handle CSV files
    elif file_type == 'csv':
        if file_stream:
            extracted_text = extract_text_from_csv(file_stream)
        elif file_path and os.path.exists(file_path):
            with open(file_path, "rb") as f:
                extracted_text = extract_text_from_csv(f)
        else:
            logger.warning(f"CSV file not found at path: {file_path}")
            
    # Handle text files
    elif file_type == 'text':
        if file_stream:
            extracted_text = extract_text_from_text_file(file_stream)
        elif file_path and os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                extracted_text = f.read()
        else:
            logger.warning(f"Text file not found at path: {file_path}")
            
    # Handle unknown file types
    else:
        logger.warning(f"Unsupported file type: {file_name}. Currently supported: PDF, Excel, CSV, and text files.")
        st.warning(f"Unsupported file type: {file_name}. Currently supported: PDF, Excel, CSV, and text files.")

    # Log extraction results
    if extracted_text:
        text_length = len(extracted_text)
        logger.info(f"Successfully extracted {text_length} characters from {file_name}")
    else:
        logger.warning(f"No text extracted from {file_name}")
        
    return extracted_text

def extract_text_from_files(files: List[Union[Any, str]]) -> str:
    """
    Extracts text from a list of uploaded files or file paths.

    Args:
        files: A list of file-like objects or string paths.

    Returns:
        A single string containing all extracted text.
    """
    if not files:
        logger.warning("No files provided for text extraction")
        return ""
        
    logger.info(f"Extracting text from {len(files)} files")
    
    full_text = ""
    for file in files:
        file_text = extract_text_from_file(file)
        if file_text:
            # Add file separator if not the first file
            if full_text:
                full_text += "\n\n---NEW DOCUMENT---\n\n"
            full_text += file_text
    
    logger.info(f"Total extracted text: {len(full_text)} characters")
    return full_text


# --- Text Cleaning ---

def clean_text(text: str) -> str:
    """
    Cleans extracted text by removing extra whitespaces, fixing common OCR issues, etc.
    
    Args:
        text: The input text string
        
    Returns:
        Cleaned text string
    """
    if not text:
        return ""
        
    # Replace multiple newlines with double newline
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Replace multiple spaces with single space
    text = re.sub(r' {2,}', ' ', text)
    
    # Fix common OCR errors (optional, can be expanded)
    text = text.replace('|', 'I').replace('0', 'O')
    
    # Remove non-printable characters
    text = ''.join(char for char in text if char.isprintable() or char in ['\n', '\t'])
    
    return text.strip()


# --- Text Chunking ---

def get_text_chunks(text: str, 
                    chunk_size: int = 500, 
                    chunk_overlap: int = 100,
                    use_recursive_splitter: bool = True) -> List[str]:
    """
    Splits a large string of text into smaller chunks using text splitters.

    Args:
        text: The input text string.
        chunk_size: Target size for each chunk.
        chunk_overlap: Overlap between consecutive chunks.
        use_recursive_splitter: Whether to use RecursiveCharacterTextSplitter (True) or 
                               basic CharacterTextSplitter (False).

    Returns:
        A list of text chunks. Returns an empty list if input text is empty or splitting fails.
    """
    if not text:
        logger.warning("No text provided for chunking")
        return []
    
    try:
        logger.info(f"Chunking text of length {len(text)} with chunk_size={chunk_size}, overlap={chunk_overlap}")
        
        # Clean the text before chunking
        clean_content = clean_text(text)
        
        # Choose the appropriate text splitter
        if use_recursive_splitter:
            # RecursiveCharacterTextSplitter is better for most use cases
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
                length_function=len
            )
        else:
            # Basic CharacterTextSplitter as a fallback
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                is_separator_regex=False,
            )
            
        # Split text into chunks
        chunks = text_splitter.split_text(clean_content)
        
        # Log results
        if not chunks:
            logger.warning("Text was provided, but splitting resulted in zero chunks. Check text format and splitter settings.")
        else:
            logger.info(f"Split text into {len(chunks)} chunks")
            
        return chunks
    
    except Exception as e:
        logger.error(f"Error splitting text into chunks: {e}")
        st.error(f"Error splitting text into chunks: {e}")
        return []

def get_metadata_from_file(file: Union[Any, str]) -> Dict[str, Any]:
    """
    Extracts metadata from a file.
    
    Args:
        file: Either a file-like object or a string path
        
    Returns:
        Dictionary containing metadata
    """
    metadata = {}
    
    # Get filename
    if isinstance(file, str):
        metadata["source"] = os.path.basename(file)
        metadata["file_path"] = file
    elif hasattr(file, 'name'):
        metadata["source"] = file.name
    else:
        metadata["source"] = "Unknown"
        
    # Get file type
    metadata["file_type"] = detect_file_type(metadata["source"])
    
    # Add timestamp
    import datetime
    metadata["processed_at"] = datetime.datetime.now().isoformat()
    
    return metadata

def create_document_chunks(files: List[Union[Any, str]], 
                          chunk_size: int = 500,
                          chunk_overlap: int = 100) -> List[Dict[str, Any]]:
    """
    Process files and create document chunks with metadata.
    
    Args:
        files: List of files to process
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of dictionaries containing text chunks with metadata
    """
    all_chunks = []
    
    for file in files:
        # Get file metadata
        metadata = get_metadata_from_file(file)
        
        # Extract text
        text = extract_text_from_file(file)
        if not text:
            continue
            
        # Split into chunks
        chunks = get_text_chunks(text, chunk_size, chunk_overlap)
        
        # Create documents with metadata
        for i, chunk in enumerate(chunks):
            chunk_doc = {
                "text": chunk,
                "metadata": {
                    **metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            }
            all_chunks.append(chunk_doc)
    
    return all_chunks