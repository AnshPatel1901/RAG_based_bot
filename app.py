"""
Production-Grade RAG (Retrieval-Augmented Generation) System
Using Streamlit, ChromaDB, LangChain, and Groq API

This application allows users to upload documents, process them into embeddings,
store them in a vector database, and retrieve relevant information to answer queries
using an LLM powered by the Groq API.
"""

import os
import sys
import logging
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime

import streamlit as st
import chromadb
from chromadb.config import Settings
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
print(model)

from groq import Groq
import re
import string
from dotenv import load_dotenv
load_dotenv()

# ============================================================================
# CONFIGURATION AND SETUP
# ============================================================================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Streamlit page configuration
st.set_page_config(
    page_title="RAG-Based Q&A Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# ENVIRONMENT VARIABLES AND CONSTANTS
# ============================================================================

# Load API key from environment or .env file
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    logger.warning("GROQ_API_KEY not found in environment variables")

# ChromaDB configuration
CHROMA_DB_PATH = "./chroma_data"
EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RETRIEVAL = 5

# Groq LLM configuration
LLM_MODEL = "llama-3.3-70b-versatile"
TEMPERATURE = 0.7
MAX_TOKENS = 1024

# ============================================================================
# TEXT PREPROCESSING AND CLEANING
# ============================================================================

def clean_text(text: str) -> str:
    """
    Clean and preprocess raw text from documents.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text
    """
    try:
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        
        # Remove control characters but keep newlines for structure
        text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Normalize spaces
        text = text.strip()
        
        logger.info("Text cleaning completed successfully")
        return text
    except Exception as e:
        logger.error(f"Error during text cleaning: {str(e)}")
        raise


def normalize_text(text: str) -> str:
    """
    Normalize text for consistency.
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text
    """
    try:
        # Convert to lowercase for consistency in embeddings
        text = text.lower()
        
        # Remove extra spaces
        text = ' '.join(text.split())
        
        logger.info("Text normalization completed successfully")
        return text
    except Exception as e:
        logger.error(f"Error during text normalization: {str(e)}")
        raise


# ============================================================================
# DOCUMENT PROCESSING
# ============================================================================

def extract_text_from_pdf(pdf_file) -> str:
    """
    Extract text from PDF file.
    
    Args:
        pdf_file: Uploaded PDF file object
        
    Returns:
        Extracted text from PDF
    """
    try:
        logger.info(f"Starting PDF extraction from {pdf_file.name}")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.getbuffer())
            tmp_path = tmp_file.name
        
        # Extract text using PyPDF2
        text = ""
        try:
            with open(tmp_path, 'rb') as pdf:
                reader = PyPDF2.PdfReader(pdf)
                num_pages = len(reader.pages)
                
                for page_num in range(num_pages):
                    page = reader.pages[page_num]
                    text += f"\n--- Page {page_num + 1} ---\n"
                    text += page.extract_text()
            
            logger.info(f"Successfully extracted text from {num_pages} pages")
            return text
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise


def extract_text_from_txt(txt_file) -> str:
    """
    Extract text from TXT file.
    
    Args:
        txt_file: Uploaded TXT file object
        
    Returns:
        Extracted text from TXT file
    """
    try:
        logger.info(f"Starting text extraction from {txt_file.name}")
        text = txt_file.read().decode('utf-8', errors='ignore')
        logger.info("Successfully extracted text from TXT file")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from TXT: {str(e)}")
        raise


def chunk_documents(text: str, chunk_size: int = CHUNK_SIZE, 
                   chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split document text into chunks using RecursiveCharacterTextSplitter.
    
    Args:
        text: Full document text
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    try:
        logger.info(f"Starting document chunking (size={chunk_size}, overlap={chunk_overlap})")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = splitter.split_text(text)
        logger.info(f"Document split into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.error(f"Error chunking documents: {str(e)}")
        raise


# ============================================================================
# EMBEDDING AND VECTOR DATABASE
# ============================================================================

def initialize_chroma_db() -> chromadb.Client:
    """
    Initialize ChromaDB client with persistent storage using the new architecture.

    Returns:
        ChromaDB client instance
    """
    try:
        logger.info(f"Initializing ChromaDB with path: {CHROMA_DB_PATH}")

        # Create directory if it doesn't exist
        os.makedirs(CHROMA_DB_PATH, exist_ok=True)

        # Initialize ChromaDB with the new client configuration
        client = chromadb.Client(
            chromadb.config.Settings(
                persist_directory=CHROMA_DB_PATH,
                anonymized_telemetry=False
            )
        )

        logger.info("ChromaDB initialized successfully")
        return client
    except Exception as e:
        logger.error(f"Error initializing ChromaDB: {str(e)}")
        raise


def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a batch of texts using HuggingFaceEmbeddings.

    Args:
        texts: List of text chunks to embed

    Returns:
        List of embedding vectors
    """
    try:
        logger.info(f"Generating embeddings for {len(texts)} text chunks")
        embeddings = EMBEDDING_MODEL.embed_documents(texts)
        logger.info(f"Successfully generated {len(embeddings)} embeddings")
        return embeddings
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise


def store_embeddings_in_chromadb(collection_name: str, chunks: List[str], 
                                 embeddings: List[List[float]], 
                                 metadata: List[dict] = None) -> None:
    """
    Store embeddings and chunks in ChromaDB.
    
    Args:
        collection_name: Name of the collection in ChromaDB
        chunks: List of text chunks
        embeddings: List of embedding vectors
        metadata: Optional metadata for each chunk
    """
    try:
        logger.info(f"Storing embeddings in ChromaDB collection: {collection_name}")
        
        client = initialize_chroma_db()
        
        # Get or create collection
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Prepare metadata if not provided
        if metadata is None:
            metadata = [{"chunk_index": i} for i in range(len(chunks))]
        
        # Add embeddings and texts to collection
        ids = [f"{collection_name}_chunk_{i}" for i in range(len(chunks))]
        
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadata
        )

        logger.info(f"Successfully stored {len(chunks)} chunks in ChromaDB")
    except Exception as e:
        logger.error(f"Error storing embeddings in ChromaDB: {str(e)}")
        raise


# ============================================================================
# RETRIEVAL AND CONTEXT CONSTRUCTION
# ============================================================================

def retrieve_relevant_chunks(query: str, collection_name: str, 
                            k: int = TOP_K_RETRIEVAL) -> Tuple[List[str], List[dict]]:
    """
    Retrieve top-k most relevant document chunks based on query.
    
    Args:
        query: User query
        collection_name: Name of ChromaDB collection to search
        k: Number of top chunks to retrieve
        
    Returns:
        Tuple of (retrieved chunks, metadata)
    """
    try:
        logger.info(f"Retrieving top-{k} relevant chunks for query: {query[:100]}")
        
        # Initialize ChromaDB client
        client = initialize_chroma_db()
        collection = client.get_collection(name=collection_name)
        
        # Retrieve similar documents
        results = collection.query(
            query_texts=[query],
            n_results=k
        )
        
        # Extract documents and metadata
        retrieved_chunks = results['documents'][0] if results['documents'] else []
        metadatas = results['metadatas'][0] if results['metadatas'] else []
        distances = results['distances'][0] if results['distances'] else []
        
        logger.info(f"Retrieved {len(retrieved_chunks)} relevant chunks")
        return retrieved_chunks, metadatas
    except Exception as e:
        logger.error(f"Error retrieving chunks from ChromaDB: {str(e)}")
        return [], []


def construct_context(chunks: List[str], metadatas: List[dict]) -> str:
    """
    Construct context from retrieved chunks for LLM.
    
    Args:
        chunks: Retrieved text chunks
        metadatas: Metadata associated with chunks
        
    Returns:
        Formatted context string
    """
    try:
        logger.info("Constructing context for LLM")
        
        context = "## Retrieved Context:\n\n"
        
        for i, (chunk, metadata) in enumerate(zip(chunks, metadatas), 1):
            context += f"**Source {i}:**\n"
            if metadata and 'chunk_index' in metadata:
                context += f"(Chunk {metadata['chunk_index']})\n"
            context += f"{chunk}\n\n"
        
        logger.info("Context construction completed")
        return context
    except Exception as e:
        logger.error(f"Error constructing context: {str(e)}")
        return ""


# ============================================================================
# LLM ANSWER GENERATION
# ============================================================================

def generate_answer(query: str, context: str, groq_client: Groq) -> str:
    """
    Generate answer using Groq LLM based on query and context.

    Args:
        query: User's question
        context: Retrieved context from documents
        groq_client: Groq API client

    Returns:
        Generated answer from LLM
    """
    try:
        logger.info("Generating answer using Groq LLM")

        # Construct prompt with context
        system_prompt = """You are a helpful and knowledgeable assistant. 
        You answer questions based on the provided context from documents. 
        If the information is not available in the context, clearly state that.
        Always provide clear, accurate, and concise responses."""

        user_prompt = f"""Based on the following context, please answer the question.
        
{context}

Question: {query}

Please provide a comprehensive and accurate answer based on the context above."""

        # Call Groq API
        response = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )

        answer = response.choices[0].message.content
        logger.info("Answer generation completed successfully")
        return answer
    except Exception as e:
        logger.error(f"Error generating answer from LLM: {str(e)}")
        raise


def post_process_answer(answer: str) -> str:
    """
    Post-process LLM-generated answer for formatting and safety.
    
    Args:
        answer: Raw LLM output
        
    Returns:
        Cleaned and formatted answer
    """
    try:
        logger.info("Post-processing answer")
        
        # Remove redundant information
        answer = re.sub(r'\n\n+', '\n\n', answer)
        
        # Ensure proper ending
        if answer and not answer.rstrip().endswith(('.', '!', '?')):
            answer = answer.rstrip() + '.'
        
        logger.info("Answer post-processing completed")
        return answer
    except Exception as e:
        logger.error(f"Error post-processing answer: {str(e)}")
        return answer


# ============================================================================
# SESSION STATE MANAGEMENT
# ============================================================================

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'chroma_collections' not in st.session_state:
        st.session_state.chroma_collections = []
    if 'current_collection' not in st.session_state:
        st.session_state.current_collection = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'groq_api_key' not in st.session_state:
        st.session_state.groq_api_key = os.getenv("GROQ_API_KEY", "")
    if 'groq_client' not in st.session_state:
        if st.session_state.groq_api_key:
            st.session_state.groq_client = Groq(api_key=st.session_state.groq_api_key)
        else:
            st.session_state.groq_client = None


def get_available_collections() -> List[str]:
    """Get list of available ChromaDB collections."""
    try:
        client = initialize_chroma_db()
        collections = client.list_collections()
        return [col.name for col in collections]
    except Exception as e:
        logger.error(f"Error retrieving collections: {str(e)}")
        return []


# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    from ui import run_streamlit_ui
    run_streamlit_ui()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        import streamlit as st
        st.error(f"❌ Application error: {str(e)}")

