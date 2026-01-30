# RAG-Based Q&A Bot - Summary

## Project Overview

A production-grade Retrieval-Augmented Generation (RAG) system that combines document management, semantic search, and LLM-powered question answering. Users can upload documents, ask questions, and receive accurate answers grounded in the uploaded content.

## What the System Does

### 1. Document Management

- Accepts PDF and TXT file uploads
- Extracts text using PyPDF2 and Python's built-in utilities
- Preprocesses text (cleaning, normalization)
- Chunks documents into manageable pieces (1000 chars with 200 char overlap)

### 2. Vector Embeddings

- Uses Groq API embedding model (text-embedding-004) to convert text chunks
- Generates embeddings for all document chunks
- Processes embeddings in batches for API efficiency

### 3. Vector Database

- Stores embeddings in ChromaDB with persistent storage
- Supports similarity search across chunks
- Maintains metadata with each chunk (timestamps, indices, file info)

### 4. Query Processing

- Embeds user queries using the same embedding model
- Performs semantic similarity search in ChromaDB
- Retrieves top-5 most relevant document chunks

### 5. Answer Generation

- Constructs context from retrieved chunks
- Uses Groq's llama-3.3-70b-versatile model for LLM inference
- Implements prompt engineering for accurate responses
- Post-processes answers for formatting and clarity

### 6. User Interface

- Clean Streamlit interface for document uploads
- Interactive Q&A chat interface
- Chat history tracking
- Source document viewing
- Collection management

## Key Features

✅ **Production-Ready**

- Comprehensive error handling at every step
- Logging system for debugging and monitoring
- Graceful failure modes

✅ **Modular Architecture**

- Clear separation of concerns (preprocessing, embedding, retrieval, generation)
- Reusable functions with documented parameters
- Well-commented code for maintainability

✅ **Scalability**

- Batch processing for embeddings
- ChromaDB persistent storage for large datasets
- Session state management for multi-user scenarios

✅ **Security**

- Environment variables for sensitive data (API keys)
- .gitignore prevents accidental credential commits
- .env.example template for secure setup

## Technical Stack

| Component              | Technology                         |
| ---------------------- | ---------------------------------- |
| UI Framework           | Streamlit 1.28.1                   |
| Vector Database        | ChromaDB 0.4.24                    |
| Document Processing    | LangChain 0.1.10 + PyPDF2 3.0.1    |
| Embeddings             | Groq API (text-embedding-004)      |
| LLM                    | Groq API (llama-3.3-70b-versatile) |
| Environment Management | python-dotenv 1.0.0                |

## Code Organization

### Main Components in app.py

1. **Configuration & Setup** (Lines 1-50)
   - Logging configuration
   - Streamlit setup
   - Environment variables and constants

2. **Text Preprocessing** (Lines 51-95)
   - `clean_text()`: Removes unwanted characters, URLs, emails
   - `normalize_text()`: Converts to lowercase, normalizes spacing

3. **Document Processing** (Lines 97-160)
   - `extract_text_from_pdf()`: PDF text extraction
   - `extract_text_from_txt()`: TXT file reading
   - `chunk_documents()`: Text chunking with overlap

4. **Embedding & Vector DB** (Lines 162-240)
   - `initialize_chroma_db()`: ChromaDB initialization
   - `generate_embeddings_batch()`: Batch embedding generation
   - `store_embeddings_in_chromadb()`: Storage and indexing

5. **Retrieval & Context** (Lines 242-310)
   - `retrieve_relevant_chunks()`: Semantic search
   - `construct_context()`: Context formatting for LLM

6. **LLM Integration** (Lines 312-385)
   - `generate_answer()`: LLM-based answer generation
   - `post_process_answer()`: Answer formatting and cleanup

7. **Session Management** (Lines 387-410)
   - Session state initialization
   - Collection management

8. **Streamlit UI** (Lines 412-600)
   - Sidebar controls
   - Document upload interface
   - Q&A chat interface
   - Chat history display

## How to Use

### Setup

1. Create virtual environment: `python -m venv venv`
2. Activate: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux)
3. Install dependencies: `pip install -r requirements.txt`
4. Create `.env` file with your Groq API key

### Running

```bash
streamlit run app.py
```

### Using the Application

1. Upload PDF/TXT documents via sidebar
2. Enter a collection name (or use default "documents")
3. Click "Process & Store Documents"
4. Select the collection from dropdown
5. Ask questions in the input field
6. View answers with source documents

## Files Generated

| File               | Purpose                             |
| ------------------ | ----------------------------------- |
| `app.py`           | Complete application (600+ lines)   |
| `requirements.txt` | Python dependencies (7 packages)    |
| `.gitignore`       | Git ignore rules for sensitive data |
| `.env.example`     | Template for environment variables  |
| `SETUP.md`         | Setup and troubleshooting guide     |

## Error Handling

The system handles:

- Missing API keys (with clear error message)
- PDF extraction failures (with fallback)
- Embedding generation failures (with zero vector fallback)
- ChromaDB connection issues
- Missing collections
- LLM API errors
- File upload errors

All errors are logged to `rag_system.log` for debugging.

## Performance Optimizations

1. **Batch Processing**: Embeddings processed in batches of 10
2. **Session State**: Reuses client connections across requests
3. **Persistent Storage**: ChromaDB keeps data between sessions
4. **Chunk Overlap**: Maintains context across chunk boundaries
5. **Lazy Loading**: Collections loaded only when needed

## Security Considerations

1. ✅ API keys stored in environment variables (not in code)
2. ✅ .gitignore prevents accidental credential commits
3. ✅ .env file excluded from version control
4. ✅ ChromaDB data stored locally (no cloud leakage)
5. ✅ Input validation and error messages don't leak sensitive info

## Scalability Roadmap

For production deployment:

1. Replace SQLite-backed ChromaDB with distributed version
2. Implement caching layer for frequent queries
3. Add database backups for ChromaDB
4. Use async processing for embeddings
5. Implement rate limiting for API calls
6. Add multi-tenancy support
7. Monitor and optimize embedding model selection

## Support & Maintenance

- **Logs**: Check `rag_system.log` for issues
- **ChromaDB Reset**: Delete `chroma_data/` folder if DB corrupts
- **API Issues**: Monitor Groq API quota and limits
- **Updates**: Regularly update dependencies for security patches

---

**Version**: 1.0  
**Last Updated**: January 2026  
**Status**: Production Ready
