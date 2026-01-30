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

- Uses HuggingFace embedding model (all-MiniLM-L6-v2) to convert text chunks
- Generates embeddings for all document chunks
- Processes embeddings in batches for efficiency

### 3. Vector Database

- Stores embeddings in ChromaDB with persistent storage
- Supports similarity search across chunks
- Maintains metadata with each chunk (timestamps, indices, file info)

### 4. Query Processing

- Embeds user queries using the same HuggingFace model
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
- Snow effect on answer output

## Key Features

✅ **Production-Ready**

- Comprehensive error handling at every step

✅ **Modular Architecture**

- Clear separation of concerns (preprocessing, embedding, retrieval, generation)

✅ **Scalability**

- Batch processing for embeddings

✅ **Security**

- Environment variables for sensitive data (API keys)

## Technical Stack

| Component              | Technology                         |
| ---------------------- | ---------------------------------- |
| UI Framework           | Streamlit 1.28.1                   |
| Vector Database        | ChromaDB 0.4.24                    |
| Document Processing    | LangChain 0.1.10 + PyPDF2 3.0.1    |
| Embeddings             | HuggingFace (all-MiniLM-L6-v2)     |
| LLM                    | Groq API (llama-3.3-70b-versatile) |
| Environment Management | python-dotenv 1.0.0                |

## Code Organization

### Main Components

1. **Configuration & Setup**
   - Logging configuration
   - Streamlit setup
   - Environment variables and constants

2. **Text Preprocessing**
   - `clean_text()`: Removes unwanted characters, URLs, emails
   - `normalize_text()`: Converts to lowercase, normalizes spacing

3. **Document Processing**
   - `extract_text_from_pdf()`: PDF text extraction
   - `extract_text_from_txt()`: TXT file reading
   - `chunk_documents()`: Text chunking with overlap

4. **Embedding & Vector DB**
   - `initialize_chroma_db()`: ChromaDB initialization
   - `generate_embeddings_batch()`: Batch embedding generation (HuggingFace)
   - `store_embeddings_in_chromadb()`: Storage and indexing

5. **Retrieval & Context**
   - `retrieve_relevant_chunks()`: Semantic search
   - `construct_context()`: Context formatting for LLM

6. **LLM Integration**
   - `generate_answer()`: LLM-based answer generation (Groq)
   - `post_process_answer()`: Answer formatting and cleanup

7. **Session Management**
   - Session state initialization
   - Collection management

8. **Streamlit UI** (in `ui.py`)
   - Sidebar controls
   - Document upload interface
   - Q&A chat interface
   - Chat history display
   - Snow effect on answer output

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
| `app.py`           | Core logic and backend functions    |
| `ui.py`            | Streamlit UI entry point            |
| `requirements.txt` | Python dependencies                 |
| `.gitignore`       | Git ignore rules for sensitive data |
| `.env.example`     | Template for environment variables  |
| `SETUP.md`         | Setup and troubleshooting guide     |

## Error Handling

The system handles:

- Missing API keys (with clear error message)
- PDF extraction failures (with fallback)
- Embedding generation failures (with error message)
- ChromaDB connection issues
- Missing collections
- LLM API errors
- File upload errors

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

**Version**: 1.0  
**Last Updated**: January 2026  
**Status**: Production Ready
