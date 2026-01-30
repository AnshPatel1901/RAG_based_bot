# Setup Instructions

## 1. Create Virtual Environment

```bash
python -m venv venv
```

## 2. Activate Virtual Environment

### On Windows:

```bash
venv\Scripts\activate
```

### On macOS/Linux:

```bash
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Configure Environment Variables

- Copy `.env.example` to `.env`
- Open `.env` and add your Groq API key:
  ```
  GROQ_API_KEY=your_actual_api_key_here
  ```
- Get your API key from: https://console.groq.com/keys

## 5. Run the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## Features

- Upload PDF and TXT documents
- Automatic text cleaning and chunking
- Vector embeddings using Groq API
- ChromaDB for persistent vector storage
- RAG-based question answering
- Chat history tracking
- Source document viewing

## File Structure

```
RAG_based_bot/
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variables template
├── .gitignore          # Git ignore rules
├── .env                # Local environment variables (not in git)
├── chroma_data/        # ChromaDB storage (not in git)
├── rag_system.log      # Application logs (not in git)
└── venv/              # Virtual environment (not in git)
```

## Troubleshooting

### Issue: GROQ_API_KEY not found

- Ensure `.env` file exists in the project root
- Verify the API key is correctly formatted
- Restart the application after updating `.env`

### Issue: ChromaDB errors

- Delete `chroma_data/` folder to reset the database
- Restart the application

### Issue: Embedding generation fails

- Check your Groq API quota
- Ensure internet connection is stable

## Production Deployment

- Use environment variables from your deployment platform
- Never commit `.env` file
- Monitor `rag_system.log` for errors
- Consider using a dedicated vector database for scale
- Implement rate limiting for Groq API calls
