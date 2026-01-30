import os
from datetime import datetime
import streamlit as st
from groq import Groq

# Import all logic and helpers from app.py
from app import (
    initialize_session_state,
    extract_text_from_pdf,
    extract_text_from_txt,
    clean_text,
    normalize_text,
    chunk_documents,
    generate_embeddings_batch,
    store_embeddings_in_chromadb,
    get_available_collections,
    retrieve_relevant_chunks,
    construct_context,
    generate_answer,
    post_process_answer,
    logger,
    TOP_K_RETRIEVAL
)

def run_streamlit_ui():
    """Main Streamlit application UI."""
    # Initialize session state
    initialize_session_state()

    # Sidebar for configuration
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.markdown("---")

    # API Key Configuration Section
    st.sidebar.subheader("🔑 API Key Configuration")
    api_key_input = st.sidebar.text_input(
        "Enter Groq API Key",
        value=st.session_state.groq_api_key,
        type="password",
        help="Enter your Groq API key (required for LLM generation)"
    )

    if api_key_input and api_key_input != st.session_state.groq_api_key:
        st.session_state.groq_api_key = api_key_input
        st.session_state.groq_client = Groq(api_key=api_key_input)
        os.environ["GROQ_API_KEY"] = api_key_input
        st.sidebar.success("✅ API Key updated successfully!")
        logger.info("GROQ_API_KEY updated via Streamlit UI")

    if not st.session_state.groq_api_key:
        st.sidebar.warning("⚠️ Groq API Key not configured. Enter your key above to enable LLM features.")
    else:
        st.sidebar.info(f"✅ API Key configured (Key: {st.session_state.groq_api_key[:10]}...)")

    st.sidebar.markdown("---")

    # Document upload section
    st.sidebar.subheader("📚 Upload Documents")
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF or TXT files",
        type=['pdf', 'txt'],
        accept_multiple_files=True,
        help="Upload documents to build the knowledge base"
    )

    # Collection name input
    collection_name = st.sidebar.text_input(
        "Collection Name",
        value="documents",
        help="Name for this document collection"
    )

    # Process uploaded files
    if uploaded_files and st.sidebar.button("📤 Process & Store Documents", use_container_width=True):
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()

        try:
            all_text = ""
            total_files = len(uploaded_files)

            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {idx + 1}/{total_files}: {uploaded_file.name}")

                # Extract text based on file type
                if uploaded_file.type == "application/pdf":
                    text = extract_text_from_pdf(uploaded_file)
                else:
                    text = extract_text_from_txt(uploaded_file)

                all_text += f"\n--- File: {uploaded_file.name} ---\n{text}"
                progress_bar.progress((idx + 1) / (total_files + 3))

            # Clean and normalize text
            status_text.text("Cleaning and normalizing text...")
            all_text = clean_text(all_text)
            all_text = normalize_text(all_text)
            progress_bar.progress((total_files + 1) / (total_files + 3))

            # Chunk documents
            status_text.text("Chunking documents...")
            chunks = chunk_documents(all_text)
            progress_bar.progress((total_files + 2) / (total_files + 3))

            # Generate embeddings
            status_text.text("Generating embeddings...")
            embeddings = generate_embeddings_batch(chunks)

            # Store in ChromaDB
            status_text.text("Storing in vector database...")
            metadata = [
                {
                    "chunk_index": i,
                    "timestamp": datetime.now().isoformat(),
                    "file_count": total_files
                }
                for i in range(len(chunks))
            ]
            store_embeddings_in_chromadb(collection_name, chunks, embeddings, metadata)

            progress_bar.progress(1.0)
            status_text.text("✅ Documents processed and stored successfully!")
            st.sidebar.success(f"✅ Processed {total_files} file(s) with {len(chunks)} chunks")

            # Update session state
            st.session_state.current_collection = collection_name
            st.session_state.chroma_collections = get_available_collections()

        except Exception as e:
            st.sidebar.error(f"❌ Error processing documents: {str(e)}")
            logger.error(f"Error in document processing: {str(e)}")

    st.sidebar.markdown("---")

    # Collection selector
    st.sidebar.subheader("📑 Available Collections")
    available_collections = get_available_collections()
    if available_collections:
        selected_collection = st.sidebar.selectbox(
            "Select Collection",
            available_collections,
            index=0 if st.session_state.current_collection not in available_collections \
                  else available_collections.index(st.session_state.current_collection)
        )
        st.session_state.current_collection = selected_collection
    else:
        st.sidebar.info("No collections available. Upload documents to create one.")

    # Main content area
    st.title("🤖 RAG-Based Q&A Bot")
    st.markdown("Ask questions about your uploaded documents using AI-powered retrieval and generation.")

    # Check if collection is selected
    if not st.session_state.current_collection or st.session_state.current_collection not in get_available_collections():
        st.warning("⚠️ Please upload documents and select a collection to proceed.")
        return

    # Chat interface
    st.markdown("---")
    st.subheader("💬 Ask a Question")

    # User query input
    user_query = st.text_input(
        "Your Question:",
        placeholder="Enter your question here...",
        key="user_query_input"
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        submit_button = st.button("🔍 Get Answer", use_container_width=True)

    with col2:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    # Process query and generate answer
    if submit_button and user_query.strip():
        # Check if API key is configured
        if not st.session_state.groq_api_key:
            st.error("❌ Groq API Key not configured. Please enter your API key in the Configuration section on the left sidebar.")
            return

        try:
            with st.spinner("🔄 Retrieving relevant information and generating answer..."):
                # Retrieve relevant chunks
                retrieved_chunks, metadatas = retrieve_relevant_chunks(
                    user_query,
                    st.session_state.current_collection,
                    TOP_K_RETRIEVAL
                )

                if not retrieved_chunks:
                    st.warning("⚠️ No relevant information found in the documents.")
                    logger.warning(f"No relevant chunks retrieved for query: {user_query}")
                    return

                # Construct context
                context = construct_context(retrieved_chunks, metadatas)

                # Generate answer
                answer = generate_answer(user_query, context, st.session_state.groq_client)
                answer = post_process_answer(answer)

                # Add to chat history
                st.session_state.chat_history.append({
                    "query": user_query,
                    "answer": answer,
                    "timestamp": datetime.now().isoformat(),
                    "sources": len(retrieved_chunks)
                })

                # Display answer
                st.markdown("---")
                st.subheader("✅ Answer")
                st.markdown(answer)

                st.snow()

                # Display sources
                # with st.expander("📖 View Sources"):
                #     for i, chunk in enumerate(retrieved_chunks, 1):
                #         st.write(f"**Source {i}:**")
                #         st.text(chunk[:500] + "..." if len(chunk) > 500 else chunk)
                #         st.divider()

        except Exception as e:
            st.error(f"❌ Error generating answer: {str(e)}")
            logger.error(f"Error in query processing: {str(e)}")

    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("📜 Chat History")

        for idx, chat in enumerate(st.session_state.chat_history, 1):
            with st.expander(f"Q{idx}: {chat['query'][:50]}..."):
                st.write(f"**Question:** {chat['query']}")
                st.write(f"**Answer:** {chat['answer']}")
                st.caption(f"Time: {chat['timestamp']} | Sources: {chat['sources']}")

    # Footer
    st.markdown("---")
    st.caption("RAG-Based Q&A Bot v1.0 | Powered by Streamlit, ChromaDB, and Groq API")