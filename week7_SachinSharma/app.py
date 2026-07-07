import streamlit as st
import os
import dotenv
from rag_pipeline import (
    load_pdf,
    process_and_store_documents,
    get_rag_chain,
    get_vectorstore,
    VECTOR_STORE_DIR
)
import shutil

dotenv.load_dotenv()

st.set_page_config(page_title="RAG Document QA System", layout="wide")

st.title("📚 Document Question Answering System (RAG)")
st.write("Upload your own PDFs or use the default Hugging Face dataset to test Retrieval-Augmented Generation!")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key Input
    api_key = st.text_input("Google Gemini API Key", type="password", value=os.getenv("GOOGLE_API_KEY", ""))
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        
    st.divider()
    
    st.header("📄 Document Source")
    st.write("Upload your own custom data (PDFs) to create your private knowledge base.")
    
    if st.button("Clear Database"):
        vectorstore = get_vectorstore()
        if vectorstore:
            try:
                vectorstore.delete_collection()
            except Exception:
                pass
        
        if os.path.exists(VECTOR_STORE_DIR):
            import shutil
            shutil.rmtree(VECTOR_STORE_DIR, ignore_errors=True)
            
        if "processed_file" in st.session_state:
            del st.session_state["processed_file"]
            
        st.success("Database cleared successfully!")
        st.rerun()

    st.divider()

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file is not None:
        if st.session_state.get("processed_file") != uploaded_file.name:
            with st.spinner("Extracting text and building vector store..."):
                docs = load_pdf(uploaded_file)
                process_and_store_documents(docs)
                st.session_state["processed_file"] = uploaded_file.name
                st.success(f"✅ {uploaded_file.name} loaded and processed!")
        else:
            st.success(f"✅ {uploaded_file.name} loaded and processed!")

# --- Main QA Interface ---
st.header("💬 Ask Questions")

if not os.environ.get("GOOGLE_API_KEY"):
    st.warning("⚠️ Please enter your Google Gemini API Key in the sidebar to ask questions.")
elif not os.path.exists(VECTOR_STORE_DIR) or get_vectorstore() is None:
    st.info("👈 Please upload a document in the sidebar first to build the vector database.")
else:
    question = st.text_input("What would you like to know about the document(s)?")
    
    if st.button("Get Answer"):
        if question:
            with st.spinner("Generating answer..."):
                try:
                    rag_chain = get_rag_chain()
                    if rag_chain:
                        response = rag_chain.invoke({"input": question})
                        
                        st.subheader("Answer:")
                        st.write(response["answer"])
                        
                        # Display sources for transparency
                        with st.expander("View Source Citations"):
                            for i, doc in enumerate(response["context"]):
                                source_file = doc.metadata.get('source', 'Unknown Source')
                                page = doc.metadata.get('page', 'Unknown Page')
                                st.markdown(f"**Source File:** `{source_file}` (Page {page})")
                                st.info(doc.page_content)
                                st.divider()
                    else:
                        st.error("Could not initialize the RAG chain.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a question.")
