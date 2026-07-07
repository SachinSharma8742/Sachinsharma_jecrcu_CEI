import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from datasets import load_dataset
import dotenv

dotenv.load_dotenv()

# Global variables for vectorstore to keep it loaded in memory during app run
VECTOR_STORE_DIR = "./chroma_db"

def get_embeddings_model():
    """Initializes the HuggingFace embeddings model (runs locally)."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")



def load_pdf(uploaded_file):
    """
    Loads a PDF from a Streamlit UploadedFile object.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    os.remove(tmp_path)
    return documents

def process_and_store_documents(documents):
    """
    Splits documents into chunks and stores them in ChromaDB.
    """
    # 1. Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        add_start_index=True
    )
    splits = text_splitter.split_documents(documents)

    # 2. Embeddings & Vector DB
    embeddings = get_embeddings_model()
    
    # Create or update vector store
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_DIR
    )
    return vectorstore

def get_vectorstore():
    """
    Retrieves the existing vector store if it exists and is not empty.
    """
    if os.path.exists(VECTOR_STORE_DIR):
        vs = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=get_embeddings_model())
        try:
            # Check if there are any documents in the DB
            if vs._collection.count() == 0:
                return None
        except Exception:
            pass
        return vs
    return None

def get_rag_chain():
    """
    Sets up the RAG pipeline using Google Gemini and ChromaDB retriever.
    """
    vectorstore = get_vectorstore()
    if not vectorstore:
        return None

    # Retrieve Google API Key
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("Google API key is missing. Please set GOOGLE_API_KEY in the .env file or UI.")

    # Initialize Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        max_tokens=500,
        google_api_key=google_api_key
    )

    # Setup retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Setup Prompt Template
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "Use three sentences maximum and keep the answer concise.\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Create the chains using LCEL
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {
            "context": (lambda x: x["input"]) | retriever, 
            "input": (lambda x: x["input"])
        }
    ).assign(answer=rag_chain_from_docs)

    return rag_chain_with_source
