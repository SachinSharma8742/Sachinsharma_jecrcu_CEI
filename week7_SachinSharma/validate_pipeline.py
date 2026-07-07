import os
import sys
import dotenv
from rag_pipeline import (
    load_huggingface_dataset,
    process_and_store_documents,
    get_rag_chain
)

# Load environment variables
dotenv.load_dotenv()

def validate_pipeline():
    print("=" * 60)
    print("Starting RAG Pipeline Validation")
    print("=" * 60)

    # 1. Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY is not set in the environment or .env file.")
        print("Please set it before running the validation.")
        sys.exit(1)

    log_file = open("validation_logs.txt", "w", encoding="utf-8")
    
    def log_and_print(text):
        print(text)
        log_file.write(text + "\n")

    log_and_print("Step 1: Loading default Hugging Face Dataset for verification...")
    # Using ag_news fallback dataset slice as it is small and loads quickly for validation
    docs = load_huggingface_dataset(dataset_name="ag_news", max_samples=20)
    log_and_print(f"Loaded {len(docs)} documents.")

    log_and_print("\nStep 2: Chunking, Embedding, and Storing in Vector DB (Chroma)...")
    process_and_store_documents(docs)
    log_and_print("Vector Database created successfully.")

    log_and_print("\nStep 3: Initializing RAG Retrieval Chain...")
    try:
        rag_chain = get_rag_chain()
        log_and_print("RAG Chain initialized successfully.")
    except Exception as e:
        log_and_print(f"Failed to initialize RAG chain: {e}")
        sys.exit(1)

    log_and_print("\nStep 4: Executing Dynamic Sample Queries...")
    
    sample_queries = [
        "What is the main topic discussed in the documents?",
        "Are there any mentions of sports or business?"
    ]

    for i, query in enumerate(sample_queries, 1):
        log_and_print("-" * 60)
        log_and_print(f"Query {i}: {query}")
        
        try:
            response = rag_chain.invoke({"input": query})
            
            log_and_print("\n>>> Grounded Answer:")
            log_and_print(response['answer'])
            
            log_and_print("\n>>> Retrieved Context (Extraction Validation):")
            for j, doc in enumerate(response['context'], 1):
                log_and_print(f"  Source Chunk {j}:")
                log_and_print(f"  {doc.page_content[:200]}...")
                log_and_print(f"  Metadata: {doc.metadata}\n")
                
        except Exception as e:
            log_and_print(f"Error during retrieval/generation: {e}")

    log_and_print("=" * 60)
    log_and_print("Validation Complete. Logs saved to 'validation_logs.txt'.")
    log_file.close()

if __name__ == "__main__":
    validate_pipeline()
