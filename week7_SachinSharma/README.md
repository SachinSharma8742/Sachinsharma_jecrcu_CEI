# Document Question Answering System (RAG)

This project is a Retrieval-Augmented Generation (RAG) system built with Python, LangChain, and Streamlit. It allows you to upload custom PDF documents and ask questions about their content. Instead of just guessing, the language model uses the actual text from your documents to generate accurate, grounded answers.

## How it works

1. **Document Ingestion**: You upload a PDF file through the web interface.
2. **Text Chunking**: The system extracts the text and splits it into smaller, manageable chunks.
3. **Embeddings**: These text chunks are converted into mathematical vectors using a local Hugging Face embedding model (all-MiniLM-L6-v2).
4. **Storage**: The vectors are saved in a local Chroma vector database for fast similarity searches.
5. **Retrieval and Generation**: When you ask a question, the system finds the most relevant text chunks from the database and passes them to the Google Gemini model to construct a final answer. 

## Setup

First, make sure you have Python installed. Then, install the required dependencies:

```bash
pip install -r requirements.txt
```

You will need a Google Gemini API key to generate answers. You can create a `.env` file in the root directory and add your key like this:

```
GOOGLE_API_KEY=your_api_key_here
```

Alternatively, you can just paste your API key directly into the sidebar when you launch the application.

## Running the application

To start the web interface, run the following command in your terminal:

```bash
streamlit run app.py
```

This will automatically open the application in your default web browser. From there, you can upload a PDF, wait a moment for it to process, and start asking questions about your data.
