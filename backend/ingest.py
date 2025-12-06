import os
import glob
from typing import List
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- Configuration ---
SOURCE_DIRECTORY = os.path.join(os.path.dirname(__file__), "data", "source_documents")
PERSIST_DIRECTORY = os.path.join(os.path.dirname(__file__), "data", "chroma_db")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def load_documents(source_dir: str) -> List:
    """
    Loads all .pdf, .txt, and .md files from the source directory.
    """
    documents = []
    
    # Define supported file types and their loaders
    loaders = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".md": UnstructuredMarkdownLoader,
    }

    print(f"Scanning {source_dir} for documents...")
    
    for ext, loader_cls in loaders.items():
        file_pattern = os.path.join(source_dir, f"*{ext}")
        files = glob.glob(file_pattern)
        
        for file_path in files:
            try:
                print(f"Loading: {file_path}")
                loader = loader_cls(file_path)
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    return documents

def main():
    # 1. Load Documents
    if not os.path.exists(SOURCE_DIRECTORY):
        os.makedirs(SOURCE_DIRECTORY)
        print(f"Created directory {SOURCE_DIRECTORY}. Please put your files there and run this again.")
        return

    documents = load_documents(SOURCE_DIRECTORY)
    
    if not documents:
        print("No documents found to ingest.")
        return

    print(f"Loaded {len(documents)} documents.")

    # 2. Split Text
    # We split text into smaller chunks so the AI can find specific details easily
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    texts = text_splitter.split_documents(documents)
    
    print(f"Split into {len(texts)} chunks of text.")

    # 3. Create Embeddings & Store in DB
    print("Creating embeddings and saving to ChromaDB (this may take a while)...")
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    # This creates the database on disk
    db = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    
    # db.persist() is automatic in newer versions of Chroma, but good to know
    print(f"Success! Data saved to {PERSIST_DIRECTORY}")

if __name__ == "__main__":
    main()
