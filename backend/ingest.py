"""
Memory Builder Script (ingest.py)

This script ingests documents (PDF, TXT, MD) and stores them in ChromaDB
for the Jarvis agent to retrieve and use in conversations.

Usage:
    python ingest.py

Make sure to place your documents in: backend/data/source_documents/
"""

import os
import glob
from typing import List

# Try to import langchain and chroma dependencies (optional)
try:
    from langchain_community.document_loaders import (
        PyPDFLoader,
        TextLoader,
        UnstructuredMarkdownLoader,
    )
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    print(f"[WARNING] LangChain dependencies not available: {e}")
    print("[INFO] This script requires: pip install langchain langchain-community langchain-huggingface")

# --- Configuration ---
BACKEND_DIR = os.path.dirname(__file__)
SOURCE_DIRECTORY = os.path.join(BACKEND_DIR, "data", "source_documents")
PERSIST_DIRECTORY = os.path.join(BACKEND_DIR, "data", "chroma_db")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def load_documents(source_dir: str) -> List:
    """
    Loads all .pdf, .txt, and .md files from the source directory.
    
    Args:
        source_dir: Path to directory containing documents
        
    Returns:
        List of loaded documents
    """
    if not LANGCHAIN_AVAILABLE:
        print("[ERROR] LangChain is not installed. Cannot load documents.")
        return []
    
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
                print(f"  ✓ Loading: {os.path.basename(file_path)}")
                loader = loader_cls(file_path)
                docs = loader.load()
                documents.extend(docs)
                print(f"    → {len(docs)} pages/sections loaded")
            except Exception as e:
                print(f"  ✗ Error loading {file_path}: {e}")

    return documents


def split_documents(documents: List) -> List:
    """
    Splits documents into smaller chunks for better retrieval.
    
    Args:
        documents: List of documents
        
    Returns:
        List of text chunks
    """
    if not LANGCHAIN_AVAILABLE:
        return []
    
    # Split text into smaller chunks so the AI can find specific details easily
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    texts = text_splitter.split_documents(documents)
    return texts


def create_embeddings_and_store(texts: List) -> bool:
    """
    Creates embeddings and stores them in ChromaDB.
    
    Args:
        texts: List of text chunks
        
    Returns:
        True if successful, False otherwise
    """
    if not LANGCHAIN_AVAILABLE:
        print("[ERROR] LangChain is not installed. Cannot create embeddings.")
        return False
    
    try:
        print(f"Creating embeddings using '{EMBEDDING_MODEL_NAME}' model...")
        print("(This may take a few minutes on first run...)")
        
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        
        print(f"Storing {len(texts)} chunks in ChromaDB...")
        # This creates the database on disk
        db = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=PERSIST_DIRECTORY
        )
        
        print(f"✓ Success! Data saved to {PERSIST_DIRECTORY}")
        return True
    except Exception as e:
        print(f"✗ Error creating embeddings: {e}")
        return False


def main():
    """Main ingestion pipeline."""
    print("="*60)
    print("PADCOM Memory Builder - Document Ingestion Script")
    print("="*60)
    
    if not LANGCHAIN_AVAILABLE:
        print("\n[ERROR] LangChain is required to run this script.")
        print("\nInstall with:")
        print("  pip install langchain langchain-community langchain-huggingface")
        return
    
    # 1. Create directories if they don't exist
    if not os.path.exists(SOURCE_DIRECTORY):
        os.makedirs(SOURCE_DIRECTORY)
        print(f"\n✓ Created directory: {SOURCE_DIRECTORY}")
        print("  → Please place your PDF, TXT, or MD files here")
        print("  → Then run this script again")
        return
    
    # 2. Load Documents
    print(f"\nStep 1: Loading documents from {SOURCE_DIRECTORY}...\n")
    documents = load_documents(SOURCE_DIRECTORY)
    
    if not documents:
        print("\n[INFO] No documents found.")
        print(f"  → Place PDF, TXT, or MD files in: {SOURCE_DIRECTORY}")
        return

    print(f"\n✓ Loaded {len(documents)} documents\n")

    # 3. Split Text
    print("Step 2: Splitting documents into chunks...\n")
    texts = split_documents(documents)
    
    if not texts:
        print("[ERROR] Failed to split documents")
        return
    
    print(f"✓ Split into {len(texts)} chunks of text\n")

    # 4. Create Embeddings & Store in DB
    print("Step 3: Creating embeddings and storing in ChromaDB...\n")
    
    success = create_embeddings_and_store(texts)
    
    if success:
        print("\n" + "="*60)
        print("✓ MEMORY BUILDER COMPLETE!")
        print("="*60)
        print(f"Database location: {PERSIST_DIRECTORY}")
        print("\nJarvis can now access this knowledge in conversations.")
        print("="*60)
    else:
        print("\n[ERROR] Ingestion failed. Please check the errors above.")


if __name__ == "__main__":
    main()
