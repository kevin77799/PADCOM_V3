# Memory Builder Guide - PADCOM Jarvis

## Overview
The Memory Builder (ingest.py) allows Jarvis to read and remember your documents, PDFs, and notes. It converts them into embeddings that Jarvis can quickly search through.

## Folder Structure

```
backend/
├── data/
│   ├── source_documents/     ← Place your files here
│   └── chroma_db/            ← Auto-generated database
├── ingest.py                 ← The ingestion script
└── ...other files
```

## Step 1: Add Your Documents

Place any of these file types in `backend/data/source_documents/`:
- **PDF files** (.pdf)
- **Text files** (.txt)
- **Markdown files** (.md)

Examples:
- `my_notes.txt`
- `research_paper.pdf`
- `personal_info.md`

## Step 2: Run the Ingestion Script

### Prerequisites
Make sure LangChain dependencies are installed:

```bash
pip install langchain langchain-community langchain-huggingface
```

### Run the Script

Open a terminal in the `backend/` folder and run:

```bash
python ingest.py
```

### What It Does

1. **Scans** `source_documents/` for all PDF, TXT, and MD files
2. **Loads** the content from each file
3. **Splits** large documents into 1000-character chunks (with 200-char overlap for context)
4. **Embeds** each chunk using HuggingFace's "all-MiniLM-L6-v2" model
5. **Stores** embeddings in ChromaDB at `backend/data/chroma_db/`

### Example Output

```
============================================================
PADCOM Memory Builder - Document Ingestion Script
============================================================

Step 1: Loading documents from .../backend/data/source_documents...

  ✓ Loading: notes.txt
    → 2 pages/sections loaded
  ✓ Loading: research.pdf
    → 15 pages/sections loaded

✓ Loaded 17 documents

Step 2: Splitting documents into chunks...

✓ Split into 85 chunks of text

Step 3: Creating embeddings and storing in ChromaDB...

Creating embeddings using 'all-MiniLM-L6-v2' model...
(This may take a few minutes on first run...)
Storing 85 chunks in ChromaDB...
✓ Success! Data saved to .../backend/data/chroma_db

============================================================
✓ MEMORY BUILDER COMPLETE!
============================================================
Database location: .../backend/data/chroma_db
Jarvis can now access this knowledge in conversations.
============================================================
```

## How Jarvis Uses Your Data

When you chat with Jarvis:

1. Your question is converted to an embedding
2. ChromaDB searches for the **most similar** chunks from your documents
3. Those chunks are passed to Jarvis as "context"
4. Jarvis answers based on your documents + general knowledge

## Notes

- **First run is slower**: The first time you run this, HuggingFace needs to download the embedding model (~100MB). Subsequent runs are faster.
- **Update your memory**: Want to add more documents? Just place them in `source_documents/` and run `ingest.py` again. It will create a new database.
- **Database size**: The ChromaDB folder size depends on how many documents you ingest.
- **Optional**: This feature requires LangChain. If not installed, Jarvis will still work but without document retrieval.

## Troubleshooting

**Issue**: "No documents found to ingest"
- Solution: Make sure your files are in `backend/data/source_documents/`

**Issue**: "LangChain is not installed"
- Solution: Run `pip install langchain langchain-community langchain-huggingface`

**Issue**: "Error loading PDF"
- Solution: The PDF might be corrupted or encrypted. Try a different PDF or convert it to TXT.

## Advanced: Manual Ingestion

If you want to ingest documents programmatically:

```python
from ingest import load_documents, split_documents, create_embeddings_and_store
import os

SOURCE_DIR = "backend/data/source_documents"
docs = load_documents(SOURCE_DIR)
texts = split_documents(docs)
create_embeddings_and_store(texts)
```

---

**Created for**: PADCOM Medical Imaging System
**Component**: Jarvis AI Agent - Memory System
