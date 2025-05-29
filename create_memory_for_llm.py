import os
from pathlib import Path
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

## Uncomment the following files if you're not using pipenv as your virtual environment manager
#from dotenv import load_dotenv, find_dotenv
#load_dotenv(find_dotenv())

# Constants
DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

def ensure_directory_exists(directory_path: str) -> None:
    """Create directory if it doesn't exist."""
    Path(directory_path).mkdir(parents=True, exist_ok=True)

def load_single_pdf(pdf_path: str) -> List:
    """Load a single PDF file."""
    try:
        loader = PyPDFLoader(pdf_path)
        return loader.load()
    except Exception as e:
        print(f"Error loading PDF {pdf_path}: {str(e)}")
        return []

def load_pdf_files(data_path: str = DATA_PATH) -> List:
    """
    Load PDF files from the specified directory.
    Returns a list of documents.
    """
    ensure_directory_exists(data_path)
    
    try:
        # Use DirectoryLoader to load all PDFs in the directory
        loader = DirectoryLoader(data_path,
                               glob='*.pdf',
                               loader_cls=PyPDFLoader)
        
        documents = loader.load()
        print(f"Successfully loaded {len(documents)} pages from PDFs in {data_path}")
        return documents
    except Exception as e:
        print(f"Error loading PDFs from directory: {str(e)}")
        return []

def create_chunks(extracted_data: List, 
                 chunk_size: int = 500,
                 chunk_overlap: int = 50) -> List:
    """Create text chunks from the documents."""
    if not extracted_data:
        print("No documents to process!")
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    print(f"Created {len(text_chunks)} text chunks")
    return text_chunks

def get_embedding_model():
    """Get the embedding model."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def update_or_create_vectorstore(documents: List,
                               db_path: str = DB_FAISS_PATH) -> Optional[FAISS]:
    """Update existing vector store or create a new one."""
    if not documents:
        print("No documents to process!")
        return None

    ensure_directory_exists(os.path.dirname(db_path))
    embedding_model = get_embedding_model()
    
    try:
        text_chunks = create_chunks(documents)
        
        # If vector store exists, load and update it
        if os.path.exists(db_path):
            print(f"Updating existing vector store at {db_path}")
            db = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)
            db.add_documents(text_chunks)
        else:
            print(f"Creating new vector store at {db_path}")
            db = FAISS.from_documents(text_chunks, embedding_model)
        
        db.save_local(db_path)
        print("Vector store successfully updated!")
        return db
    except Exception as e:
        print(f"Error updating vector store: {str(e)}")
        return None

def main():
    """Main function to process PDFs and update vector store."""
    print(f"Looking for PDFs in {DATA_PATH}")
    
    # Ensure data directory exists
    ensure_directory_exists(DATA_PATH)
    
    # Load documents
    documents = load_pdf_files()
    if not documents:
        print(f"No PDFs found in {DATA_PATH}. Please add PDFs and try again.")
        return
    
    # Update or create vector store
    db = update_or_create_vectorstore(documents)
    if db:
        print(f"Successfully processed PDFs and updated vector store at {DB_FAISS_PATH}")
    else:
        print("Failed to process PDFs and update vector store")

if __name__ == "__main__":
    main()