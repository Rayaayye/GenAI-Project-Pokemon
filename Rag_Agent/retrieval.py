import os 
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import streamlit as st

#Define paths since we had bugs with them before
BASE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

CHROMA_DIRECTORY = os.path.join(
    BASE_DIRECTORY,
    "..",
    "data",
    "chroma_pokedex"
)

# Function that load the Chroma vector database and returns a retriever for semantic search
@st.cache_resource
def get_chroma_db(k: int = 8):
    """Get Chroma DB retriever with caching to improve performance."""
    # Initialize the HuggingFace embedding model for semantic search
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-miniLM-L6-v2"
    )

    # Load the Chroma database from the directory with embeddings
    chroma_db = Chroma(
        persist_directory=CHROMA_DIRECTORY,
        embedding_function=embeddings,
    )
    # Return a retriever that returns the top-k most relevant documents
    return chroma_db.as_retriever(search_kwargs={"k":k})