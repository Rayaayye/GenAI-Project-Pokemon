import os 
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# We had bugs with paths before so we did that to not have any problems when running the project

#Define paths

BASE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

CHROMA_DIRECTORY = os.path.join(
    BASE_DIRECTORY,
    "..",
    "data",
    "chroma_pokedex"
)

# Function that load the Chroma vector database and returns a retriever for semantic search
def get_chroma_db(k: int = 8):
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