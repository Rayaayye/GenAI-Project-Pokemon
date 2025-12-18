import os 
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

BASE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

CHROMA_DIRECTORY = os.path.join(
    BASE_DIRECTORY,
    "..",
    "data",
    "chroma_pokedex"
)

def get_chroma_db(k: int = 8):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-miniLM-L6-v2"
    )

    chroma_db = Chroma(
        persist_directory=CHROMA_DIRECTORY,
        embedding_function=embeddings,
    )
    return chroma_db.as_retriever(search_kwargs={"k":k})