import os
from dotenv import load_dotenv

load_dotenv()
os.environ["USER_AGENT"] = "MyAgentApp/1.0"

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_tavily import TavilySearch


def setup_retriever(urls: list):
    """
    Loads web pages, splits them into chunks, and stores them in a local ChromaDB.
    """
    print("---SETTING UP VECTOR DATABASE---")

    # 1. Load data
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    # 2. Split data into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)

    # 3. Create Vector Store (Using local HuggingFace Embeddings)
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
    )

    return vectorstore.as_retriever(search_kwargs={"k": 2})


# Initialize the Web Search Tool (Tavily)
web_search_tool = TavilySearch(k=3)