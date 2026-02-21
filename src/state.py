from typing import List, TypedDict
from langchain_core.documents import Document

class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str  # "Yes" or "No"
    documents: List[Document] # Changed from List[str] to List[Document]