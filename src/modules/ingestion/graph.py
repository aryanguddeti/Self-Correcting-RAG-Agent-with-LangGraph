from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from src.modules.ingestion.state import IngestionState
from src.modules.ingestion.ingestion_pipeline import (
    load_documents_node,
    split_and_summarize_node,
    embed_chunks_node,
    index_to_vectorstore_node,
    grade_load_quality,
    grade_chunk_quality,
    grade_embedding_quality,
    verify_index,
)

from pathlib import Path
from src.core.config import settings

def build_ingestion_graph() -> CompiledStateGraph:
    ingest_graph = StateGraph(IngestionState)

    # Nodes
    ingest_graph.add_node("load_documents", load_documents_node)
    ingest_graph.add_node("split_and_summarize", split_and_summarize_node)
    ingest_graph.add_node("embed_chunks", embed_chunks_node)
    ingest_graph.add_node("index_to_vectorstore", index_to_vectorstore_node)

    # Entry point
    ingest_graph.set_entry_point("load_documents")

    # Conditional edge: after load → grade quality
    ingest_graph.add_conditional_edges(
        "load_documents",
        grade_load_quality,
        {
            "load_documents": "load_documents",
            "split_and_summarize": "split_and_summarize",
            "skip_and_proceed": END,
        }
    )

    # Conditional edge: after split → grade chunk quality
    ingest_graph.add_conditional_edges(
        "split_and_summarize",
        grade_chunk_quality,
        {
            "split_and_summarize": "split_and_summarize",
            "embed_chunks": "embed_chunks",
            "failed_exit": END,
        }
    )

    # Conditional edge: after embed → grade embedding quality
    ingest_graph.add_conditional_edges(
        "embed_chunks",
        grade_embedding_quality,
        {
            "embed_chunks": "embed_chunks",
            "index_to_vectorstore": "index_to_vectorstore",
            "failed_exit": END,
        }
    )

    # Conditional edge: after index → verify
    ingest_graph.add_conditional_edges(
        "index_to_vectorstore",
        verify_index,
        {
            "index_to_vectorstore": "index_to_vectorstore",
            "end": END,
            "failed_exit": END,
        }
    )

    return ingest_graph.compile()

if __name__ == "__main__":
    initial_state = IngestionState(
        input_dir=Path(settings.DATA_INGEST_PATH.resolve()),
        file_paths=[],
        loaded_docs=[],
        chunks=[],
        embedded_chunks=[],
        failed_files=[],
        file_retries={},
        load_retries=0,
        chunk_retries=0,
        embed_retries=0,
        index_retries=0,
        status="INIT"
    )

    graph = build_ingestion_graph()

    for event in graph.stream(initial_state):
        print(event)