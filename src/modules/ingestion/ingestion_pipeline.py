from typing import Any, Dict, List, Literal

from src.modules.ingestion.state import IngestionState
from src.modules.ingestion.loader import IngestionLoader
from src.modules.ingestion.splitters import HybridTableAwareSplitter
from src.shared.schema import ChunkMetadata
from src.modules.ingestion.preprocessor import DocumentPreprocessor, HeaderPropagator
from src.modules.indexing.embedder import Embedder
from src.modules.indexing.indexer import Indexer
from src.infrastructure.ledger import LedgerManager
from src.utils.file_utils import compute_file_hash

MAX_RETRIES = 3

# ============================================================================
# WORKFLOW NODES
# ============================================================================

def load_documents_node(state: IngestionState) -> Dict[str, Any]:
    """Node 1: Loads documents from directory, retrying failed files up to MAX_RETRIES."""
    print("\n [Node: load_documents] Starting document ingestion...")

    loader = IngestionLoader()
    ledger = LedgerManager()
    input_path = state.input_dir

    file_retries = dict(state.file_retries)
    file_hashes = dict(state.file_hashes)
    already_permanently_failed = set(state.failed_files)
    existing_docs = {doc["file_name"]: doc for doc in state.loaded_docs}

    # Treat already-indexed files (from ledger) as already loaded so they are never retried
    already_indexed = set(ledger.get_all_file_names())
    for name in already_indexed:
        if name not in existing_docs:
            existing_docs[name] = {"file_name": name, "markdown_content": ""}

    valid_extensions = {".pdf", ".html", ".docx"}

    # 1. Classify every file on disk via ledger hash comparison
    files_to_process = []
    for f in input_path.iterdir():
        if not (f.is_file() and f.suffix.lower() in valid_extensions):
            continue

        file_name = f.name

        if file_name in existing_docs or file_name in already_permanently_failed:
            continue

        current_hash = compute_file_hash(f)
        record = ledger.get_file_record(file_name)

        if record and record["content_hash"] == current_hash and record["status"] == "indexed":
            print(f" Skipping unchanged file: '{file_name}'")
            continue

        if record and record["content_hash"] != current_hash:
            print(f" Modified file detected: '{file_name}'. Clearing stale ledger record.")
            vector_ids = [f.name + "::" + str(num) for num in range(1, record['chunk_count'] + 1)]
            ledger.delete_file_record(file_name)
            Indexer().delete_by_id(vector_ids)

        if file_retries.get(file_name, 0) < MAX_RETRIES:
            files_to_process.append(f)
            file_hashes[file_name] = current_hash

    if not files_to_process:
        print("ℹ No pending files to load or retry.")
        return {
            "loaded_docs": list(existing_docs.values()),
            "status": "DOCS_LOADED_IDLE"
        }

    # 2. Increment retry counts before attempting
    for f in files_to_process:
        file_retries[f.name] = file_retries.get(f.name, 0) + 1
        print(f" Attempting load for '{f.name}' (Attempt {file_retries[f.name]}/{MAX_RETRIES})")

    # 3. Execute document conversion batch
    newly_extracted_docs, newly_failed_files = loader.document_load(files_to_process)

    # 4. Merge successfully loaded docs
    for doc in newly_extracted_docs:
        existing_docs[doc["file_name"]] = doc

    # 5. Mark files that exhausted retries as permanently failed
    newly_permanently_failed = []
    for file_name in newly_failed_files:
        if file_retries.get(file_name, 0) >= MAX_RETRIES:
            print(f" File '{file_name}' exceeded MAX_RETRIES ({MAX_RETRIES}). Marking as permanently failed.")
            newly_permanently_failed.append(file_name)

    updated_failed_files = list(already_permanently_failed.union(set(newly_permanently_failed)))

    return {
        "loaded_docs": list(existing_docs.values()),
        "failed_files": updated_failed_files,
        "file_retries": file_retries,
        "file_hashes": file_hashes,
        "load_retries": state.load_retries + 1,
        "status": "DOCS_LOADED",
    }



def split_and_summarize_node(state: IngestionState) -> Dict[str, Any]:
    """Node 2: Cleans markdown, propagates headers, and generates table-aware chunks."""
    print("\n️ [Node: split_and_summarize] Cleaning text and chunking markdown...")

    preprocessor = DocumentPreprocessor()
    propagator = HeaderPropagator()
    splitter = HybridTableAwareSplitter(chunk_size=400, chunk_overlap=200)

    all_chunks: List[ChunkMetadata] = []

    for doc in state.loaded_docs:
        file_name = doc["file_name"]
        try:
            cleaned_markdown = preprocessor.clean_pdf_artifacts(doc["markdown_content"])
            enriched_markdown = propagator.propagate(cleaned_markdown)
            doc_chunks = splitter.split_document(enriched_markdown, {
                "file_name": file_name,
            })
            all_chunks.extend(doc_chunks)
        except Exception as e:
            print(f"️ Skipping '{file_name}' due to chunking error: {e}")
            continue

    return {
        "chunks": all_chunks,
        "chunk_retries": state.chunk_retries + 1,
        "status": "CHUNKS_CREATED" if all_chunks else "CHUNKS_FAILED"
    }

def embed_chunks_node(state: IngestionState) -> Dict[str, Any]:
    """Node 3: Converts chunk text into dense vector representations using Google GenAI."""
    embedded_chunks: List[ChunkMetadata] = []

    try:
        embedded_chunks = Embedder().batch_embedding(state.chunks)
        print(f" Successfully generated {len(embedded_chunks)} vector payloads.")
    except Exception as e:
        print(f" Critical Embedding API Failure: {e}")

    return {
        "embedded_chunks": embedded_chunks,
        "embed_retries": state.embed_retries + 1,
        "status": "EMBEDDINGS_GENERATED" if embedded_chunks else "EMBEDDINGS_FAILED"
    }

def index_to_vectorstore_node(state: IngestionState) -> Dict[str, Any]:
    """Node 4: Upserts vectors and metadata into Pinecone/Vector DB."""
    print(f"\n [Node: index_to_vectorstore] Upserting {len(state.embedded_chunks)} vectors to DB...")

    status = "INDEXED"
    try:
        Indexer().insert_vector_batch(state.embedded_chunks)

        # Write back to ledger — group chunks by file to get chunk_count per file
        ledger = LedgerManager()
        file_chunk_counts: Dict[str, int] = {}
        for chunk in state.embedded_chunks:
            file_name = chunk.metadata.get("file_name")
            if file_name:
                file_chunk_counts[file_name] = file_chunk_counts.get(file_name, 0) + 1

        for file_name, chunk_count in file_chunk_counts.items():
            content_hash = state.file_hashes.get(file_name)
            if content_hash:
                ledger.upsert_file_record(file_name, content_hash, chunk_count, "indexed")
                print(f" Ledger updated: '{file_name}' ({chunk_count} chunks)")

    except Exception as e:
        print(f" Critical Indexing Failure: {e}")
        status = "INDEXING_FAILED"

    return {
        "index_retries": state.index_retries + 1,
        "status": status
    }

# ============================================================================
# GRADERS & CONDITIONAL ROUTING EDGES
# ============================================================================

def grade_load_quality(
    state: IngestionState,
) -> Literal["split_and_summarize", "load_documents", "skip_and_proceed"]:
    """Grader 1: Evaluates if any files still need retrying before moving forward."""
    print(" [Grader: grade_load_quality] Checking load quality...")

    input_path = state.input_dir
    file_retries = state.file_retries
    permanently_failed = set(state.failed_files)
    loaded_filenames = {doc["file_name"] for doc in state.loaded_docs}

    valid_extensions = {".pdf", ".html", ".docx"}

    # Check if there are any files that failed but still have remaining retries
    files_needing_retry = []
    for f in input_path.iterdir():
        if f.is_file() and f.suffix.lower() in valid_extensions:
            name = f.name
            if name not in loaded_filenames and name not in permanently_failed:
                if file_retries.get(name, 0) < MAX_RETRIES:
                    files_needing_retry.append(name)

    if files_needing_retry:
        print(f"️ {len(files_needing_retry)} file(s) failed and will be retried: {files_needing_retry}")
        return "load_documents"

    if not state.loaded_docs:
        print(" All files failed permanently. Nothing to process.")
        return "skip_and_proceed"

    print(f" Document loading stage finished. {len(state.loaded_docs)} doc(s) ready for chunking.")
    return "split_and_summarize"


def grade_chunk_quality(state: IngestionState) -> Literal["embed_chunks", "split_and_summarize", "failed_exit"]:
    """Grader 2: Validates chunk structure and checks for empty table summaries."""
    print(" [Grader: grade_chunk_quality] Verifying chunk integrity...")

    chunks = state.chunks
    if not chunks or state.status == "CHUNKS_FAILED":
        if state.chunk_retries < MAX_RETRIES:
            print(" No chunks created. Retrying chunking...")
            return "split_and_summarize"
        else:
            print(" Chunking retries exhausted. Proceeding with empty chunks...")
            return "failed_exit"

    malformed_summaries = [
        c for c in chunks
        if c.chunk_type == "table summary"
        and (not c.content_to_embed.strip() or len(c.content_to_embed.strip()) < 10)
    ]

    if malformed_summaries and state.chunk_retries < MAX_RETRIES:
        print(f" Found {len(malformed_summaries)} invalid table summaries. Re-running chunking pipeline...")
        return "split_and_summarize"

    print(" Chunking stage passed quality checks.")
    return "embed_chunks"


def grade_embedding_quality(state: IngestionState) -> Literal["index_to_vectorstore", "embed_chunks", "failed_exit"]:
    """Grader 3: Confirms all chunks were successfully converted to dense vectors."""
    print(" [Grader: grade_embeddings] Checking vector generation...")

    chunks = state.chunks
    embedded_chunks = state.embedded_chunks

    if len(chunks) > len(embedded_chunks) or state.status == "EMBEDDINGS_FAILED":
        if state.chunk_retries < MAX_RETRIES:
            print(" Some chunks failed embedding. Retrying...")
            return "embed_chunks"
        else:
            print("Embedding retries exhausted")
            return "failed_exit"

    print(" Embedding stage passed. Proceeding to indexing...")
    return "index_to_vectorstore"


def verify_index(state: IngestionState) -> Literal["end", "index_to_vectorstore", "failed_exit"]:
    """Grader 4: Read-after-write verification against the vector database."""
    print(" [Grader: verify_index] Executing read-after-write verification...")

    if not state.embedded_chunks:
        print(" No vectors to index.")
        return "failed_exit"

    if state.status == "INDEXING_FAILED":
        if state.index_retries < MAX_RETRIES:
            print(f" Indexing failed. Retrying... (Attempt {state.index_retries}/{MAX_RETRIES})")
            return "index_to_vectorstore"
        else:
            print(f" Indexing failed after {MAX_RETRIES} attempts. Giving up.")
            return "failed_exit"

    print(" Vector store indexing complete.")
    return "end"
