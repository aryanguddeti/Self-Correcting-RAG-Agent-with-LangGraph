from typing import Any, Dict, List, TypedDict
from pathlib import Path
from pydantic import BaseModel

from src.shared.schema import ChunkMetadata


class IngestionState(BaseModel):
    input_dir: Path
    file_paths: List[Path]
    loaded_docs: List[Dict[str, Any]]
    chunks: List[ChunkMetadata]
    embedded_chunks: List[ChunkMetadata]
    failed_files: List[str]            # Files that failed permanently (exceeded MAX_RETRIES)
    file_retries: Dict[str, int]       # Track retry count per file: {"doc1.pdf": 2}
    load_retries: int
    chunk_retries: int
    embed_retries: int
    index_retries: int
    status: str


