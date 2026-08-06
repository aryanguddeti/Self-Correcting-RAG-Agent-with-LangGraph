from typing import Any, Dict, List
from pathlib import Path
from pydantic import BaseModel, Field

from src.shared.schema import ChunkMetadata


class IngestionState(BaseModel):
    input_dir: Path
    file_paths: List[Path] = Field(default_factory=list)
    loaded_docs: List[Dict[str, Any]] = Field(default_factory=list)
    chunks: List[ChunkMetadata] = Field(default_factory=list)
    embedded_chunks: List[ChunkMetadata] = Field(default_factory=list)
    failed_files: List[str] = Field(default_factory=list)
    file_retries: Dict[str, int] = Field(default_factory=dict)
    load_retries: int = 0
    chunk_retries: int = 0
    embed_retries: int = 0
    index_retries: int = 0
    status: str = "INIT"


