from pydantic import BaseModel
from typing import Any, Dict, List, Literal, Optional


class ChunkMetadata(BaseModel):
    chunk_id: str
    chunk_type: Literal["text", "table summary"]
    content_to_embed: str
    raw_payload: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
