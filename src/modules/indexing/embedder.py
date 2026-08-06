from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.shared.schema import ChunkMetadata
from typing import List
import time

class Embedder:
    def __init__(self, batch_size: int = 100, rate_limit_delay: float = 0.5):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
        self.batch_size = batch_size
        self.rate_limit_delay = rate_limit_delay

    def batch_embedding(self, chunks: List[ChunkMetadata]) -> List[ChunkMetadata]:
        if not chunks:
            return []

        batches = [chunks[i: i + self.batch_size] for i in range(0, len(chunks), self.batch_size)]
        embedded_chunks: List[ChunkMetadata] = []

        for idx, batch in enumerate(batches):
            try:
                vectors = self.embeddings.embed_documents([chunk.content_to_embed for chunk in batch])
                embedded_chunks.extend(chunk.model_copy(update={"embedding": v}) for chunk, v in zip(batch, vectors))
            except Exception as e:
                print(f"⚠️ Skipping batch {idx + 1}/{len(batches)} due to embedding error: {e}")
            finally:
                if idx < len(batches) - 1:
                    time.sleep(self.rate_limit_delay)

        return embedded_chunks


