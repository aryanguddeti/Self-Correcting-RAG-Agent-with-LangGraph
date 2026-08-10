from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.shared.schema import ChunkMetadata
from src.modules.indexing.embedder import Embedder
from typing import List

class Indexer:
    def __init__(self):
        self.pc = Pinecone()
        self.index_name = "self-correcting-rag"
        self._ensure_index_exists()
        self.index = self.pc.Index(self.index_name)
        self.vector_store = PineconeVectorStore(index=self.index, embedding=Embedder().embeddings)

    def _ensure_index_exists(self):
        if not self.pc.has_index(self.index_name):
            self.pc.create_index(
                name=self.index_name,
                vector_type="dense",
                dimension=3072,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                ),
                deletion_protection="enabled",
                tags={
                    "environment": "development"
                }
            )
            print(f" Pinecone index '{self.index_name}' is ready.")


    def insert_vector_batch(self, chunks: List[ChunkMetadata]):
        if not chunks:
            print("No chunks to insert.")
            return

        # Prepare documents with metadata for Pinecone
        vectors = [
            (chunk.chunk_id, chunk.embedding, chunk.metadata)
            for chunk in chunks
            if chunk.embedding is not None
        ]

        # Insert into Pinecone
        try:
            self.index.upsert(vectors=vectors, batch_size=100, show_progress=True)
            print(f"Successfully inserted {len(chunks)} chunks into Pinecone.")
        except Exception as e:
            print(f"Error inserting chunks into Pinecone: {e}")


    def verify_vectors(self, chunk_ids: List[str], sample_size: int = 50) -> bool:
        """Spot-checks a sample of chunk IDs against Pinecone to confirm they were persisted."""
        sample = chunk_ids[:sample_size]
        try:
            response = self.index.fetch(ids=sample)
            found = len(response.vectors)
            if found < len(sample):
                print(f" Verification failed: expected {len(sample)} vectors, found {found} in Pinecone.")
                return False
            print(f" Verification passed: {found}/{len(sample)} sampled vectors confirmed in Pinecone.")
            return True
        except Exception as e:
            print(f" Pinecone fetch verification failed: {e}")
            return False

    def delete_by_id(self, vector_ids: List[str]) -> None:
        try:
            num_ids = len(vector_ids)
            if num_ids < 1000:
                self.index.delete(ids=vector_ids)
            else:
                for i in range(0, num_ids, 1000):
                    batch = vector_ids[i:i + 1000]
                    self.index.delete(ids=batch)

            print(f"Successfully deleted vectors with IDs: {vector_ids}")
        except Exception as e:
            print(f"Error deleting vectors from Pinecone: {e}")
            raise


