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


    def delete_by_id(self, vector_ids: List[str]) -> None:
        try:
            self.index.delete(ids=vector_ids)
            print(f"Successfully deleted vectors with IDs: {vector_ids}")
        except Exception as e:
            print(f"Error deleting vectors from Pinecone: {e}")



