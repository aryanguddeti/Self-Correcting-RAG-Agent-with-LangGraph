import os
from pathlib import Path
from pprint import pprint

# Adjust these imports based on your exact file names and paths
from src.modules.ingestion.loader import IngestionLoader
from src.modules.ingestion.preprocessor import DocumentPreprocessor, HeaderPropagator
from src.modules.ingestion.splitters import HybridTableAwareSplitter


def test_ingestion_pipeline():
    print("🚀 Initializing Enterprise Ingestion Pipeline...")

    # 1. Initialize all components
    loader = IngestionLoader()
    preprocessor = DocumentPreprocessor()
    propagator = HeaderPropagator()
    splitter = HybridTableAwareSplitter(chunk_size=400, chunk_overlap=100)

    # 2. Define the test directory (ensure you have at least 1 PDF in this folder)
    # You can point this to settings.DATA_INGEST_PATH if configured
    test_data_path = Path("../data")

    if not test_data_path.exists():
        test_data_path.mkdir(parents=True, exist_ok=True)
        print(f"⚠️ Created dummy directory at {test_data_path}. Please add a test PDF and run again.")
        return

    print(f"\n📂 Loading documents from: {test_data_path}")
    loaded_docs = loader.document_load(test_data_path)

    if not loaded_docs:
        print("❌ No valid documents found or extraction failed.")
        return

    # 3. Process each document through the pipeline
    for doc in loaded_docs:
        file_name = doc["file_name"]
        raw_markdown = doc["markdown_content"]

        print(f"\n================ Processing: {file_name} ================")

        # Step A: Clean PDF Artifacts & PII
        print("🧹 Running DocumentPreprocessor (Artifact removal, Unicode fixes, PII Redaction)...")
        cleaned_markdown = preprocessor.process(raw_markdown)

        # Step B: Attach Markdown Header Breadcrumbs
        print("🔗 Running HeaderPropagator (Attaching contextual breadcrumbs)...")
        propagated_markdown = propagator.propagate(cleaned_markdown)

        # Define Base Metadata for this specific document
        base_metadata = {
            "source_file": file_name,
            "access_tier": "general"  # Example enterprise metadata
        }

        # Step C: Split Tables and Narrative Text
        print("✂️ Running HybridTableAwareSplitter (Chunking text and summarizing tables)...")
        chunks = splitter.split_document(propagated_markdown, base_metadata)

        # 4. Output Results
        print(f"\n✅ Pipeline Complete! Generated {len(chunks)} chunks.")
        print("-" * 60)

        # Print a summary of the generated chunks to verify the schema
        for i, chunk in enumerate(chunks):
            print(f"\n--- Chunk {i + 1} | Type: {chunk.chunk_type} ---")
            print(f"ID: {chunk.chunk_id}")
            print(f"Content to Embed (Preview): {chunk.content_to_embed[:150]}...")
            print(f"Metadata: {chunk.metadata}")

            # Print a snippet of the raw payload to confirm tables are intact
            if chunk.chunk_type == "table_summary":
                print(f"Raw Payload (Table Preview):\n{chunk.raw_payload[:150]}...")

        print("\n" + "=" * 60)


if __name__ == "__main__":
    # Ensure you have your OPENAI_API_KEY (or relevant LLM key) in your environment
    # because the splitter will call the LLM for table summarization!
    test_ingestion_pipeline()