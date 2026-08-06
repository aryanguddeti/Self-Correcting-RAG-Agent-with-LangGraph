import re
import uuid
from typing import List, Dict, Any, Union
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.llm import get_llm
from src.shared.prompts import TABLE_SUMMARIZATION_PROMPT
from src.shared.schema import ChunkMetadata

class HybridTableAwareSplitter:

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        self.table_pattern = re.compile(
            r"(?:^\s*\|[^\n]+\|\s*\n\s*\|[\s\:\|\-]*\|\s*\n(?:^\s*\|[^\n]+\|\s*\n?)+)",
            re.MULTILINE
        )

    def extract_headers(self, table_text: str) -> List[str]:
        lines = table_text.strip().split("\n")
        if len(lines) >= 1:
            headers = [h.strip() for h in lines[0].split("|") if h.strip()]
            return headers
        return []

    def _parse_response_content(self, content: Union[str, list]) -> str:
        """Helper to extract clean text whether Gemini returns a string or block list."""
        if isinstance(content, list):
            return "".join(
                block.get("text", "") for block in content if isinstance(block, dict)
            ).strip()
        return str(content).strip()

    def summarize_tables_batch(self, tables: List[str]) -> List[str]:
        """Summarizes all extracted tables in parallel using batch invocation."""
        if not tables:
            return []

        llm = get_llm("summarization")
        chain = TABLE_SUMMARIZATION_PROMPT | llm

        # 1. Build the list of prompt inputs
        batch_inputs = [{"table_content": table} for table in tables]

        print(f"⚡ Batch processing {len(tables)} table summaries with LLM...")

        # 2. Call batch with concurrency control to avoid hitting 429 rate limits
        responses = chain.batch(
            batch_inputs,
            config={"max_concurrency": 5}  # Adjust based on your API quota
        )

        # 3. Parse and return extracted text summaries in matching order
        return [self._parse_response_content(resp.content) for resp in responses]

    @staticmethod
    def _extract_section_headers(text: str) -> str:
        """Extracts the breadcrumb injected by HeaderPropagator, if present."""
        match = re.search(r"\*\*\[Context: ([^\]]+)\]\*\*", text)
        return match.group(1) if match else ""

    @staticmethod
    def _has_code_block(text: str) -> bool:
        return bool(re.search(r"```", text))

    def split_document(
            self,
            cleaned_markdown: str,
            base_metadata: Dict[str, Any]
    ) -> List[ChunkMetadata]:

        chunks: List[ChunkMetadata] = []

        # Find all Markdown tables and their character spans
        table_matches = list(self.table_pattern.finditer(cleaned_markdown))
        print(f"📊 Found {len(table_matches)} tables in document.")

        # 1. Collect all raw tables and batch-generate all summaries up front
        raw_tables = [match.group(0).strip() for match in table_matches]
        table_summaries = self.summarize_tables_batch(raw_tables)

        last_idx = 0

        # 2. Build narrative and table chunks sequentially
        for i, match in enumerate(table_matches):
            start_idx, end_idx = match.span()

            # Process narrative text BEFORE the table
            if start_idx > last_idx:
                text_block = cleaned_markdown[last_idx:start_idx].strip()
                if text_block:
                    text_chunks = self.text_splitter.split_text(text_block)
                    for text_chunk in text_chunks:
                        chunks.append(
                            ChunkMetadata(
                                chunk_id=str(uuid.uuid4()),
                                chunk_type="text",
                                content_to_embed=text_chunk,
                                raw_payload=text_chunk,
                                metadata={
                                    **base_metadata,
                                    "has_table": False,
                                    "section_headers": self._extract_section_headers(text_chunk),
                                    "has_code_block": self._has_code_block(text_chunk),
                                }
                            )
                        )

            # Process table chunk using the pre-fetched summary at index i
            raw_table = raw_tables[i]
            summary = table_summaries[i]
            headers = self.extract_headers(raw_table)

            chunks.append(
                ChunkMetadata(
                    chunk_id=str(uuid.uuid4()),
                    chunk_type="table summary",
                    content_to_embed=summary,
                    raw_payload=raw_table,
                    metadata={
                        **base_metadata,
                        "has_table": True,
                        "table_columns": headers,
                        "is_table_summary": True,
                        "section_headers": self._extract_section_headers(cleaned_markdown[max(0, table_matches[i].start()-500):table_matches[i].start()]),
                        "has_code_block": False,
                    }
                )
            )

            last_idx = end_idx

        # 3. Process remaining narrative text AFTER the last table
        if last_idx < len(cleaned_markdown):
            text_block = cleaned_markdown[last_idx:].strip()
            if text_block:
                text_chunks = self.text_splitter.split_text(text_block)
                for text_chunk in text_chunks:
                    chunks.append(
                        ChunkMetadata(
                            chunk_id=str(uuid.uuid4()),
                            chunk_type="text",
                            content_to_embed=text_chunk,
                            raw_payload=text_chunk,
                            metadata={
                                **base_metadata,
                                "has_table": False,
                                "section_headers": self._extract_section_headers(text_chunk),
                                "has_code_block": self._has_code_block(text_chunk),
                            }
                        )
                    )

        return chunks