import re
import unicodedata
from typing import Dict, List
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine


class DocumentPreprocessor:
    """
    Handles PDF artifact cleaning, unicode normalization, whitespace
    correction, and PII redaction for RAG ingestion pipelines.
    """

    def __init__(self, target_entities: List[str] = None):
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        self.target_entities = target_entities or [
            "PERSON",
            "EMAIL_ADDRESS",
            "PHONE_NUMBER",
            "CREDIT_CARD",
            "US_SSN",
        ]

    def clean_pdf_artifacts(self, text: str) -> str:
        if not text:
            return ""
        patterns = [
            r"^(?!\s*\|.*\|\s*$)\W+$\n",
            r"(?i)^\s*(page\s+\d+\s+of\s+\d+|-\s*\d+\s*-)\s*$",
            r"!\[.*?\]\(.*?\)",
            r"<[^>]+>",
            r"\x00",
        ]
        for pattern in patterns:
            text = re.sub(pattern, "", text, flags=re.MULTILINE)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def normalize_text(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = text.replace("\u200b", "").replace("\xa0", " ").replace("\ufeff", "")
        ligatures = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl", "œ": "oe", "æ": "ae"}
        for search, replace in ligatures.items():
            text = text.replace(search, replace)
        text = re.sub(r"([a-zA-Z]+)-\n([a-zA-Z]+)", r"\1\2", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def redact_pii(self, text: str) -> str:
        if not text:
            return ""
        analyzer_results = self.analyzer.analyze(
            text=text, entities=self.target_entities, language="en"
        )
        anonymized_result = self.anonymizer.anonymize(
            text=text, analyzer_results=analyzer_results
        )
        return anonymized_result.text

    def process(self, raw_text: str) -> str:
        cleaned = self.clean_pdf_artifacts(raw_text)
        normalized = self.normalize_text(cleaned)
        return self.redact_pii(normalized)


class HeaderPropagator:
    """
    Prepends the active Markdown heading path to isolated paragraphs so they
    retain their context when split by a text chunker.
    """

    def __init__(self):
        self.header_pattern = re.compile(r"^(#{1,6})\s+(.*)")

    def propagate(self, markdown_text: str) -> str:
        lines = markdown_text.split("\n")
        current_headers: Dict[int, str] = {}
        enriched_blocks: List[str] = []
        current_block: List[str] = []

        for line in lines:
            match = self.header_pattern.match(line)
            if match:
                if current_block:
                    breadcrumb = self._build_breadcrumb(current_headers)
                    enriched_blocks.append(self._format_block(breadcrumb, current_block))
                    current_block = []
                level = len(match.group(1))
                title = match.group(2).strip()
                current_headers[level] = title
                current_headers = {k: v for k, v in current_headers.items() if k <= level}
                enriched_blocks.append(line.strip())
            else:
                if line.strip():
                    current_block.append(line)

        if current_block:
            breadcrumb = self._build_breadcrumb(current_headers)
            enriched_blocks.append(self._format_block(breadcrumb, current_block))

        return "\n\n".join(enriched_blocks)

    def _build_breadcrumb(self, headers: Dict[int, str]) -> str:
        if not headers:
            return ""
        return " > ".join([headers[lvl] for lvl in sorted(headers.keys())])

    def _format_block(self, breadcrumb: str, lines: List[str]) -> str:
        block_text = "\n".join(lines).strip()
        if breadcrumb and block_text:
            return f"**[Context: {breadcrumb}]**\n{block_text}"
        return block_text
