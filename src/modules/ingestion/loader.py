from docling.document_converter import DocumentConverter, PdfFormatOption, WordFormatOption, HTMLFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions, TableFormerMode, TesseractCliOcrOptions
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import pandas as pd
from src.core.config import settings

print(f"------------------Loading data from: {settings.DATA_INGEST_PATH.resolve()}------------------")

class IngestionLoader:
    def __init__(self):
        # PDF Format Options
        self.pdf_pipeline_options = ThreadedPdfPipelineOptions()
        self.pdf_pipeline_options.allow_external_plugins = True
        self.pdf_pipeline_options.do_ocr = True
        self.pdf_pipeline_options.ocr_options = TesseractCliOcrOptions(
            lang = ['eng'],
            force_full_page_ocr = False
        )

        self.pdf_pipeline_options.do_table_structure = True
        self.pdf_pipeline_options.table_structure_options.do_cell_matching = False
        self.pdf_pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

        #Initialize the loader
        self.loader = DocumentConverter(
            allowed_formats=[InputFormat.PDF, InputFormat.DOCX, InputFormat.HTML],
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=self.pdf_pipeline_options
                ),

                InputFormat.DOCX: WordFormatOption(),
                InputFormat.HTML: HTMLFormatOption(),
            }
        )

    def document_load(
        self, target: Union[Path, List[Path]]
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Converts files using Docling.

        Args:
            target: Either a directory Path or an explicit list of file Paths.

        Returns:
            Tuple of (extracted_docs, failed_files)
        """
        valid_extensions = {".pdf", ".html", ".docx"}

        # Resolve files to process
        if isinstance(target, Path):
            if target.is_dir():
                docs = [
                    f
                    for f in target.iterdir()
                    if f.is_file() and f.suffix.lower() in valid_extensions
                ]
            else:
                docs = [target] if target.suffix.lower() in valid_extensions else []
        else:
            docs = [
                f
                for f in target
                if f.is_file() and f.suffix.lower() in valid_extensions
            ]

        extracted_docs = []
        failed_files = []

        if not docs:
            print("⚠️ No valid documents found to load.")
            return extracted_docs, failed_files

        try:
            # convert_all accepts an iterable of files and processes them
            results = self.loader.convert_all(docs, raises_on_error=False)

            for result in results:
                file_name = result.input.file.name
                print(
                    f"\n================ Document: {file_name} ================"
                )

                # Check if conversion succeeded
                if result.status.name == "SUCCESS" and result.document:
                    markdown_content = result.document.export_to_markdown()

                    extracted_docs.append(
                        {
                            "file_name": file_name,
                            "markdown_content": markdown_content,
                        }
                    )
                    print(f"✅ Successfully converted: {file_name}")
                else:
                    print(
                        f"❌ Failed to convert: {file_name} | Status: {result.status.name}"
                    )
                    failed_files.append(file_name)

        except Exception as e:
            print(f"❌ Catastrophic error during document conversion: {e}")
            # If batch conversion crashes entirely, mark all target files as failed
            failed_files.extend([f.name for f in docs])

        return extracted_docs, failed_files

if __name__ == "__main__":
    loader = IngestionLoader()
    extracted_doc, failed = loader.document_load(Path(settings.DATA_INGEST_PATH.resolve()))
