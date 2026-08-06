import traceback
from contextlib import contextmanager
from typing import Generator


class PipelineError(Exception):
    """Reusable structured exception for any pipeline stage failure."""

    def __init__(self, stage: str, file_name: str, original_error: Exception):
        self.stage = stage
        self.file_name = file_name
        self.original_error = original_error

        tb = traceback.extract_tb(original_error.__traceback__)
        if tb:
            last_frame = tb[-1]
            self.error_file = last_frame.filename
            self.line_number = last_frame.lineno
            self.function_name = last_frame.name
        else:
            self.error_file = "Unknown"
            self.line_number = 0
            self.function_name = "Unknown"

        formatted_message = (
            f"\n"
            f"============================================================"
            f"\n🚨 PIPELINE FAILURE"
            f"\n============================================================"
            f"\n  • Document:     {self.file_name}"
            f"\n  • Failed Stage: {self.stage}"
            f"\n  • File Location:{self.error_file}"
            f"\n  • Function:     {self.function_name}() [Line {self.line_number}]"
            f"\n  • Exception:    {type(original_error).__name__}"
            f"\n  • Details:      {str(original_error)}"
            f"\n============================================================\n"
        )
        super().__init__(formatted_message)


@contextmanager
def track_stage(stage_name: str, file_name: str) -> Generator[None, None, None]:
    try:
        yield
    except Exception as error:
        raise PipelineError(
            stage=stage_name,
            file_name=file_name,
            original_error=error
        ) from error
