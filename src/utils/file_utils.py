import hashlib
from pathlib import Path

def compute_file_hash(file_path: Path) -> str:
    sha256_hash = hashlib.sha256()
    try:
        with file_path.open('rb') as f:
            while chunk := f.read(65536):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    except Exception as e:
        return f"Error computing hash for {file_path}: {e}"
