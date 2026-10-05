import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file= ROOT_DIR/".env",
        env_file_encoding="utf-8",
        extra='ignore'
    )
    PROJECT_ROOT : Path = ROOT_DIR
    DATA_INGEST_PATH: Path = ROOT_DIR/'data'
    GOOGLE_API_KEY: str
    PINECONE_API_KEY: str
    MONGODB_URI: str
    PINECONE_INDEX_NAME: str = "self-correcting-rag"


settings = Settings()

# Synchronize key settings back into system environment variables
os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY
os.environ["PINECONE_API_KEY"] = settings.PINECONE_API_KEY