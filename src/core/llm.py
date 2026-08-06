from typing import Literal
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
# from src.core.config import settings


def get_llm(use_case: Literal["summarization", "grading", "generation"] = "generation") -> BaseChatModel:
    models = {
        "summarization": "gemini-3.5-flash-lite",
        "grading":       "gemini-3.5-flash-lite",
        "generation":    "gemini-3.0-flash",
    }
    return ChatGoogleGenerativeAI(model=models[use_case])
