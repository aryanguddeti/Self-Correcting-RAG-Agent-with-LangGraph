from langchain_core.prompts import ChatPromptTemplate

TABLE_SUMMARIZATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert data analyst. Summarize tables to make them easily searchable via semantic vector search.
        Include:
        1. Main topic or purpose of the table.
        2. Key metrics, entity names, or variables listed.
        3. Notable trends or key data highlights."""
    ),
    (
        "human",
        """Please summarize the following table in a clean, concise paragraph:
        {table_content}"""
    )
])