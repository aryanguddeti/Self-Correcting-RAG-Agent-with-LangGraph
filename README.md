# 🧠 Self-Correcting RAG (CRAG) with LangGraph & Gemini

This project implements a **Corrective Retrieval-Augmented Generation (CRAG)** agent. Unlike standard RAG, which blindly trusts retrieved documents, this system treats retrieval as a hypothesis that must be tested. It uses an agentic loop to grade documents for relevance and triggers an automated web search fallback if the local knowledge base is insufficient.

## 🚀 The "Self-Correction" Logic

Most RAG systems fail because they feed irrelevant data into the LLM, leading to hallucinations. This project solves that by implementing a **LangGraph** state machine:

1. **Retrieve:** Queries a local **ChromaDB** vector store (populated with Lilian Weng’s AI research blogs).
2. **Grade:** A specialized **Gemini 1.5 Flash** grader evaluates each chunk.
3. **Route:** * **Relevant:** If the data is good, it proceeds to generation.
* **Irrelevant/Missing:** If the data is poor, the agent recognizes its own "ignorance" and triggers the **Tavily Web Search** node.


4. **Generate:** Synthesizes the final answer using the combined, validated context.

---

## 🏗️ Technical Skillset Demonstrated

### 1. Agentic Orchestration (LangGraph)

Beyond simple chains, I implemented a cyclic graph. This includes **Conditional Edges**, where the graph decides its own path (Generate vs. Web Search) based on the real-time state of the application.

### 2. Modern AI Stack

* **LLM:** Google Gemini 1.5 Flash (leveraging high-speed inference).
* **Vector Database:** ChromaDB for local persistence of embeddings.
* **Embeddings:** `all-MiniLM-L6-v2` (HuggingFace) for local, cost-effective vectorization.
* **Search:** Tavily API (Optimized for LLM-ready search results).

### 3. Engineering Best Practices

* **Environment Management:** Secure API key handling using `python-dotenv`.
* **Rate-Limit Handling:** Implementation of `time.sleep()` and request throttling to handle Gemini's free-tier Resource Exhaustion (429 errors).
* **Path Resolution:** Use of `pathlib` for robust file/env discovery across different OS environments.

---

## 📂 Project Architecture

```text
├── main.py           # Entry point: Manages the execution stream and user query
├── src/
│   ├── graph.py      # The StateMachine definition & workflow logic
│   ├── nodes.py      # Logic for Retrieval, Grading, Web Search, and Generation
│   ├── state.py      # TypedDict defining the shared "memory" of the agent
│   ├── tools.py      # Vector DB initialization and Search tool configuration
│   └── __init__.py   # Python package markers
├── .env              # API Configuration
└── requirements.txt  # Project dependencies

```

---

## 🛠️ Setup & Execution

1. **Environment:** Create a virtual environment and install dependencies:
```bash
pip install langchain-google-genai langgraph langchain-community langchain-chroma langchain-huggingface sentence-transformers tavily-python python-dotenv

```


2. **API Keys:** Add your `GOOGLE_API_KEY` and `TAVILY_API_KEY` to a `.env` file in the root directory.
3. **Run:**
```bash
python3 main.py

```



---

## 📈 Real-World Application

This architecture is ideal for enterprise environments where local documentation (e.g., internal PDFs, Wikis) might be outdated. By allowing the agent to verify its data and check the live web, we significantly reduce the risk of outdated or "hallucinated" responses.

---
