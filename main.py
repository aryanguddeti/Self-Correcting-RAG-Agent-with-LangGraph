import os
from dotenv import load_dotenv

# 1. Import the compiled app from your graph definition
# This assumes your graph.py has 'app = workflow.compile()'
from src.graph import app

# Load environment variables (Google API Key, Tavily Key)
load_dotenv()

def run_crag_agent(query: str):
    """
    Executes the Self-Correcting RAG agent.
    """
    print(f"\n{'=' * 20} AGENT START {'=' * 20}")

    # Initial state required by your GraphState TypedDict
    initial_input = {
        "question": query,
        "web_search": "No",
        "documents": [],
        "generation": ""
    }

    # 2. Run the graph with streaming to see the "thought process"
    # 'stream_mode="updates"' allows you to see which node just finished
    for output in app.stream(initial_input, stream_mode="updates"):
        for node_name, state_update in output.items():
            print(f"\n[Finished Node]: {node_name}")
            # Optional: Print metadata like how many docs were retrieved/graded
            if "documents" in state_update:
                print(f"   -> Current Document Count: {len(state_update['documents'])}")
            if "web_search" in state_update:
                print(f"   -> Web Search Triggered: {state_update['web_search']}")

    # 3. Get the final state to show the answer
    # We invoke it one last time or grab the final chunk from the stream
    final_result = app.invoke(initial_input)

    print(f"\n{'=' * 20} FINAL ANSWER {'=' * 20}")
    print(final_result.get("generation", "I'm sorry, I couldn't find an answer."))
    print(f"{'=' * 53}\n")


if __name__ == "__main__":
    # Example queries to test the "Self-Correction"
    # Query 1: Should be found in your local docs (Lilian Weng's blog)
    # Query 2: Should trigger the Web Search node (Random recent event)

    test_query = "What are the common types of memory in AI agents?"

    run_crag_agent(test_query)