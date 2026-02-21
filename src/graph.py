from langgraph.graph import END, StateGraph, START
from src.state import GraphState
from src.nodes import (
    retrieve_node,
    grade_documents,
    web_search_node,
    generate_node,
    decide_to_generate
)

workflow = StateGraph(GraphState)

# Define the Nodes
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("web_search", web_search_node)
workflow.add_node("generate", generate_node)

# Build the connections
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")

# Use the conditional routing logic
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "search": "web_search",
        "generate": "generate",
    },
)

workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()