import os
from dotenv import load_dotenv

# MUST be here to load GOOGLE_API_KEY before the LLM class looks for it
load_dotenv()

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from src.tools import setup_retriever, web_search_tool

# ... initialization code ...
# Change 2.5 to 1.5 if you continue to get model errors
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# --- GLOBAL SETUP ---
# In a real app, load these from an env var or config
urls = ["https://lilianweng.github.io/posts/2023-06-23-agent/"]

# Initialize resources once
print("---INITIALIZING RETRIEVER---")
retriever = setup_retriever(urls)


def retrieve_node(state):
    print("---NODE: RETRIEVING---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents}


def grade_documents(state):
    print("---NODE: GRADING---")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = "No"

    # Improved prompt to ensure strict binary output
    grade_prompt = ChatPromptTemplate.from_template(
        """You are a grader assessing relevance of a retrieved document to a user question.
        Document: {document}
        Question: {question}
        If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    )
    grader_chain = grade_prompt | llm

    for d in documents:
        score = grader_chain.invoke({"question": question, "document": d.page_content})
        grade = score.content.lower()

        if "yes" in grade:
            print("   -> GRADE: RELEVANT")
            filtered_docs.append(d)
        else:
            print("   -> GRADE: NOT RELEVANT")
            web_search = "Yes"

    return {"documents": filtered_docs, "web_search": web_search}


def web_search_node(state):
    print("---NODE: SEARCHING WEB---")
    question = state["question"]
    documents = state.get("documents", [])

    results = web_search_tool.invoke({"query": question})

    # Fix: Tavily returns a list of dicts, we need to convert to Documents
    search_docs = [
        Document(page_content=d["content"], metadata={"source": d["url"]})
        for d in results
    ]

    # Add to existing docs
    documents.extend(search_docs)

    return {"documents": documents}


def generate_node(state):
    print("---NODE: GENERATING---")
    question = state["question"]
    documents = state["documents"]

    # Safety check: If no docs exist, tell the user
    if not documents:
        return {"generation": "I could not find any relevant information to answer your question."}

    context = "\n\n".join([d.page_content for d in documents])

    gen_prompt = ChatPromptTemplate.from_template(
        """You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 

        Question: {question} 
        Context: {context} 
        Answer:"""
    )
    gen_chain = gen_prompt | llm

    generation = gen_chain.invoke({"context": context, "question": question})
    return {"generation": generation.content}


def decide_to_generate(state):
    """
    Determines whether to go to web search or generate directly.
    """
    print("---DECISION POINT---")
    if state["web_search"] == "Yes":
        print("   -> DECISION: TRANSFORM QUERY & SEARCH WEB")
        return "search"
    else:
        print("   -> DECISION: GENERATE")
        return "generate"