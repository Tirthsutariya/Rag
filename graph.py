# workflow/graph.py
from typing_extensions import TypedDict, List
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def create_workflow(vectorstore, llm, prompt):
    def retrieve(state: State):
        retrieved_docs = vectorstore.similarity_search(state['question'], k=1)
        return {'context': retrieved_docs}

    def generate(state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": docs_content})
        answer = llm.invoke(messages)
        return {"answer": answer}

    graph_builder = StateGraph(State)
    graph_builder.add_node("retrieve", retrieve)
    graph_builder.add_node("generate", generate)
    graph_builder.add_edge(START, "retrieve")
    graph_builder.add_edge("retrieve", "generate")
    graph_builder.add_edge("generate", END)

    return graph_builder.compile()