from typing_extensions import TypedDict, List
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def create_workflow(vectorstore, llm, prompt):
    def retrieve(state: State):
        # Perform similarity search to retrieve relevant documents
        retrieved_docs = vectorstore.similarity_search(state['question'], k=1)
        return {'context': retrieved_docs}

    def generate(state: State):
        # Combine the content of the retrieved documents
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        
        # Use the custom prompt that was passed in, including the question and context
        custom_prompt = prompt.format(question=state['question'], context=docs_content)
        
        # Send the combined prompt to the model for generating an answer
        answer = llm.invoke(custom_prompt)
        return {"answer": answer}

    # Initialize the graph builder
    graph_builder = StateGraph(State)
    
    # Add the nodes for retrieving and generating answers
    graph_builder.add_node("retrieve", retrieve)
    graph_builder.add_node("generate", generate)
    
    # Add edges between the nodes
    graph_builder.add_edge(START, "retrieve")
    graph_builder.add_edge("retrieve", "generate")
    graph_builder.add_edge("generate", END)

    # Compile the graph into a runnable workflow
    return graph_builder.compile()
