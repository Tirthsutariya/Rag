# llm/model.py
from langchain_groq import ChatGroq
from langchain import hub

def initialize_llm(groq_api_key):
    return ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile"
    )

def get_prompt():
    prompt="""You are a helpful assistant. Answer the question based on the context provided this loggically. If the answer is not in the context, say "I don't know"""
    return prompt