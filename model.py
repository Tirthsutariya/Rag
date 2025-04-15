from langchain_groq import ChatGroq
from langchain import hub

def initialize_llm(groq_api_key):
    return ChatGroq(
        groq_api_key=groq_api_key,
        model_name="gemma2-9b-it",
        max_tokens=4096  # You can go higher if needed (e.g., 8192)
    )

def get_prompt():
    # Custom prompt (replace this with your desired prompt)
    custom_prompt = """
    You are a helpful assistant. Please answer the user's questions based on the uploaded PDF document.
    If the answer is not directly available, provide the most relevant information from the document.
    """
    return custom_prompt