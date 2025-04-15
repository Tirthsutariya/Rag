# config/settings.py
import os
from dotenv import load_dotenv

def load_config():
    load_dotenv()
    return {
        "groq_api_key": os.getenv("GROQ_API_KEY"),
        "langsmith_api_key": os.getenv("LANGSMITH_API_KEY"),
        "pdf_path": r"T:\mental langraph\mental health.pdf"  # Optional default path
    }