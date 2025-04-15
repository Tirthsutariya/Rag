import streamlit as st
from settings import load_config
from doc_loader import load_documents, split_documents
from vector_store import create_vectorstore
from model import initialize_llm, get_prompt
from graph import create_workflow

def main():
    st.set_page_config(page_title="PDF Chatbot", layout="wide")
    st.title("📄 PDF Chatbot")
    st.write("Upload a PDF and chat with its content!")

    # Load configuration
    config = load_config()
    if not config["groq_api_key"]:
        st.error("🚨 GROQ_API_KEY not found in environment variables.")
        return

    # Initialize session state
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "graph" not in st.session_state:
        st.session_state.graph = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False

    # Reset session state button
    if st.button("🔁 Reset and Upload New PDF"):
        for key in ["vectorstore", "graph", "chat_history", "pdf_processed"]:
            if key in st.session_state:
                del st.session_state[key]
        st.experimental_rerun()

    # File uploader
    uploaded_file = st.file_uploader("📤 Choose a PDF file", type="pdf", key="pdf_uploader")

    # Process PDF when uploaded
    if uploaded_file is not None and not st.session_state.pdf_processed:
        with st.spinner("📚 Loading and splitting PDF..."):
            try:
                # Load and split documents
                docs = load_documents(pdf_file=uploaded_file)
                st.info("🔍 Splitting document into chunks...")
                all_splits = split_documents(docs)
                st.success(f"✅ PDF processed! Total chunks: {len(all_splits)}")

                # Create vectorstore and workflow
                st.info("⚙️ Embedding chunks and initializing graph...")
                st.session_state.vectorstore = create_vectorstore(all_splits)
                llm = initialize_llm(config["groq_api_key"])
                prompt = get_prompt()
                st.session_state.graph = create_workflow(st.session_state.vectorstore, llm, prompt)
                st.session_state.pdf_processed = True
                st.session_state.chat_history = []  # Reset chat history for new PDF
            except Exception as e:
                st.error(f"❌ Error processing PDF: {str(e)}")

    # Chat interface
    if st.session_state.pdf_processed and st.session_state.graph is not None:
        # Display chat history
        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(chat["question"])
            with st.chat_message("assistant"):
                st.write(chat["answer"])

        # Question input
        question = st.chat_input("💬 Ask a question about the PDF:")
        if question:
            # Add user question to history
            st.session_state.chat_history.append({"question": question, "answer": ""})
            with st.chat_message("user"):
                st.write(question)

            # Generate answer
            with st.spinner("🤖 Thinking..."):
                try:
                    response = st.session_state.graph.invoke({"question": question})
                    answer = response["answer"].content
                    st.session_state.chat_history[-1]["answer"] = answer
                    with st.chat_message("assistant"):
                        st.write(answer)
                except Exception as e:
                    st.error(f"❌ Error generating answer: {str(e)}")
                    st.session_state.chat_history.pop()
    else:
        st.info("📂 Please upload a PDF to start chatting.")

if __name__ == "__main__":
    main()
