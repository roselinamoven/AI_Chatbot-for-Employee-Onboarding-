from data.employees import generate_employee_data
from dotenv import load_dotenv
import streamlit as st
import logging

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings

# Local imports
from assistant import Assistant
from prompts import SYSTEM_PROMPT, WELCOME_MESSAGE
from gui import AssistantGUI


# ---------------------------
# Streamlit App Entry Point
# ---------------------------
if __name__ == "__main__":

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    st.set_page_config(page_title="Alohomora AI", page_icon="🗝️", layout="wide")

    # ---------------------------
    # Employee Data
    # ---------------------------
    @st.cache_data(ttl=3600, show_spinner="Loading Employee Data...")
    def get_user_data():
        return generate_employee_data(1)[0]

    # ---------------------------
    # Vector Store Initialization
    # ---------------------------
    @st.cache_resource(ttl=3600, show_spinner="Loading Vector Store...")
    def init_vector_store(pdf_path):
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(docs)

            embedding_function = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            persistent_path = "./data/vectorstore"

            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embedding_function,
                persist_directory=persistent_path,
            )
            return vectorstore

        except Exception as e:
            logging.error(f"Error initializing vector store: {str(e)}")
            st.error(f"Failed to initialize vector store: {str(e)}")
            return None

    # ---------------------------
    # Initialize Data & Vector Store
    # ---------------------------
    customer_data = get_user_data()
    vector_store = init_vector_store("data/umbrella_corp_policies.pdf")

    if "customer" not in st.session_state:
        st.session_state.customer = customer_data
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "ai", "content": WELCOME_MESSAGE}]

    # ---------------------------
    # Local LLM via Ollama
    # ---------------------------
    # Make sure you've run in terminal:
    #    ollama pull llama3
    # or choose any model you installed (e.g. mistral, phi3, gemma)
    llm = Ollama(
        model="mistral",
        temperature=0,
        num_predict=512
    )

    # ---------------------------
    # Assistant and GUI
    # ---------------------------
    assistant = Assistant(
        system_prompt=SYSTEM_PROMPT,
        llm=llm,
        message_history=st.session_state.messages,
        employee_information=st.session_state.customer,
        vector_store=vector_store,
    )

    gui = AssistantGUI(assistant)
    gui.render()
