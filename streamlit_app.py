import streamlit as st
import logging
import os
import tempfile
import shutil
import fitz  # PyMuPDF
import ollama
import time

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from typing import List, Tuple, Dict, Any, Optional

# Streamlit page configuration
st.set_page_config(
    page_title="Ollama PDF RAG Streamlit UI",
    page_icon="🎈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner=True)
def extract_model_names(
    models_info: Dict[str, List[Dict[str, Any]]],
) -> Tuple[str, ...]:
    """
    Extract model names from the provided models information.

    Args:
        models_info (Dict[str, List[Dict[str, Any]]]): Dictionary containing information about available models.

    Returns:
        Tuple[str, ...]: A tuple of model names.
    """
    logger.info("Extracting model names from models_info")
    model_names = tuple(model["name"] for model in models_info["models"])
    logger.info(f"Extracted model names: {model_names}")
    return model_names


def extract_text_from_pdf(file_upload) -> str:
    """
    Extract text from an uploaded PDF file using PyMuPDF.

    Args:
        file_upload (st.UploadedFile): Streamlit file upload object containing the PDF.

    Returns:
        str: The extracted text from the PDF.
    """
    logger.info(f"Extracting text from PDF file: {file_upload.name}")
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, file_upload.name)

    with open(path, "wb") as f:
        f.write(file_upload.getvalue())

    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    
    doc.close()

    # Retry mechanism to ensure the file is not in use
    for _ in range(5):
        try:
            shutil.rmtree(temp_dir)
            break
        except PermissionError:
            logger.warning("Temporary directory in use, retrying...")
            time.sleep(1)
    else:
        logger.error("Failed to remove temporary directory after retries")
    
    logger.info(f"Temporary directory {temp_dir} removed")
    return text


def create_vector_db(file_upload, embedding_model) -> Chroma:
    """
    Create a vector database from an uploaded PDF file.

    Args:
        file_upload (st.UploadedFile): Streamlit file upload object containing the PDF.
        embedding_model (str): The embedding model to use.

    Returns:
        Chroma: A vector store containing the processed document chunks.
    """
    logger.info(f"Creating vector DB from file upload: {file_upload.name}")
    text = extract_text_from_pdf(file_upload)
    chunks = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100).split_text(text)
    logger.info("Document split into chunks")

    embeddings = OllamaEmbeddings(model=embedding_model, show_progress=True)
    vector_db = Chroma.from_texts(chunks, embeddings, collection_name="myRAG")
    logger.info("Vector DB created")
    return vector_db


def process_question(question: str, vector_db: Optional[Chroma], selected_model: str) -> str:
    """
    Process a user question using the vector database and selected language model.

    Args:
        question (str): The user's question.
        vector_db (Optional[Chroma]): The vector database containing document embeddings.
        selected_model (str): The name of the selected language model.

    Returns:
        str: The generated response to the user's question.
    """
    logger.info(f"Processing question: {question} using model: {selected_model}")
    llm = ChatOllama(model=selected_model, temperature=0)

    if vector_db is not None:
        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI language model assistant. Your task is to generate 3
            different versions of the given user question to retrieve relevant documents from
            a vector database. By generating multiple perspectives on the user question, your
            goal is to help the user overcome some of the limitations of the distance-based
            similarity search. Provide these alternative questions separated by newlines.
            Original question: {question}""",
        )

        retriever = MultiQueryRetriever.from_llm(
            vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
        )

        template = """Answer the question based ONLY on the following context:
        {context}
        Question: {question}
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Only provide the answer from the {context}, nothing else.
        Add snippets of the context you used to answer the question.
        """

        prompt = ChatPromptTemplate.from_template(template)

        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        response = chain.invoke(question)
    else:
        response = llm(question)

    logger.info("Question processed and response generated")
    return response


@st.cache_data
def extract_all_pages_as_images(file_upload) -> List[Any]:
    """
    Extract all pages from a PDF file as images using PyMuPDF.

    Args:
        file_upload (st.UploadedFile): Streamlit file upload object containing the PDF.

    Returns:
        List[Any]: A list of image objects representing each page of the PDF.
    """
    logger.info(f"Extracting all pages as images from file: {file_upload.name}")
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, file_upload.name)

    with open(path, "wb") as f:
        f.write(file_upload.getvalue())

    doc = fitz.open(path)
    images = []
    for page in doc:
        pix = page.get_pixmap()
        image = pix.tobytes("png")
        images.append(image)
    
    doc.close()

    # Retry mechanism to ensure the file is not in use
    for _ in range(5):
        try:
            shutil.rmtree(temp_dir)
            break
        except PermissionError:
            logger.warning("Temporary directory in use, retrying...")
            time.sleep(1)
    else:
        logger.error("Failed to remove temporary directory after retries")
    
    logger.info(f"Temporary directory {temp_dir} removed")
    return images


def delete_vector_db(vector_db: Optional[Chroma]) -> None:
    """
    Delete the vector database and clear related session state.

    Args:
        vector_db (Optional[Chroma]): The vector database to be deleted.
    """
    logger.info("Deleting vector DB")
    if vector_db is not None:
        vector_db.delete_collection()
        st.session_state.pop("pdf_pages", None)
        st.session_state.pop("file_upload", None)
        st.session_state.pop("vector_db", None)
        st.success("Collection and temporary files deleted successfully.")
        logger.info("Vector DB and related session state cleared")
        st.rerun()
    else:
        st.error("No vector database found to delete.")
        logger.warning("Attempted to delete vector DB, but none was found")


def main() -> None:
    """
    Main function to run the Streamlit application.

    This function sets up the user interface, handles file uploads,
    processes user queries, and displays results.
    """
    st.subheader("🧠 Ollama PDF RAG playground", divider="gray", anchor=False)

    models_info = ollama.list()
    available_models = extract_model_names(models_info)

    col1, col2 = st.columns([1.5, 2])

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "vector_db" not in st.session_state:
        st.session_state["vector_db"] = None

    if available_models:
        selected_model = col2.selectbox(
            "Pick a model available locally on your system ↓", available_models
        )

    embedding_model = st.sidebar.selectbox(
        "Select an embedding model ↓", available_models
    )

    file_upload = col1.file_uploader(
        "Upload a PDF file ↓", type="pdf", accept_multiple_files=False
    )

    if file_upload:
        st.session_state["file_upload"] = file_upload
        if st.session_state["vector_db"] is None:
            st.session_state["vector_db"] = create_vector_db(file_upload, embedding_model)
        pdf_pages = extract_all_pages_as_images(file_upload)
        st.session_state["pdf_pages"] = pdf_pages

        zoom_level = col1.slider(
            "Zoom Level", min_value=100, max_value=1000, value=700, step=50
        )

        with col1:
            for page_image in pdf_pages:
                st.image(page_image, width=zoom_level)

    delete_collection = col1.button("⚠️ Delete collection", type="secondary")

    if delete_collection:
        delete_vector_db(st.session_state["vector_db"])

    with col2:
        message_container = st.container()

        for message in st.session_state["messages"]:
            avatar = "🤖" if message["role"] == "assistant" else "😎"
            with message_container:
                st.markdown(message["content"])

        if prompt := st.chat_input("Enter a prompt here..."):
            try:
                st.session_state["messages"].append({"role": "user", "content": prompt})
                with message_container:
                    st.markdown(prompt)

                with message_container:
                    with st.spinner("Processing..."):
                        response = process_question(
                            prompt, st.session_state.get("vector_db"), selected_model
                        )
                        st.markdown(response)

                st.session_state["messages"].append(
                    {"role": "assistant", "content": response}
                )

            except Exception as e:
                st.error(e, icon="⛔️")
                logger.error(f"Error processing prompt: {e}")
        else:
            st.warning("Upload a PDF file to begin chat or start a new chat...")

if __name__ == "__main__":
    main()

