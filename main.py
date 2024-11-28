import streamlit as st
import json
import redis
from utils.pdf_processing import process_pdf_task
from utils.respondent import ask_question
from utils.config import redis_host, redis_pass
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
from docx import Document
import uuid
import tiktoken
from docx.shared import Pt
import re

MAX_TOKEN_LIMIT = 700000  # Maximum token limit

def remove_markdown(text):
    """Remove Markdown formatting from text."""
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    return text

def count_tokens(text, model="gpt-4"):
    """Count tokens in the text."""
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)

# Initialize Redis client
redis_client = redis.Redis(
    host=redis_host,
    port=6379,
    password=redis_pass,
)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_token" not in st.session_state:
    st.session_state.doc_token = 0

# Function to save document data to Redis
def save_document_to_redis(session_id, file_name, document_data):
    redis_key = f"{session_id}:document_data:{file_name}"
    redis_client.set(redis_key, json.dumps(document_data))

# Function to retrieve document data from Redis
def get_document_from_redis(session_id, file_name):
    redis_key = f"{session_id}:document_data:{file_name}"
    data = redis_client.get(redis_key)
    if data:
        return json.loads(data)
    return None

# Function to retrieve all user documents from Redis
def retrieve_user_documents_from_redis(session_id):
    documents = {}
    for key in redis_client.keys(f"{session_id}:document_data:*"):
        file_name = key.decode().split(f"{session_id}:document_data:")[1]
        documents[file_name] = get_document_from_redis(session_id, file_name)
    return documents

# Function to handle user question and response
def handle_question(prompt, spinner_placeholder):
    if prompt:
        try:
            documents_data = retrieve_user_documents_from_redis(st.session_state.session_id)
            with spinner_placeholder.container():
                st.markdown(
                    """
                    <header>
                    <div style="text-align: center;">
                        <div class="spinner" style="margin: 20px;">
                            <div class="bounce1"></div>
                            <div class="bounce2"></div>
                            <div class="bounce3"></div>
                        </div>
                    </div>
                    </header>
                    """,
                    unsafe_allow_html=True,
                )
                answer, tot_tokens = ask_question(documents_data, prompt, st.session_state.chat_history)
            st.session_state.chat_history.append(
                {
                    "question": prompt,
                    "answer": f"{answer}\nTotal tokens: {tot_tokens}",
                }
            )
        except Exception as e:
            st.error(f"Error processing question: {e}")
        finally:
            spinner_placeholder.empty()

# Function to process uploaded documents and check token limit
def handle_uploaded_files(uploaded_files):
    total_token_count = st.session_state.doc_token  # Start with existing token count
    new_files = []

    for uploaded_file in uploaded_files:
        # If document is already processed, skip it
        if not redis_client.exists(f"{st.session_state.session_id}:document_data:{uploaded_file.name}"):
            new_files.append(uploaded_file)
        else:
            st.info(f"{uploaded_file.name} is ready.")

    if new_files:
        progress_text = st.empty()
        progress_bar = st.progress(0)
        total_files = len(new_files)

        with st.spinner("Learning about your document(s)..."):
            # Count the tokens in new files before processing
            for uploaded_file in new_files:
                # Assuming the PDF processing is related to text extraction, we simulate the token count check
                with uploaded_file:
                    file_content = uploaded_file.read().decode("utf-8", errors="ignore")
                    document_token_count = count_tokens(file_content)

                    # Check if adding this document will exceed the token limit
                    if total_token_count + document_token_count > MAX_TOKEN_LIMIT:
                        st.error("The total token count exceeds the limit of 700,000 tokens. Please upload smaller documents.")
                        return  # Exit the function if the token count is exceeded

                    # Add the tokens of the current document to the total
                    total_token_count += document_token_count

            # Now process the files since they are within the token limit
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_to_file = {
                    executor.submit(process_pdf_task, uploaded_file, first_file=(index == 0)): uploaded_file
                    for index, uploaded_file in enumerate(new_files)
                }

                for i, future in enumerate(as_completed(future_to_file)):
                    uploaded_file = future_to_file[future]
                    try:
                        document_data = future.result()
                        save_document_to_redis(st.session_state.session_id, uploaded_file.name, document_data)
                        st.session_state.doc_token = total_token_count  # Update token count in session state
                        st.success(f"{uploaded_file.name} processed!")
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {e}")
                    
                    progress_bar.progress((i + 1) / total_files)

        progress_text.text("Processing complete.")
        progress_bar.empty()

    st.sidebar.write(f"Total document tokens: {st.session_state.doc_token}")

# UI Layout
with st.sidebar:
    uploaded_files = st.file_uploader(
        "Upload your documents",
        type=["pdf", "docx", "xlsx", "pptx"],
        accept_multiple_files=True,
        help="Supports PDF, DOCX, XLSX, and PPTX formats.",
    )

    if uploaded_files:
        handle_uploaded_files(uploaded_files)

    if retrieve_user_documents_from_redis(st.session_state.session_id):
        download_data = json.dumps(
            retrieve_user_documents_from_redis(st.session_state.session_id), indent=4
        )
        st.download_button(
            label="Download Document Analysis",
            data=download_data,
            file_name="document_analysis.json",
            mime="application/json",
        )

st.image("logoD.png", width=200)
st.title("docQuest")
st.subheader("Unveil the Essence, Compare Easily, Analyze Smartly", divider="orange")

# Display question input if documents are loaded
if retrieve_user_documents_from_redis(st.session_state.session_id):
    prompt = st.chat_input("Ask me anything about your documents", key="chat_input")
    spinner_placeholder = st.empty()
    if prompt:
        handle_question(prompt, spinner_placeholder)

display_chat()
