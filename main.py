import streamlit as st
import json
import redis
from utils.pdf_processing import process_pdf_task
from utils.respondent import ask_question
from utils.config import redis_host, redis_pass
import uuid
import tiktoken
import time


def count_tokens(text, model="gpt-4o"):
    """Count the number of tokens in the text for a specific model."""
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)


# Initialize Redis client
redis_client = redis.Redis(
    host=redis_host,
    port=6379,
    password=redis_pass,
)

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "documents" not in st.session_state:
    st.session_state.documents = {}  # Store documents with unique IDs
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_token" not in st.session_state:
    st.session_state.doc_token = 0
if "removed_documents" not in st.session_state:
    st.session_state.removed_documents = []  # Track removed document names


def save_document_to_redis(session_id, doc_id, document_data):
    """Save document data to Redis."""
    redis_key = f"{session_id}:document_data:{doc_id}"
    redis_client.set(redis_key, json.dumps(document_data))


def handle_question(prompt, spinner_placeholder):
    """Handle user question by querying the documents in the session."""
    if prompt:
        try:
            # Only use documents currently in the session
            documents_data = {
                doc_id: doc_info["data"]
                for doc_id, doc_info in st.session_state.documents.items()
            }
            if not documents_data:
                st.warning(
                    "No documents available in the session to answer the question."
                )
                return

            with spinner_placeholder.container():
                st.spinner("Thinking...")
                answer, tot_tokens = ask_question(
                    documents_data, prompt, st.session_state.chat_history
                )

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


def display_chat():
    """Display chat history."""
    if st.session_state.chat_history:
        for i, chat in enumerate(st.session_state.chat_history):
            with st.chat_message("user"):
                st.write(chat["question"])
            with st.chat_message("assistant"):
                st.write(chat["answer"])


# Main UI
st.image("logoD.png", width=200)
st.title("docQuest")
st.subheader("Unveil the Essence, Compare Easily, Analyze Smartly")

with st.sidebar:
    with st.expander("Document(s) are ready:", expanded=True):
        to_remove = []
        for doc_id, doc_info in st.session_state.documents.items():
            # st.write(f"{doc_info['name']}")
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"{doc_info['name']}")
            with col2:
                if st.button(f"⨯", key=f"remove_{doc_id}"):
                    to_remove.append(doc_id)

        for doc_id in to_remove:
            st.session_state.doc_token -= count_tokens(
                str(st.session_state.documents[doc_id]["data"])
            )
            st.session_state.removed_documents.append(
                st.session_state.documents[doc_id]["name"]
            )
            redis_client.delete(f"{st.session_state.session_id}:document_data:{doc_id}")
            del st.session_state.documents[doc_id]
            st.success("Document removed successfully!")
            time.sleep(1.3)
            st.rerun()


# Sidebar
with st.sidebar:
    with st.expander("Upload Document(s)", expanded=False):
        uploaded_files = st.file_uploader(
            "Upload files less than 400 pages",
            type=["pdf", "docx", "xlsx", "pptx"],
            accept_multiple_files=True,
            help="If your question is not answered properly or there's an error, consider uploading smaller documents or splitting larger ones.",
            label_visibility="collapsed",
        )

        if uploaded_files:
            new_files = []
            for uploaded_file in uploaded_files:
                # Skip files that are removed or already uploaded
                if (
                    uploaded_file.name
                    not in [
                        st.session_state.documents[doc_id]["name"]
                        for doc_id in st.session_state.documents
                    ]
                    and uploaded_file.name not in st.session_state.removed_documents
                ):
                    new_files.append(uploaded_file)
                # else:
                # st.info(f"{uploaded_file.name} is already uploaded or was removed.")

            if new_files:
                progress_text = st.empty()
                progress_bar = st.progress(0)
                total_files = len(new_files)

                with st.spinner("Learning about your document(s)..."):
                    try:
                        for i, uploaded_file in enumerate(new_files):
                            document_data = process_pdf_task(
                                uploaded_file, first_file=(i == 0)
                            )
                            if not document_data:
                                st.warning(
                                    "The document exceeds the size limit for processing!",
                                    icon="⚠️",
                                )
                                uploaded_file.seek(0)
                                continue

                            doc_token_count = count_tokens(str(document_data))
                            if st.session_state.doc_token + doc_token_count > 600000:
                                st.warning(
                                    "Document contents so far are too large to query. Not processing further documents. "
                                    "Results may be inaccurate; consider uploading smaller documents.",
                                    icon="⚠️",
                                )
                                continue

                            doc_id = str(uuid.uuid4())
                            st.session_state.documents[doc_id] = {
                                "name": uploaded_file.name,
                                "data": document_data,
                            }
                            st.session_state.doc_token += doc_token_count
                            save_document_to_redis(
                                st.session_state.session_id, doc_id, document_data
                            )
                            st.success(f"{uploaded_file.name} processed!")
                            time.sleep(1)
                            st.rerun()
                            progress_bar.progress((i + 1) / total_files)
                    except Exception as e:
                        st.error(f"Error processing file: {e}")

                progress_text.text("Processing complete.")
                progress_bar.empty()
                st.rerun()


# Main input and chat display
if st.session_state.documents:
    prompt = st.chat_input("Ask me anything about your documents", key="chat_input")
    spinner_placeholder = st.empty()
    if prompt:
        handle_question(prompt, spinner_placeholder)

display_chat()
