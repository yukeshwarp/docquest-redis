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

def remove_markdown(text):
    """Remove Markdown formatting from text."""
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    return text

def count_tokens(text, model="gpt-4o"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)


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


def save_document_to_redis(session_id, file_name, document_data):
    redis_key = f"{session_id}:document_data:{file_name}"
    redis_client.set(redis_key, json.dumps(document_data))


def get_document_from_redis(session_id, file_name):
    redis_key = f"{session_id}:document_data:{file_name}"
    data = redis_client.get(redis_key)
    if data:
        return json.loads(data)
    return None


def retrieve_user_documents_from_redis(session_id):
    documents = {}
    for key in redis_client.keys(f"{session_id}:document_data:*"):
        file_name = key.decode().split(f"{session_id}:document_data:")[1]
        documents[file_name] = get_document_from_redis(session_id, file_name)
    return documents


def handle_question(prompt, spinner_placeholder):
    if prompt:
        try:
            documents_data = retrieve_user_documents_from_redis(
                st.session_state.session_id
            )
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


def reset_session():
    st.session_state.chat_history = []
    st.session_state.doc_token = 0
    for key in redis_client.keys(f"{st.session_state.session_id}:document_data:*"):
        redis_client.delete(key)


def display_chat():
    if st.session_state.chat_history:
        for i, chat in enumerate(st.session_state.chat_history):
            user_message = f"""
            <div style='padding:10px; border-radius:10px; margin:5px 0; text-align:right;'>
            {chat['question']}
            </div>
            """
            assistant_message = f"""
            <div style='padding:10px; border-radius:10px; margin:5px 0; text-align:left;'>
            {chat['answer']}
            </div>
            """
            st.markdown(user_message, unsafe_allow_html=True)
            st.markdown(assistant_message, unsafe_allow_html=True)
            chat_content = {
                "question": chat["question"],
                "answer": chat["answer"],
            }
            doc = generate_word_document(chat_content)
            word_io = io.BytesIO()
            doc.save(word_io)
            word_io.seek(0)
            st.download_button(
                label="↴",
                data=word_io,
                file_name=f"chat_{i+1}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )


def generate_word_document(content):
    doc = Document()

    # Set up the heading
    heading = doc.add_heading("Chat Response", level=0)
    heading.runs[0].font.name = "Aptos"
    heading.runs[0].font.size = Pt(14)

    # Add the question
    question_para = doc.add_paragraph()
    question_run = question_para.add_run("Question: ")
    question_run.font.name = "Aptos"
    question_run.font.size = Pt(12)

    question_text = question_para.add_run(remove_markdown(content["question"]))
    question_text.font.name = "Aptos"
    question_text.font.size = Pt(12)

    # Add the answer
    answer_para = doc.add_paragraph()
    answer_run = answer_para.add_run("Answer: ")
    answer_run.font.name = "Aptos"
    answer_run.font.size = Pt(12)

    answer_text = answer_para.add_run(remove_markdown(content["answer"]))
    answer_text.font.name = "Aptos"
    answer_text.font.size = Pt(12)

    return doc
from PyPDF2 import PdfReader
from docx import Document
from pptx import Presentation
import pandas as pd

def extract_text_from_files(file_list):
    """
    Extract text from a list of uploaded files.
    """
    all_text = {}
    for uploaded_file in file_list:
        file_name = uploaded_file.name
        try:
            text = extract_text_from_file(uploaded_file)
            all_text[file_name] = text
        except Exception as e:
            all_text[file_name] = f"Error processing {file_name}: {e}"
    return all_text


def extract_text_from_file(uploaded_file):
    """
    Extract text from uploaded files of various types (PDF, DOCX, PPTX, XLSX).
    """
    file_type = uploaded_file.type

    try:
        if file_type == "application/pdf":
            return extract_text_from_pdf(uploaded_file)
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return extract_text_from_docx(uploaded_file)
        elif file_type == "application/vnd.openxmlformats-officedocument.presentationml.presentation":
            return extract_text_from_pptx(uploaded_file)
        elif file_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            return extract_text_from_xlsx(uploaded_file)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    except Exception as e:
        raise RuntimeError(f"Error extracting text from file: {e}")


def extract_text_from_pdf(file):
    """
    Extract text from a PDF file.
    """
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def extract_text_from_docx(file):
    """
    Extract text from a DOCX file.
    """
    doc = Document(file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text


def extract_text_from_pptx(file):
    """
    Extract text from a PPTX file.
    """
    presentation = Presentation(file)
    text = ""
    for slide in presentation.slides:
        for shape in slide.shapes:
            if shape.has_text_frame:
                text += shape.text + "\n"
    return text


def extract_text_from_xlsx(file):
    """
    Extract text from an XLSX file by reading all cells.
    """
    df = pd.read_excel(file, sheet_name=None)  # Read all sheets
    text = ""
    for sheet_name, sheet_data in df.items():
        text += f"Sheet: {sheet_name}\n"
        text += sheet_data.to_string(index=False, header=True) + "\n"
    return text


with st.sidebar:
    uploaded_files = st.file_uploader(
        "Upload your documents",
        type=["pdf", "docx", "xlsx", "pptx"],
        accept_multiple_files=True,
        help="Supports PDF, DOCX, XLSX, and PPTX formats.",
    )

    if uploaded_files:
        new_files = []
        for uploaded_file in uploaded_files:
            if not redis_client.exists(
                f"{st.session_state.session_id}:document_data:{uploaded_file.name}"
            ):
                new_files.append(uploaded_file)
            else:
                st.info(f"{uploaded_file.name} is ready.")

        if new_files:
            progress_text = st.empty()
            progress_bar = st.progress(0)
            total_files = len(new_files)
            text = extract_text_from_files(new_files)
            if count_tokens(str(text))>300000:
                st.warning(
                    f"The uploaded files are too large to process, Please upload smaller documents, or consider splitting files."
                )
                st.write("The application will restart in 3 seconds...")
                time.sleep(3)  # Pause for 3 seconds
                st.experimental_rerun()

            with st.spinner("Learning about your document(s)..."):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    future_to_file = {
                        executor.submit(
                            process_pdf_task, uploaded_file, first_file=(index == 0)
                        ): uploaded_file
                        for index, uploaded_file in enumerate(new_files)
                    }

                    for i, future in enumerate(as_completed(future_to_file)):
                        uploaded_file = future_to_file[future]
                        try:
                            document_data = future.result()
                            st.session_state.doc_token += count_tokens(
                                str(document_data)
                            )
                            save_document_to_redis(
                                st.session_state.session_id,
                                uploaded_file.name,
                                document_data,
                            )
                            st.success(
                                f"{uploaded_file.name} processed!"
                            )
                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {e}")

                        progress_bar.progress((i + 1) / total_files)
            st.sidebar.write(f"Total document tokens: {st.session_state.doc_token}")
            progress_text.text("Processing complete.")
            progress_bar.empty()

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

if retrieve_user_documents_from_redis(st.session_state.session_id):
    prompt = st.chat_input("Ask me anything about your documents", key="chat_input")
    spinner_placeholder = st.empty()
    if prompt:
        handle_question(prompt, spinner_placeholder)

display_chat()
