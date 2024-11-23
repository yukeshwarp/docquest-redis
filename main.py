import streamlit as st
import redis
import json
from utils.pdf_processing import process_pdf_task
from utils.llm_interaction import (
    extract_topics_from_text,
    check_page_relevance,
    is_summary_request,
)
from concurrent.futures import ThreadPoolExecutor
import uuid
import tiktoken
from openai import AzureOpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from itertools import islice


# Initialize Redis client
redis_client = redis.Redis(
    host="yuktestredis.redis.cache.windows.net",
    port=6379,
    password="VBhswgzkLiRpsHVUf4XEI2uGmidT94VhuAzCaB2tVjs=",
)

# Initialize Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint="https://uswest3daniel.openai.azure.com",
    api_key="fcb2ce5dc289487fad0f6674a0b35312",
    api_version="2024-10-01-preview",
)

# Session states
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "doc_token" not in st.session_state:
    st.session_state.doc_token = 0


# Helper Functions
def count_tokens(text, model="gpt-4o"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)


def save_document_to_redis(session_id, file_name, document_data):
    redis_key = f"{session_id}:document_data:{file_name}"
    redis_client.set(redis_key, json.dumps(document_data))


def retrieve_user_documents_from_redis(session_id):
    documents = {}
    for key in redis_client.keys(f"{session_id}:document_data:*"):
        file_name = key.decode().split(f"{session_id}:document_data:")[1]
        documents[file_name] = json.loads(redis_client.get(key))
    return documents


# Sidebar: Document Upload
with st.sidebar:
    st.write(f"**Total Document Tokens:** {st.session_state.doc_token}")
    uploaded_files = st.file_uploader(
        "Upload your documents",
        type=["pdf", "docx", "xlsx", "pptx"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if not redis_client.exists(
                f"{st.session_state.session_id}:document_data:{uploaded_file.name}"
            ):
                try:
                    document_data = process_pdf_task(uploaded_file)
                    save_document_to_redis(
                        st.session_state.session_id,
                        uploaded_file.name,
                        document_data,
                    )
                    st.session_state.doc_token += count_tokens(str(document_data))
                    st.success(f"{uploaded_file.name} processed successfully!")
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")

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


def batched(iterable, batch_size):
    """Helper function to batch an iterable into fixed-size chunks."""
    iterator = iter(iterable)
    while batch := list(islice(iterator, batch_size)):
        yield batch


# Main Interface
st.image("logoD.png", width=200)
st.title("docQuest")
st.subheader("Unveil the Essence, Compare Easily, Analyze Smartly")

# Input field for user prompt
if prompt := st.chat_input("Ask me anything about your documents"):
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve user documents
    documents = retrieve_user_documents_from_redis(st.session_state.session_id)
    if not documents:
        st.error("No documents uploaded. Please upload documents to proceed.")
    else:

        if is_summary_request(prompt):
            # Combine all full text from the pages to perform NMF topic modeling
            combined_text = "\n".join(
                page.get("full_text", "")
                for doc_data in documents.values()
                for page in doc_data["pages"]
            )

            # Topic modeling with NMF
            vectorizer = TfidfVectorizer(stop_words="english")
            tfidf_matrix = vectorizer.fit_transform([combined_text])
            nmf_model = NMF(n_components=5, random_state=1)
            nmf_topics = nmf_model.fit_transform(tfidf_matrix)

            # Extract prominent topics and terms
            topic_terms = []
            for topic_idx, topic in enumerate(nmf_model.components_):
                topic_terms.append(
                    [
                        vectorizer.get_feature_names_out()[i]
                        for i in topic.argsort()[: -5 - 1 : -1]
                    ]
                )

            # Create a list of prominent terms from all topics
            all_prominent_terms = [term for sublist in topic_terms for term in sublist]

            def get_page_topic_relevance(page_text):
                page_vectorized = vectorizer.transform([page_text])
                topic_scores = nmf_model.transform(page_vectorized)
                return sum(topic_scores[0])

            # Extract pages that are relevant to the prominent topics
            relevant_pages = []
            for doc_name, doc_data in documents.items():
                for page in doc_data["pages"]:
                    if get_page_topic_relevance(page.get("full_text", "")) > 0:
                        page_summary = page.get("text_summary", "")
                        image_explanation = ""
                        for image_data in page.get("image_analysis", []):
                            image_explanation += f"Page {image_data['page_number']} Image Explanation: {image_data['explanation']}\n"
                        relevant_pages.append(
                            {
                                "doc_name": doc_name,
                                "page_number": page["page_number"],
                                "summary": page_summary,
                                "image_explanation": image_explanation,
                            }
                        )

            if relevant_pages:
                # Process pages in batches of 10
                batched_summaries = []
                for batch in batched(relevant_pages, 10):
                    combined_relevant_content = "\n".join(
                        f"**Document: {data['doc_name']}, Page {data['page_number']}**\n"
                        f"Summary:\n{data['summary']}\n"
                        f"Image Explanation:\n{data['image_explanation']}"
                        for data in batch
                    )

                    # Send the combined batch content to LLM to generate a structured summary
                    try:
                        with st.chat_message("assistant"):
                            response_placeholder = st.empty()
                            full_response = ""

                            # Prepare the prompt for LLM
                            prompt_for_llm = f"""
                            Given the following content with summaries and image explanations from various pages of documents, generate a structured summary with subheadings and bullet points for each document. 
                            Use subheadings for each document and bullet points for each page within the document. 
                            
                            {combined_relevant_content}
                            """

                            # Query LLM to generate the structured summary
                            response = client.chat.completions.create(
                                model="GPT-4Omni",
                                messages=[
                                    {
                                        "role": "system",
                                        "content": "You are an assistant that generates structured, readable summaries with subheadings and bullet points.",
                                    },
                                    {
                                        "role": "user",
                                        "content": prompt_for_llm,
                                    },
                                ],
                            )

                            # Safely access the content of the response
                            if isinstance(response, dict):
                                full_response = (
                                    response.get("choices", [{}])[0]
                                    .get("message", {})
                                    .get("content", "")
                                )
                            else:
                                st.error("Unexpected response format from the API.")

                            # Append the batch response
                            batched_summaries.append(full_response)

                    except Exception as e:
                        st.error(f"Error generating summary: {e}")

                # Combine all batch summaries into a final summary
                final_summary = "\n\n".join(batched_summaries)

                # Display the combined final summary
                response_placeholder.markdown(final_summary)

        else:
            relevant_pages = []
            # Identify relevant pages using extracted topics and image explanations
            for doc_name, doc_data in documents.items():
                for page in doc_data["pages"]:
                    # Combine full text and image explanation
                    page_text = page.get("full_text", "")

                    # Check if image analysis exists for the page
                    image_explanation = ""
                    for image_data in page.get("image_analysis", []):
                        # Extract the image explanation for this page
                        image_explanation += f"Page {image_data['page_number']}: {image_data['explanation']}\n"

                    # Combine full text and image explanation
                    combined_content = page_text + "\n" + image_explanation

                    # Extract topics from combined content (full text + image explanation)
                    extracted_topics = extract_topics_from_text(combined_content)

                    # Ask LLM if the extracted topics are relevant to the question
                    relevance_check_prompt = f"""
                    Extracted Topics: {extracted_topics}
                    User's Question: {prompt}

                    Are these topics relevant to answering the question?
                    Answer "yes" or "no".
                    """

                    relevance_check_data = {
                        "model": "GPT-4Omni",
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are an assistant that determines if topics are relevant to a question.",
                            },
                            {"role": "user", "content": relevance_check_prompt},
                        ],
                        "temperature": 0.0,
                    }

                    try:
                        response = client.chat.completions.create(
                            model="GPT-4Omni",
                            messages=relevance_check_data["messages"],
                        )

                        relevance_answer = (
                            response.model_dump_json()
                            .get("choices", [{}])[0]
                            .get("message", {})
                            .get("content", "no")
                            .strip()
                            .lower()
                        )

                        if relevance_answer == "yes":
                            # Add relevant page to the response
                            relevant_pages.append(
                                {
                                    "doc_name": doc_name,
                                    "page_number": page["page_number"],
                                    "full_text": page.get("full_text", ""),
                                    "image_explanation": image_explanation
                                    or "No image analysis.",
                                }
                            )

                    except Exception as e:
                        st.error(
                            f"Error checking relevance of page {page['page_number']} in '{doc_name}': {e}"
                        )

            if relevant_pages:
                # Combine all relevant pages for the prompt
                relevant_text = "\n".join(
                    f"Document: {data['doc_name']}, Page {data['page_number']}:\n"
                    f"{data['full_text']}\nImage Analysis: {data['image_explanation']}"
                    for data in relevant_pages
                )
                # Query LLM with the relevant text
                try:
                    with st.chat_message("assistant"):
                        response_placeholder = st.empty()
                        full_response = ""

                        # Stream the response
                        stream = client.chat.completions.create(
                            model="GPT-4Omni",
                            messages=[
                                {
                                    "role": "system",
                                    "content": (
                                        "You are an assistant that answers questions strictly based "
                                        "on the provided document content."
                                    ),
                                },
                                {
                                    "role": "user",
                                    "content": f"Context:\n{relevant_text}\n\n{prompt}",
                                },
                            ],
                            stream=True,
                        )

                        for chunk in stream:
                            if chunk.choices and len(chunk.choices) > 0:
                                delta_content = chunk.choices[0].delta.content
                                if delta_content:
                                    full_response += delta_content
                                    response_placeholder.markdown(full_response)

                except Exception as e:
                    st.error(f"Error generating response: {e}")
            else:
                st.warning("No relevant information found in the uploaded documents.")
