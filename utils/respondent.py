import requests
from utils.config import azure_endpoint, api_key, api_version, model
import logging
import time
import random
import re
import nltk
from nltk.corpus import stopwords
import tiktoken
import concurrent.futures
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import numpy as np

logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s [%(levelname)s] %(message)s"
)
nltk.download("stopwords", quiet=True)

HEADERS = {"Content-Type": "application/json", "api-key": api_key}


def count_tokens(text, model="gpt-4o"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    stop_words = set(stopwords.words("english"))
    text = " ".join([word for word in text.split() if word not in stop_words])

    return text


def is_summary_request(question):
    summary_check_prompt = f"""
        The user asked the question: {question}
        
        Determine if this question is about requesting a complete summary of the entire document, tell about the document or any request similar to that.
        Answer "yes" or "no".
        """
    response = requests.post(
        f"{azure_endpoint}/openai/deployments/{model}/chat/completions?api-version={api_version}",
        headers=HEADERS,
        json={
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an assistant that detects summary requests.",
                },
                {"role": "user", "content": summary_check_prompt},
            ],
            "temperature": 0.5,
        },
    )
    return (
        response.json()
        .get("choices", [{}])[0]
        .get("message", {})
        .get("content", "no")
        .strip()
        .lower()
        == "yes"
    )


def extract_topics_from_text(text, max_topics=50, max_top_words=50):
    try:
        # Adjust the number of features dynamically based on the text length
        max_features = min(1000, len(text.split()))
        vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features)
        tfidf = vectorizer.fit_transform([text])
        
        # Dynamically set the number of topics to avoid overfitting small texts
        n_topics = min(max_topics, tfidf.shape[1])
        nmf = NMF(n_components=n_topics, random_state=42, max_iter=500)
        nmf.fit(tfidf)

        feature_names = vectorizer.get_feature_names_out()
        
        # Ensure we capture as many words per topic as possible
        n_top_words = min(max_top_words, len(feature_names))
        topics = [
            ", ".join([feature_names[i] for i in topic.argsort()[-n_top_words:][::-1]])
            for topic in nmf.components_
        ]
        return " | ".join(topics)
    except Exception as e:
        logging.error(f"Error extracting topics: {e}")
        return "Error extracting topics."


# Update the relevance check function
def check_page_relevance(doc_name, page, preprocessed_question):
    page_full_text = page.get("full_text", "No full text available")
    page_summary = page.get("text_summary", "No summary available for this page")
    extracted_topics = extract_topics_from_text(page_full_text, 50, 50)

    image_explanation = (
        "\n".join(
            f"Page {img['page_number']}: {img['explanation']}"
            for img in page.get("image_analysis", [])
        )
        or "No image analysis."
    )

    relevance_check_prompt = f"""
    Here's the extracted topics and image analysis of a page:

    Document: {doc_name}, Page {page['page_number']}
    Extracted Topics: {extracted_topics}
    Image Analysis: {image_explanation}

    Question asked by user: {preprocessed_question}

    Respond with "yes" if this page contains any relevant information related to the user's question, even if only a small part of the page has relevant content. Otherwise, respond with "no".
    """

    relevance_data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are an assistant that determines if a page is relevant to a question.",
            },
            {"role": "user", "content": relevance_check_prompt},
        ],
        "temperature": 0.0,
    }

    for attempt in range(5):
        try:
            response = requests.post(
                f"{azure_endpoint}/openai/deployments/{model}/chat/completions?api-version={api_version}",
                headers=HEADERS,
                json=relevance_data,
                timeout=60,
            )
            response.raise_for_status()
            relevance_answer = (
                response.json()
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "no")
                .strip()
                .lower()
            )
            if relevance_answer == "yes":
                return {
                    "doc_name": doc_name,
                    "page_number": page["page_number"],
                    "page_summary": page_summary,
                    "image_explanation": image_explanation,
                }
                
        except requests.exceptions.RequestException as e:
            logging.error(
                f"Error checking relevance of page {page['page_number']} in '{doc_name}': {e}"
            )

            backoff_time = (2**attempt) + random.uniform(0, 1)
            time.sleep(backoff_time)

    return None


def summarize_pages_in_batches(pages, batch_size=10):
    summaries = []
    for i in range(0, len(pages), batch_size):
        batch_pages = pages[i : i + batch_size]
        combined_batch_text = "\n".join(
            f"Page {page['page_number']}, Full Text: {page.get('full_text', '')}\nImage Explanation: {page.get('image_explanation', '')}"
            for page in batch_pages
        )

        # Apply NMF for topic extraction
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(
            [page.get("full_text", "") for page in batch_pages]
        )
        nmf_model = NMF(n_components=3, random_state=42)
        nmf_topics = nmf_model.fit_transform(tfidf_matrix)

        # Extract top terms for each topic
        topic_terms = []
        for topic_idx, topic in enumerate(nmf_model.components_):
            topic_terms.append(
                [
                    vectorizer.get_feature_names_out()[i]
                    for i in topic.argsort()[: -5 - 1 : -1]
                ]
            )

        all_prominent_terms = [term for sublist in topic_terms for term in sublist]

        batch_summary_prompt = f"""
                Summarize the following content concisely while retaining the key points and mention range of pages given as input:
                Additionally, ensure that the summary reflects the following prominent terms derived from the content:
                Present the summary in a proper human readable format using subheadings and bullets wherever necessary. Don't mention as summary:
                {', '.join(all_prominent_terms)}

                {combined_batch_text}
                """
        batch_summary_data = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an assistant that creates concise summaries.",
                },
                {"role": "user", "content": batch_summary_prompt},
            ],
            "temperature": 0.0,
        }

        for attempt in range(5):
            try:
                response = requests.post(
                    f"{azure_endpoint}/openai/deployments/{model}/chat/completions?api-version={api_version}",
                    headers=HEADERS,
                    json=batch_summary_data,
                    timeout=60,
                )
                response.raise_for_status()
                batch_summary = (
                    response.json()
                    .get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    .strip()
                )
                summaries.append(batch_summary)
                break
            except requests.exceptions.RequestException as e:
                logging.error(f"Error summarizing batch starting at page {i}: {e}")
                backoff_time = (2**attempt) + random.uniform(0, 1)
                time.sleep(backoff_time)

    return "\n\n".join(summaries)


def is_detailed_summary_request(question):
    headers = HEADERS
    # LLM prompt to classify the intent
    intent_prompt = f"""
    You are an assistant that classifies user intents. The user's question will be provided, 
    and you must determine if the question explicitly asks for a detailed summary, 
    pagewise summary, topic-wise summary any request similar to that. 

    User's question: {question}

    Respond with only "yes" or "no".
    """

    # Prepare data for the LLM request
    data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that classifies intents.",
            },
            {"role": "user", "content": intent_prompt},
        ],
        "temperature": 0.5,
    }

    try:
        # Make a request to the LLM
        response = requests.post(
            f"{azure_endpoint}/openai/deployments/{model}/chat/completions?api-version={api_version}",
            headers=headers,
            json=data,
            timeout=60,
        )
        response.raise_for_status()
        return (
            response.json()
            .get("choices", [{}])[0]
            .get("message", {})
            .get("content", "no")
            .strip()
            .lower()
            == "yes"
        )
        # return llm_response.lower() == "yes"

    except requests.exceptions.RequestException as e:
        logging.error(f"Error determining intent: {e}")
        return False


def ask_question(documents, question, chat_history):
    headers = HEADERS
    preprocessed_question = preprocess_text(question)

    # Check for summary-related intents
    if is_summary_request(preprocessed_question):
        # Handle specific types of summaries
        if not is_detailed_summary_request(preprocessed_question):

            # Combine all pages into a single text corpus
            combined_text = "\n".join(
                page.get("full_text", "")
                for doc_data in documents.values()
                for page in doc_data["pages"]
            )

            # Perform NMF topic modeling
            vectorizer = TfidfVectorizer(stop_words="english")
            tfidf_matrix = vectorizer.fit_transform([combined_text])
            nmf_model = NMF(n_components=5, random_state=1)
            nmf_topics = nmf_model.fit_transform(tfidf_matrix)

            # Extract prominent terms for each topic
            topic_terms = []
            for topic_idx, topic in enumerate(nmf_model.components_):
                topic_terms.append(
                    [
                        vectorizer.get_feature_names_out()[i]
                        for i in topic.argsort()[:-6:-1]
                    ]
                )

            # Flatten the terms from all topics
            all_prominent_terms = [term for sublist in topic_terms for term in sublist]

            # Define a function to calculate topic relevance for a page
            def get_page_topic_relevance(page_text):
                page_vectorized = vectorizer.transform([page_text])
                topic_scores = nmf_model.transform(page_vectorized)
                return sum(topic_scores[0])

            # Select relevant pages based on topic relevance
            relevant_page_summaries = [
                page.get("text_summary", "")
                for doc_name, doc_data in documents.items()
                for page in doc_data["pages"]
                if get_page_topic_relevance(page.get("full_text", "")) > 0
            ]

            # Combine relevant page summaries
            combined_summary_prompt = f"""
            Combine the following summaries into a single, comprehensive summary of the document.
            Ensure the summary is thorough yet concise, presenting the key points in a structured, readable format using subheaders and bullets that highlight major themes and strategies:

            {' '.join(relevant_page_summaries)}
            """

            # Prepare request for LLM
            final_summary_data = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an assistant that creates a document summary.",
                    },
                    {"role": "user", "content": combined_summary_prompt},
                ],
                "temperature": 0.0,
            }

            # Call the LLM to generate the final summary
            final_response = requests.post(
                f"{azure_endpoint}/openai/deployments/{model}/chat/completions?api-version={api_version}",
                headers=headers,
                json=final_summary_data,
            )
            final_summary = (
                final_response.json()
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "No summary provided.")
            )

            total_tokens = count_tokens(combined_summary_prompt+final_summary)

            return final_summary, total_tokens

        else:
            # Perform batch summarization if it's a specific summary request
            all_pages = [
                page for doc_data in documents.values() for page in doc_data["pages"]
            ]
            final_summary = summarize_pages_in_batches(all_pages)
            total_tokens = count_tokens(all_pages)
            return final_summary, total_tokens

    # If not a summary request, proceed with question answering
    total_tokens = count_tokens(preprocessed_question)

    for doc_name, doc_data in documents.items():
        for page in doc_data["pages"]:
            total_tokens += count_tokens(
                page.get("full_text", "No full text available")
            )

    relevant_pages = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_page = {
            executor.submit(
                check_page_relevance, doc_name, page, preprocessed_question
            ): (doc_name, page, preprocessed_question)
            for doc_name, doc_data in documents.items()
            for page in doc_data["pages"]
        }

        for future in concurrent.futures.as_completed(future_to_page):
            result = future.result()
            if result:
                relevant_pages.append(result)

    if not relevant_pages:
        return (
            "The content of the provided documents does not contain an answer to your question.",
            total_tokens,
        )

    relevant_pages_content = "\n".join(
        f"Document: {page['doc_name']}, Page {page['page_number']}\nSummary: {page['page_summary']}\nImage Analysis: {page['image_explanation']}"
        for page in relevant_pages
    )
    relevant_tokens = count_tokens(relevant_pages_content)

    combined_relevant_content = (
        relevant_pages_content
        if relevant_tokens <= 125000
        else "Content is too large to process."
    )

    conversation_history = "".join(
        f"User: {preprocess_text(chat['question'])}\nAssistant: {preprocess_text(chat['answer'])}\n"
        for chat in chat_history
    )

    prompt_message = f"""
        You are given the following relevant content from multiple documents:

        ---
        {combined_relevant_content}
        ---

        Previous responses over the current chat session: {conversation_history}

        Answer the following question based **strictly and only** on the factual information provided in the content above.
        Carefully verify all details from the content and do not generate any information that is not explicitly mentioned in it.
        Ensure the response is clearly formatted for readability using subheadings and bullets if necessary.

        Question: {preprocessed_question}
        """

    final_data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are an assistant that answers questions based only on provided knowledge base.",
            },
            {"role": "user", "content": prompt_message},
        ],
        "temperature": 0.0,
    }

    for attempt in range(5):
        try:
            response = requests.post(
                f"{azure_endpoint}/openai/deployments/{model}/chat/completions?api-version={api_version}",
                headers=headers,
                json=final_data,
                timeout=60,
            )
            response.raise_for_status()
            answer_content = (
                response.json()
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "No answer provided.")
                .strip()
            )

            total_tokens = count_tokens(prompt_message+answer_content)
            return answer_content, total_tokens

        except requests.exceptions.RequestException as e:
            logging.error(f"Error answering question '{question}': {e}")
            backoff_time = (2**attempt) + random.uniform(0, 1)
            time.sleep(backoff_time)

    return "Error processing question.", total_tokens
