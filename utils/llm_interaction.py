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


def get_image_explanation(base64_image, retries=5, initial_delay=2):
    headers = HEADERS
    data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that responds in Markdown.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Explain the contents and figures or tables if present of this image of a document page. The explanation should be concise and semantically meaningful. Do not make assumptions about the specification and be accurate in your explanation.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ],
            },
        ],
        "temperature": 0.0,
    }

    url = f"{azure_endpoint}/openai/deployments/{model}/chat/completions?api-version={api_version}"

    for attempt in range(retries):
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            return (
                response.json()
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "No explanation provided.")
            )

        except requests.exceptions.Timeout as e:
            if attempt < retries - 1:
                wait_time = initial_delay * (2**attempt)
                logging.warning(
                    f"Timeout error. Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{retries})"
                )
                time.sleep(wait_time)
            else:
                logging.error(
                    f"Request failed after {retries} attempts due to timeout: {e}"
                )
                return f"Error: Request timed out after {retries} retries."

        except requests.exceptions.RequestException as e:
            logging.error(f"Error requesting image explanation: {e}")
            return f"Error: Unable to fetch image explanation due to network issues or API error."

    return "Error: Max retries reached without success."


def generate_system_prompt(document_content):
    headers = HEADERS
    preprocessed_content = preprocess_text(document_content)
    data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that serves the task given.",
            },
            {
                "role": "user",
                "content": f"""You are provided with a document. Based on its content, extract and identify the following details:
            Document_content: {preprocessed_content}

            1. **Domain**: Identify the specific domain or field of expertise the document is focused on. Examples include technology, finance, healthcare, law, etc.
            2. **Subject Matter**: Determine the main topic or focus of the document. This could be a detailed concept, theory, or subject within the domain.
            3. **Experience**: Based on the content, infer the level of experience required to understand or analyze the document (e.g., novice, intermediate, expert).
            4. **Expertise**: Identify any specialized knowledge, skills, or proficiency in a particular area that is necessary to evaluate the content.
            5. **Educational Qualifications**: Infer the level of education or qualifications expected of someone who would need to review or write the document (e.g., PhD, Master's, Bachelor's, or certification in a field).
            6. **Style**: Describe the writing style of the document. Is it formal, technical, conversational, academic, or instructional?
            7. **Tone**: Identify the tone used in the document. For example, is it neutral, authoritative, persuasive, or informative?
            8. **Voice**: Analyze whether the voice is active, passive, first-person, third-person, or impersonal, and whether it's personal or objective.

            After extracting this information, use it to fill in the following template:
    
            ---

            You are now assuming a persona based on the content of the provided document. Your persona should reflect the <domain> and <subject matter> of the content, with the requisite <experience>, <expertise>, and <educational qualifications> to analyze the document effectively. Additionally, you should adopt the <style>, <tone> and <voice> present in the document. Your expertise includes:
    
            <Domain>-Specific Expertise:
            - In-depth knowledge and experience relevant to the <subject matter> of the document.
            - Familiarity with the key concepts, terminology, and practices within the <domain>.
            
            Analytical Proficiency:
            - Skilled in interpreting and evaluating the content, structure, and purpose of the document.
            - Ability to assess the accuracy, clarity, and completeness of the information presented.
    
            Style, Tone, and Voice Adaptation:
            - Adopt the writing <style>, <tone>, and <voice> used in the document to ensure consistency and coherence.
            - Maintain the level of formality, technicality, or informality as appropriate to the document’s context.
            
            Your analysis should include:
            - A thorough evaluation of the content, ensuring it aligns with <domain>-specific standards and practices.
            - An assessment of the clarity and precision of the information and any accompanying diagrams or illustrations.
            - Feedback on the strengths and potential areas for improvement in the document.
            - A determination of whether the document meets its intended purpose and audience requirements.
            - Proposals for any necessary amendments or enhancements to improve the document’s effectiveness and accuracy.
        
            ---

            Generate a response filling the template with appropriate details based on the content of the document and return the filled in template as response.""",
            },
        ],
        "temperature": 0.5,
    }

    try:
        response = requests.post(
            f"{azure_endpoint}/openai/deployments/{model}/chat/completions?api-version={api_version}",
            headers=headers,
            json=data,
            timeout=20,
        )
        response.raise_for_status()
        prompt_response = (
            response.json()
            .get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        return prompt_response.strip()

    except requests.exceptions.RequestException as e:
        logging.error(f"Error generating system prompt: {e}")
        return f"Error: Unable to generate system prompt due to network issues or API error."


def summarize_page(
    page_text,
    previous_summary,
    page_number,
    system_prompt,
    max_retries=5,
    base_delay=1,
    max_delay=32,
):
    headers = HEADERS
    preprocessed_page_text = preprocess_text(page_text)
    preprocessed_previous_summary = preprocess_text(previous_summary)

    prompt_message = (
        f"Please rewrite the following page content from (Page {page_number}) along with context from the previous page summary "
        f"to make them concise and well-structured. Maintain proper listing and referencing of the contents if present."
        f"Do not add any new information or make assumptions. Keep the meaning accurate and the language clear.\n\n"
        f"Previous page summary: {preprocessed_previous_summary}\n\n"
        f"Current page content:\n{preprocessed_page_text}\n"
    )

    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_message},
        ],
        "temperature": 0.0,
    }

    attempt = 0
    while attempt < max_retries:
        try:
            response = requests.post(
                f"{azure_endpoint}/openai/deployments/{model}/chat/completions?api-version={api_version}",
                headers=headers,
                json=data,
                timeout=50,
            )
            response.raise_for_status()
            logging.info(
                f"Summary retrieved for page {page_number} at {time.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            return (
                response.json()
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "No summary provided.")
                .strip()
            )

        except requests.exceptions.RequestException as e:
            attempt += 1
            if attempt >= max_retries:
                logging.error(f"Error summarizing page {page_number}: {e}")
                return f"Error: Unable to summarize page {page_number} due to network issues or API error."

            delay = min(max_delay, base_delay * (2**attempt))
            jitter = random.uniform(0, delay)
            logging.warning(
                f"Retrying in {jitter:.2f} seconds (attempt {attempt}) due to error: {e}"
            )
            time.sleep(jitter)




def ask_question(documents, question, chat_history):
    headers = HEADERS
    preprocessed_question = preprocess_text(question)

    def calculate_token_count(text):
        return len(text.split())

    # Calculate initial total tokens from question
    total_tokens = calculate_token_count(preprocessed_question)

    def check_page_relevance_batch(pages_batch):
        relevant_pages = []
        for doc_name, page in pages_batch:
            page_full_text = page.get("full_text", "")
            image_explanation = (
                "\n".join(
                    f"Page {img['page_number']}: {img['explanation']}"
                    for img in page.get("image_analysis", [])
                )
                or "No image analysis."
            )

            relevance_check_prompt = f"""
            Document: {doc_name}, Page {page['page_number']}
            Full Text: {page_full_text[:3000]}  # Limit prompt text
            Image Analysis: {image_explanation[:1000]}

            Question: {preprocessed_question}

            Answer "yes" if this page has relevant information; otherwise, "no".
            """

            relevance_data = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "Determine if a page is relevant to a question.",
                    },
                    {"role": "user", "content": relevance_check_prompt},
                ],
                "temperature": 0.0,
            }

            try:
                response = requests.post(
                    f"{azure_endpoint}/openai/deployments/{model}/chat/completions?api-version={api_version}",
                    headers=headers,
                    json=relevance_data,
                    timeout=120,
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
                    relevant_pages.append({
                        "doc_name": doc_name,
                        "page_number": page["page_number"],
                        "full_text": page_full_text,
                        "image_explanation": image_explanation,
                    })

            except requests.exceptions.RequestException as e:
                logging.error(f"Error checking relevance of page {page['page_number']} in '{doc_name}': {e}")

        return relevant_pages

    # Divide pages into batches for relevance checks
    pages_batch = [
        (doc_name, page)
        for doc_name, doc_data in documents.items()
        for page in doc_data["pages"]
    ]
    relevant_pages = []
    batch_size = 5  # Process 5 pages per request
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_batch = {
            executor.submit(check_page_relevance_batch, pages_batch[i:i + batch_size]): i
            for i in range(0, len(pages_batch), batch_size)
        }
        for future in concurrent.futures.as_completed(future_to_batch):
            relevant_pages.extend(future.result())

    if not relevant_pages:
        return "No relevant content found.", total_tokens

    # Layered summarization to stay within token limits
    def layered_summarization(content):
        tokens = calculate_token_count(content)
        while tokens > 125000:
            summary_prompt = f"""
            Summarize the following to reduce tokens but retain core information:

            {content[:5000]}  # Limit per summarization request
            """
            summary_data = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "Summarize content to reduce token count while keeping core information.",
                    },
                    {"role": "user", "content": summary_prompt},
                ],
                "temperature": 0.0,
            }

            try:
                response = requests.post(
                    f"{azure_endpoint}/openai/deployments/{model}/chat/completions?api-version={api_version}",
                    headers=headers,
                    json=summary_data,
                    timeout=60,
                )
                response.raise_for_status()
                content = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                tokens = calculate_token_count(content)
            except requests.exceptions.RequestException as e:
                logging.error(f"Error during summarization: {e}")
                break
        return content

    # Prepare combined content for final response
    combined_relevant_content = "\n".join(
        f"Document: {page['doc_name']}, Page {page['page_number']}\nFull Text: {page['full_text']}\nImage Analysis: {page['image_explanation']}"
        for page in relevant_pages
    )

    if calculate_token_count(combined_relevant_content) > 125000:
        combined_relevant_content = layered_summarization(combined_relevant_content)

    conversation_history = "".join(
        f"User: {preprocess_text(chat['question'])}\nAssistant: {preprocess_text(chat['answer'])}\n"
        for chat in chat_history
    )

    prompt_message = f"""
        You have the following relevant content:

        {combined_relevant_content}

        Previous conversation history: {conversation_history}

        Answer based strictly on the provided content.

        Question: {preprocessed_question}
    """

    final_data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You answer questions based on provided documents.",
            },
            {"role": "user", "content": prompt_message},
        ],
        "temperature": 0.0,
    }

    try:
        response = requests.post(
            f"{azure_endpoint}/openai/deployments/{model}/chat/completions?api-version={api_version}",
            headers=headers,
            json=final_data,
            timeout=120,
        )
        response.raise_for_status()
        answer_content = (
            response.json()
            .get("choices", [{}])[0]
            .get("message", {})
            .get("content", "No answer provided.")
            .strip()
        )
        response_tokens = calculate_token_count(answer_content)
        total_tokens += response_tokens

        return answer_content, total_tokens

    except requests.exceptions.RequestException as e:
        logging.error(f"Error answering question '{question}': {e}")
        return "Error processing question.", total_tokens
