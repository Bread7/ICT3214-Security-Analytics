import argparse
import json
import logging
import sys

from flask import Flask, jsonify, request
from llama_cpp import Llama
from txtai.embeddings import Embeddings

app = Flask(__name__)


def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        args: Parsed arguments containing the query string.
    """
    parser = argparse.ArgumentParser(
        description="Semantic Search and Llama-based Response Generator"
    )
    parser.add_argument(
        "-q",
        "--query",
        type=str,
        required=False,
        help="The query string to search and generate a response for.",
    )
    parser.add_argument(
        "-n",
        "--num_results",
        type=int,
        default=10,
        help="Number of top search results to retrieve (default: 10).",
    )
    parser.add_argument(
        "-i",
        "--index_path",
        type=str,
        default="mitre_attack_txtai_index",
        help="Path to the txtai embeddings index (default: 'mitre_attack_txtai_index').",
    )
    parser.add_argument(
        "-m",
        "--mapping_path",
        type=str,
        default="uid_to_content.json",
        help="Path to the UID to content mapping JSON file (default: 'uid_to_content.json').",
    )
    parser.add_argument(
        "-model",
        "--model_path",
        type=str,
        default="Mistral-7B-Instruct-v0.3.Q4_K_M.gguf",
        help="Path to the Llama model file (default: 'Mistral-7B-Instruct-v0.3.Q4_K_M.gguf').",
    )
    return parser.parse_args()


def setup_logging():
    """
    Configure the logging settings.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_embeddings(index_path):
    """
    Load the txtai embeddings index.

    Args:
        index_path (str): Path to the txtai embeddings index.

    Returns:
        Embeddings: Loaded Embeddings object.
    """
    try:
        index = Embeddings()
        index.load(index_path)
        logging.info(f"Loaded txtai index from '{index_path}'.")
        return index
    except Exception as e:
        logging.error(f"Failed to load txtai index from '{index_path}': {e}")
        sys.exit(1)


def load_uid_mapping(mapping_path):
    """
    Load the UID to content mapping from a JSON file.

    Args:
        mapping_path (str): Path to the UID to content mapping JSON file.

    Returns:
        dict: UID to content mapping.
    """
    try:
        with open(mapping_path, "r", encoding="utf-8") as f:
            uid_to_content = json.load(f)
        if not isinstance(uid_to_content, dict):
            logging.error("UID to content mapping is not a dictionary.")
            sys.exit(1)
        logging.info(f"Loaded UID to content mapping from '{mapping_path}'.")
        return uid_to_content
    except FileNotFoundError:
        logging.error(f"Mapping file '{mapping_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON format in '{mapping_path}': {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(
            f"Failed to load UID to content mapping from '{mapping_path}': {e}"
        )
        sys.exit(1)


def perform_search(index, query, num_results):
    """
    Perform a semantic search using the txtai embeddings index.

    Args:
        index (Embeddings): Loaded Embeddings object.
        query (str): The query string.
        num_results (int): Number of top search results to retrieve.

    Returns:
        list: List of tuples containing (uid, score).
    """
    try:
        results = index.search(query, num_results)  # Returns list of (uid, score)
        logging.info(f"Search Results: {results}")
        return results
    except Exception as e:
        logging.error(f"Error during search: {e}")
        sys.exit(1)


def retrieve_context(results, uid_to_content):
    """
    Retrieve and process context based on search results.

    Args:
        results (list): List of tuples containing (uid, score).
        uid_to_content (dict): UID to content mapping.

    Returns:
        str: Filtered context string.
    """
    context_lines = []
    for uid, score in results:
        content = uid_to_content.get(uid, "")
        if content:
            context_lines.append(content)
        else:
            logging.warning(f"No content found for UID '{uid}'.")

    if not context_lines:
        logging.error("No relevant context found for the query.")
        sys.exit(1)

    context = "\n".join(context_lines)

    # Remove duplicate lines while preserving order
    filtered_context = "\n".join(dict.fromkeys(context.split("\n")))

    # Debugging: Log the retrieved context
    logging.info("Retrieved Context:")
    logging.info(context)

    logging.info("Retrieved Filtered Context:")
    logging.info(filtered_context)

    return filtered_context


def load_llama_model(model_path):
    """
    Load the Llama model.

    Args:
        model_path (str): Path to the Llama model file.

    Returns:
        Llama: Loaded Llama model object.
    """
    try:
        llm = Llama(model_path=model_path, n_ctx=4092, temperature=0.2)
        logging.info(f"Loaded Llama model from '{model_path}'.")
        return llm
    except Exception as e:
        logging.error(f"Failed to load Llama model from '{model_path}': {e}")
        sys.exit(1)


def generate_prompt(context, query):
    """
    Construct the prompt for the Llama model.

    Args:
        context (str): The filtered context string.
        query (str): The user's query.

    Returns:
        str: The formatted prompt.
    """
    prompt = f"""
### Context:
{context}

### Question:
{query}

### Instructions:
Answer the question strictly based on the context provided above.
Include all relevant details explicitly mentioned in the context.
Do not omit any specific points and avoid generalizations or introducing information not found in the context.
If examples are included in the context, incorporate them into the answer.
Do not mention "in the context".

Do not answer if the context does not explicitly contain the queried term.
If the queried term is widely recognized (e.g., high-profile vulnerabilities or techniques), provide a concise explanation based on your pretrained knowledge. Otherwise, respond with: "The requested term does not exist in the provided context."

When answering questions about technique:
Ensure the retrieved context explicitly mentions the specified technique.
If there are multiple retrieved context with mention of the specified technique,
Always include platform of the technique.
If the specified ID does not appear in the retrieved context, respond with: "The requested technique does not exist in the provided context." Provide no further information.

When answering questions about technique IDs:
Ensure the retrieved context explicitly mentions the specified ID.
Always include platform of the technique ID.
If the specified ID does not appear in the retrieved context, respond with: "The requested ID does not exist in the provided context." Provide no further information.

When answering questions about APT groups:
Ensure the retrieved context explicitly mentions the specified APT group.
Always include aliases of the APT group.
If the specified APT group does not appear in the retrieved context, respond with: "The requested APT group does not exist in the provided context." Provide no further information.

When answering questions about Malware family:
Ensure the retrieved context explicitly mentions the specified Malware family
If the specified Malware family does not appear in the retrieved context, respond with: "The requested Malware family does not exist in the provided context." Provide no further information.

### Answer:

"""
    return prompt


def generate_response(llm, prompt, max_tokens=400):
    """
    Generate a response using the Llama model.

    Args:
        llm (Llama): Loaded Llama model object.
        prompt (str): The prompt to send to the model.
        max_tokens (int): Maximum number of tokens in the generated response.

    Returns:
        str: The generated answer.
    """
    try:
        response = llm(prompt, max_tokens=max_tokens)
        answer = response["choices"][0]["text"].strip()
        logging.info("Model Response:")
        logging.info(answer)
        return answer
    except Exception as e:
        logging.error(f"Error during model inference: {e}")
        sys.exit(1)


@app.route("/", methods=["GET"])
def server_is_running():
    return jsonify({"status": 200, "data": "Threat Intelligence GPT is running!"})


@app.route("/query", methods=["POST"])
def query():
    if request.method != "POST":
        return jsonify({"status": 405, "data": "Only POST method allowed"})
    output = {
        "question_id": "",
        "user_question": "",
        "predicted_domain": "",
        "model_name": "",
        "response_text": "",
    }
    req_data = request.json
    if (
        not req_data["question_id"]
        and not req_data["user_question"]
        and not req_data["predicted_domain"]
        and not req_data["model_name"]
    ):
        return jsonify({"status": 400, "data": "Missing input data"})
    output["question_id"] = req_data["question_id"]
    output["user_question"] = req_data["user_question"]
    output["predicted_domain"] = req_data["predicted_domain"]
    output["model_name"] = req_data["model_name"]

    user_question = req_data["user_question"]
    answer = model_query_process(user_question)
    output["response_text"] = answer
    return jsonify(output)


def model_query_process(user_question):
    # Parse command-line arguments
    args = parse_arguments()

    # Load the txtai index
    index = load_embeddings(args.index_path)

    # Load UID to content mapping
    uid_to_content = load_uid_mapping(args.mapping_path)

    # Perform the search
    results = perform_search(index, user_question, args.num_results)

    # Retrieve and process context
    filtered_context = retrieve_context(results, uid_to_content)

    # Load the Llama model
    llm = load_llama_model(args.model_path)

    # Format the prompt
    prompt = generate_prompt(filtered_context, user_question)

    # Generate response
    answer = generate_response(llm, prompt, max_tokens=400)

    return answer


def main():
    # Setup logging
    setup_logging()

    # Run web server
    app.run(host="0.0.0.0", port=8888)


if __name__ == "__main__":
    main()
