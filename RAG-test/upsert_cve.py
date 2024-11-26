import os
import json
import uuid
from txtai.embeddings import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
import logging
import sys

# -----------------------------
# Configuration and Paths
# -----------------------------

# Absolute paths for clarity
NEW_DATA_DIR = "cvelistV5-main/cves"
INDEX_DIR = "mitre_attack_txtai_index"
MAPPING_PATH = "uid_to_content.json"

# Paths to delta files (optional)
DELTA_JSON_PATH = ""
DELTA_LOG_JSON_PATH = ""

# -----------------------------
# Configure Logging
# -----------------------------

logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("update_txtai_index_debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# -----------------------------
# Initialize Data Structures
# -----------------------------

parsed_data = []          # List to hold data tuples for txtai indexing
uid_to_content = {}      # Dictionary to map UIDs to their corresponding content

# Load existing UID to content mapping if it exists
if os.path.exists(MAPPING_PATH):
    try:
        with open(MAPPING_PATH, "r", encoding="utf-8") as f:
            uid_to_content = json.load(f)
        logger.info(f"Loaded existing UID to content mapping from '{MAPPING_PATH}'.")
    except Exception as e:
        logger.error(f"Failed to load UID mapping from '{MAPPING_PATH}': {e}")
        logger.info("Proceeding with an empty UID to content mapping.")
        uid_to_content = {}
else:
    logger.warning("No existing UID to content mapping found. Starting fresh.")

# -----------------------------
# Initialize Text Splitter
# -----------------------------

text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)

# -----------------------------
# Initialize or Load Existing txtai Embeddings Index
# -----------------------------

def load_existing_index(index_dir, model_path="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Load an existing txtai index.

    Args:
        index_dir (str): Path to the existing index directory.
        model_path (str): Path to the embedding model.

    Returns:
        Embeddings: Loaded Embeddings instance.
    """
    if not os.path.exists(index_dir):
        logger.error(f"Index directory '{index_dir}' does not exist.")
        sys.exit(1)
    try:
        logger.info(f"Loading existing txtai embeddings index from '{index_dir}'...")
        # Initialize with the same model as when the index was created
        config = {"path": model_path}
        embeddings = Embeddings(config)
        embeddings.load(index_dir)
        logger.info(f"Successfully loaded existing index from '{index_dir}'.")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to load existing index from '{index_dir}': {e}")
        user_input = input("Do you want to create a new index and overwrite the existing one? (yes/no): ").strip().lower()
        if user_input == "yes":
            logger.info("Creating a new txtai embeddings index...")
            try:
                embeddings = Embeddings({"path": model_path})
                logger.info("New txtai embeddings index created.")
                return embeddings
            except Exception as ex:
                logger.critical(f"Failed to create a new embeddings index: {ex}")
                sys.exit(1)
        else:
            logger.info("Exiting script to prevent accidental index replacement.")
            sys.exit(1)

# Load the existing index
embeddings = load_existing_index(INDEX_DIR)

# -----------------------------
# Define JSON Processing Functions
# -----------------------------

def process_cve_json(file_path, parsed_data, uid_to_content):
    """
    Process a single CVE JSON file and extract relevant information.

    Args:
        file_path (str): Path to the JSON file.
        parsed_data (list): List to append parsed data tuples.
        uid_to_content (dict): Dictionary to update UID to content mapping.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        logger.debug(f"Processing file: {file_path}")
    except json.JSONDecodeError as e:
        logger.warning(f"JSON decoding error in '{file_path}': {e}")
        return
    except Exception as e:
        logger.warning(f"Unexpected error reading '{file_path}': {e}")
        return

    # Extract CVE Metadata
    cve_metadata = data.get("cveMetadata", {})
    cve_id = cve_metadata.get("cveId", "No CVE ID")
    date_published = cve_metadata.get("datePublished", "No Published Date")
    date_updated = cve_metadata.get("dateUpdated", "No Updated Date")

    # Extract Description
    containers = data.get("containers", {})
    cna = containers.get("cna", {})
    descriptions = cna.get("descriptions", [])
    if descriptions:
        # Assuming the first description is the primary one
        description = descriptions[0].get("value", "No Description")
    else:
        description = "No Description"

    # Extract References
    references = cna.get("references", [])
    reference_urls = [ref.get("url", "") for ref in references if ref.get("url")]

    # Construct Page Content
    page_content = f"""
    CVE ID: {cve_id}
    Published Date: {date_published}
    Last Updated Date: {date_updated}
    Description: {description}
    References: {', '.join(reference_urls) if reference_urls else 'No References'}
    """

    # Split Text into Chunks
    chunks = text_splitter.split_text(page_content)
    logger.debug(f"Generated {len(chunks)} chunks from '{file_path}'.")

    for chunk in chunks:
        unique_id = str(uuid.uuid4())  # Generate a unique UID
        # Ensure UID uniqueness
        while unique_id in uid_to_content:
            unique_id = str(uuid.uuid4())
        metadata = {
            "cve_id": cve_id,
            "published_date": date_published,
            "last_updated_date": date_updated,
            "references": reference_urls
        }
        parsed_data.append((unique_id, chunk, {"content": chunk, **metadata}))
        uid_to_content[unique_id] = chunk
        logger.debug(f"Added chunk with UID: {unique_id}")

def traverse_and_process(directory, parsed_data, uid_to_content):
    """
    Traverse the given directory recursively and process all JSON files.

    Args:
        directory (str): Root directory to start traversal.
        parsed_data (list): List to append parsed data tuples.
        uid_to_content (dict): Dictionary to update UID to content mapping.
    """
    logger.info(f"Traversing directory '{directory}' to process JSON files...")
    json_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".json") and file not in ["delta.json", "deltaLog.json"]:
                json_files.append(os.path.join(root, file))

    logger.info(f"Found {len(json_files)} JSON files to process.")

    for file_path in tqdm(json_files, desc="Processing JSON files"):
        process_cve_json(file_path, parsed_data, uid_to_content)

def process_delta_file(file_path, parsed_data, uid_to_content):
    """
    Process delta.json or deltaLog.json files if they contain CVE records.

    Args:
        file_path (str): Path to the delta JSON file.
        parsed_data (list): List to append parsed data tuples.
        uid_to_content (dict): Dictionary to update UID to content mapping.
    """
    logger.info(f"Processing delta file '{file_path}'...")
    process_cve_json(file_path, parsed_data, uid_to_content)

# -----------------------------
# Traverse and Process JSON Files
# -----------------------------

traverse_and_process(NEW_DATA_DIR, parsed_data, uid_to_content)

# -----------------------------
# Optionally Process Delta Files
# -----------------------------

# Uncomment the following lines if delta.json and deltaLog.json contain CVE records similar to other JSON files
if os.path.exists(DELTA_JSON_PATH):
    process_delta_file(DELTA_JSON_PATH, parsed_data, uid_to_content)

if os.path.exists(DELTA_LOG_JSON_PATH):
    process_delta_file(DELTA_LOG_JSON_PATH, parsed_data, uid_to_content)

# -----------------------------
# Append to Index Using `upsert`
# -----------------------------

if parsed_data:
    logger.info(f"Appending {len(parsed_data)} new entries to the index using `upsert`...")
    try:
        embeddings.upsert(parsed_data)
        logger.info("Appending completed successfully.")
    except Exception as e:
        logger.error(f"Failed to append data to the index using `upsert`: {e}")
        sys.exit(1)
else:
    logger.info("No new data found to append.")

# -----------------------------
# Save Updated Index
# -----------------------------

try:
    embeddings.save(INDEX_DIR)
    logger.info(f"Index saved successfully to '{INDEX_DIR}'.")
except Exception as e:
    logger.error(f"Failed to save the index: {e}")
    sys.exit(1)

# -----------------------------
# Save Updated UID Mapping
# -----------------------------

try:
    with open(MAPPING_PATH, "w", encoding="utf-8") as f:
        json.dump(uid_to_content, f, indent=2)
    logger.info(f"UID to content mapping saved to '{MAPPING_PATH}'.")
except Exception as e:
    logger.error(f"Failed to save UID mapping to '{MAPPING_PATH}': {e}")
    sys.exit(1)

