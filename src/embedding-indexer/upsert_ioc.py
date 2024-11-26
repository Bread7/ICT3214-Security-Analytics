import os
import json
import uuid
from txtai.embeddings import Embeddings
import logging
import sys

# -----------------------------
# Configuration and Paths
# -----------------------------

# Path to the JSON file containing the data
JSON_PATH = "urlhaus-modified.json"  # Replace with your JSON file path

# Directory where the txtai index will be stored
INDEX_DIR = "mitre_attack_txtai_index"

# Path to the JSON file that maps unique IDs to content
MAPPING_PATH = "uid_to_content.json"

# -----------------------------
# Configure Logging
# -----------------------------

logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for detailed logs; change to INFO in production
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("upsert_json_debug.log"),  # Log file for debugging
        logging.StreamHandler(sys.stdout)              # Log output to console
    ]
)

logger = logging.getLogger(__name__)

# -----------------------------
# Load or Initialize Index and UID Mapping
# -----------------------------

def load_or_initialize_index(index_dir, model_path="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Load an existing txtai index or create a new one if it doesn't exist.

    Args:
        index_dir (str): Path to the index directory.
        model_path (str): Path to the embedding model.

    Returns:
        Embeddings: txtai embeddings instance.
    """
    if os.path.exists(index_dir):
        logger.info(f"Loading existing txtai index from '{index_dir}'...")
        embeddings = Embeddings({"path": model_path})
        embeddings.load(index_dir)
        logger.debug("Index loaded successfully.")
    else:
        logger.info(f"Creating a new txtai index at '{index_dir}'...")
        embeddings = Embeddings({"path": model_path})
        logger.debug("New index initialized.")
    return embeddings

# Initialize the txtai embeddings
embeddings = load_or_initialize_index(INDEX_DIR)

# Load existing UID to content mapping, or initialize a new one
if os.path.exists(MAPPING_PATH):
    try:
        with open(MAPPING_PATH, "r", encoding="utf-8") as f:
            uid_to_content = json.load(f)
        logger.info(f"Loaded UID to content mapping from '{MAPPING_PATH}'.")
    except Exception as e:
        logger.error(f"Failed to load UID mapping: {e}")
        uid_to_content = {}
else:
    logger.info("No existing UID mapping found. Starting fresh.")
    uid_to_content = {}

# -----------------------------
# Process JSON File
# -----------------------------

def process_json(json_path, uid_to_content):
    """
    Process a JSON file to extract 'input' fields and prepare for indexing.

    Args:
        json_path (str): Path to the JSON file.
        uid_to_content (dict): Dictionary to update UID to content mapping.

    Returns:
        list: List of tuples for txtai upsert [(uid, text, metadata)].
    """
    parsed_data = []

    if not os.path.exists(json_path):
        logger.error(f"JSON file '{json_path}' does not exist.")
        return parsed_data

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

            if not isinstance(data, list):
                logger.error(f"JSON file '{json_path}' does not contain a list of objects.")
                return parsed_data

            logger.info(f"Processing {len(data)} records from '{json_path}'.")

            for idx, item in enumerate(data, start=1):
                input_text = item.get("input", "").strip()
                
                if not input_text:
                    logger.warning(f"Record {idx} missing 'input' field. Skipping.")
                    continue

                unique_id = str(uuid.uuid4())

                # Ensure UID uniqueness (optional, as UUID4 collisions are extremely rare)
                while unique_id in uid_to_content:
                    unique_id = str(uuid.uuid4())

                # Prepare metadata and text for indexing
                metadata = {"input": input_text}
                text = input_text  # Since only 'input' is needed

                parsed_data.append((unique_id, text, metadata))
                uid_to_content[unique_id] = text

                if idx % 1000 == 0:
                    logger.debug(f"Processed {idx} records.")

    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
    except Exception as e:
        logger.error(f"Failed to process JSON file: {e}")

    logger.info(f"Processed {len(parsed_data)} valid records for upserting.")
    return parsed_data

# Process the JSON file and prepare data for upserting
parsed_data = process_json(JSON_PATH, uid_to_content)

# -----------------------------
# Upsert Data into Index
# -----------------------------

if parsed_data:
    logger.info(f"Upserting {len(parsed_data)} entries into the index...")
    try:
        embeddings.upsert(parsed_data)
        logger.info("Upsert completed successfully.")
    except Exception as e:
        logger.error(f"Failed to upsert data into index: {e}")
else:
    logger.info("No new data to upsert.")

# -----------------------------
# Save Updated Index and Mapping
# -----------------------------

# Save updated index to disk
try:
    embeddings.save(INDEX_DIR)
    logger.info(f"Index saved successfully to '{INDEX_DIR}'.")
except Exception as e:
    logger.error(f"Failed to save index: {e}")

# Save updated UID mapping to JSON file
try:
    with open(MAPPING_PATH, "w", encoding="utf-8") as f:
        json.dump(uid_to_content, f, indent=2)
    logger.info(f"UID to content mapping saved to '{MAPPING_PATH}'.")
except Exception as e:
    logger.error(f"Failed to save UID mapping: {e}")
