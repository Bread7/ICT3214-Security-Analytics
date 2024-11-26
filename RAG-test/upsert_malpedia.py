import os
import json
import uuid
from txtai.embeddings import Embeddings
import logging
import sys

# -----------------------------
# Configuration and Paths
# -----------------------------

# Paths for JSONL data, index directory, and UID mapping
JSONL_PATH = "malpediaFamilySorted.jsonl"  # Replace with your JSONL file path
INDEX_DIR = "mitre_attack_txtai_index"
MAPPING_PATH = "uid_to_content.json"

# -----------------------------
# Configure Logging
# -----------------------------

logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("upsert_jsonl_debug.log"),
        logging.StreamHandler(sys.stdout)
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
    else:
        logger.info(f"Creating a new txtai index at '{index_dir}'...")
        embeddings = Embeddings({"path": model_path})
    return embeddings

# Load the index
embeddings = load_or_initialize_index(INDEX_DIR)

# Load existing UID to content mapping, or start fresh
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
# Process JSONL File
# -----------------------------

def process_jsonl(jsonl_path, uid_to_content):
    """
    Process a JSONL file to extract input-output pairs and prepare for indexing.

    Args:
        jsonl_path (str): Path to the JSONL file.
        uid_to_content (dict): Dictionary to update UID to content mapping.

    Returns:
        list: List of tuples for txtai upsert [(uid, text, metadata)].
    """
    parsed_data = []
    if not os.path.exists(jsonl_path):
        logger.error(f"JSONL file '{jsonl_path}' does not exist.")
        return parsed_data

    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    input_text = record.get("input", "No Input")
                    output_text = record.get("output", "No Output")
                    unique_id = str(uuid.uuid4())
                    
                    # Ensure UID uniqueness
                    while unique_id in uid_to_content:
                        unique_id = str(uuid.uuid4())
                    
                    # Prepare metadata and text for indexing
                    metadata = {"input": input_text, "output": output_text}
                    text = f"Malware Family: {input_text}\nArticle: {output_text}"
                    parsed_data.append((unique_id, text, metadata))
                    uid_to_content[unique_id] = text
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON decode error for line: {line.strip()} - {e}")
    except Exception as e:
        logger.error(f"Failed to process JSONL file: {e}")

    return parsed_data

# Process the JSONL file
parsed_data = process_jsonl(JSONL_PATH, uid_to_content)

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

# Save updated index
try:
    embeddings.save(INDEX_DIR)
    logger.info(f"Index saved successfully to '{INDEX_DIR}'.")
except Exception as e:
    logger.error(f"Failed to save index: {e}")

# Save updated UID mapping
try:
    with open(MAPPING_PATH, "w", encoding="utf-8") as f:
        json.dump(uid_to_content, f, indent=2)
    logger.info(f"UID to content mapping saved to '{MAPPING_PATH}'.")
except Exception as e:
    logger.error(f"Failed to save UID mapping: {e}")
