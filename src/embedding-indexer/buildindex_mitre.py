import os
import json
import uuid
from txtai.embeddings import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Directory containing all the JSON files
json_dir = "attack-stix-data/attack"
parsed_data = []
uid_to_content = {}  # Mapping from UID to content

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)

# Parse the JSON files
for filename in os.listdir(json_dir):
    if filename.endswith(".json"):
        file_path = os.path.join(json_dir, filename)
        with open(file_path, "r", encoding="utf-8") as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from {file_path}: {e}")
                continue
            for item in data.get("objects", []):
                if item.get("type") == "attack-pattern":
                    name = item.get("name", "No Name")
                    description = item.get("description", "No Description")
                    platform = item.get("x_mitre_platforms", "No Platform")
                    detection = item.get("x_mitre_detection", "No Detection")
                    external_references = item.get("external_references", [])
                    external_id = next(
                        (ref.get("external_id") for ref in external_references if "external_id" in ref),
                        "No External ID"
                    )
                    if external_id == "No External ID" or name == "No Name":
                        continue
                    page_content = f"ID: {external_id}\nName: {name}\nDescription: {description}\nPlatform: {platform}\nDetection {detection}"
                    chunks = text_splitter.split_text(page_content)
                    for chunk in chunks:
                        unique_id = str(uuid.uuid4())  # Generate a unique UID
                        parsed_data.append((unique_id, chunk, {"content": chunk, "name": name, "external_id": external_id, "platform": platform, "detection": detection}))
                        uid_to_content[unique_id] = chunk  # Map UID to content

                if item.get("type") == "intrusion-set":
                    name = item.get("name", "No Name")
                    description = item.get("description", "No Description")
                    aliases = item.get("aliases", "No Alias")
                    external_references = item.get("external_references", [])
                    external_id = next(
                        (ref.get("external_id") for ref in external_references if "external_id" in ref),
                        "No External ID"
                    )
                    if external_id == "No External ID" or name == "No Name":
                        continue
                    page_content = f"ID: {external_id}\nName: {name}\nDescription: {description}\nAliases: {aliases}"
                    chunks = text_splitter.split_text(page_content)
                    for chunk in chunks:
                        unique_id = str(uuid.uuid4())  # Generate a unique UID
                        parsed_data.append((unique_id, chunk, {"content": chunk, "name": name, "external_id": external_id, "aliases": aliases}))
                        uid_to_content[unique_id] = chunk  # Map UID to content

# Initialize txtai embeddings index
index = Embeddings({"path": "sentence-transformers/all-MiniLM-L6-v2"})

# Add parsed data to the index
print("Indexing data...")
index.index(parsed_data)

# Save the index
index_path = "mitre_attack_txtai_index"
index.save(index_path)
print(f"Indexing complete. Index saved to '{index_path}'.")

# Save the UID to content mapping
mapping_path = "uid_to_content.json"
with open(mapping_path, "w", encoding="utf-8") as f:
    json.dump(uid_to_content, f)
print(f"UID to content mapping saved to '{mapping_path}'.")
