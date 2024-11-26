import json

def extract_input_fields(input_file, output_file):
    """
    Extracts only the 'input' field from each object in a JSON array.

    Args:
        input_file (str): Path to the original JSON file.
        output_file (str): Path to the new JSON file with only 'input' fields.
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"Loaded {len(data)} records from '{input_file}'.")
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON: {e}")
        return
    except FileNotFoundError:
        print(f"File '{input_file}' not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading '{input_file}': {e}")
        return

    # Extract only the 'input' field
    extracted_data = []
    for idx, item in enumerate(data, start=1):
        input_text = item.get('input')
        if input_text is not None:
            extracted_data.append({'input': input_text})
        else:
            print(f"Warning: Record {idx} missing 'input' field.")

    print(f"Extracted 'input' fields from {len(extracted_data)} records.")

    # Write the extracted data to the new JSON file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, indent=2)
        print(f"Successfully wrote extracted data to '{output_file}'.")
    except Exception as e:
        print(f"An error occurred while writing to '{output_file}': {e}")

if __name__ == "__main__":
    input_json = "urlhaus-modified.json"  # Replace with your input JSON file path
    output_json = "urlhaus.json"  # Desired output file path
    extract_input_fields(input_json, output_json)
