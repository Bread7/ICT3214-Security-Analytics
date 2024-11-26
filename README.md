# Threat Intelligence GPT

This uses [mistral Q4_K_M model](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) with RAG to repurpose the model with new data for threat intelligence querying.

## Installation

### Environment (ARM sillicon)

Needed to install tensorflow

```bash
brew install cmake pkg-config
```

### Environment (x86_64)

Recommendation: Use python virutal env or conda

```bash
pip install -r requirements.txt
pip install -r web-requirements.txt
```

## Run the server

```bash
hugging-face-cli login # (get tokens from: https://huggingface.co/settings/tokens)
python src/app.py
```

## Evaluations

Test cases and scripts are found in `src/evaluations`
