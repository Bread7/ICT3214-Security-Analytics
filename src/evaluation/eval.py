import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score
import evaluate

# Text Normalization Function
def normalize_text(text):
    """
    Normalize text by removing extra spaces, newlines, and ensuring consistent formatting.
    """
    return " ".join(text.replace("\n", " ").split()).strip()

# Load JSON Data
def load_data(filepath):
    """
    Load the evaluation data from a JSON file.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

# Preprocess Dataset
def preprocess_data(data):
    """
    Normalize the 'generated' and 'reference' fields in the dataset.
    """
    for item in data:
        item["generated"] = normalize_text(item["generated"])
        item["reference"] = normalize_text(item["reference"])
    return data

# BLEU Score Evaluation
def evaluate_bleu(data):
    """
    Compute BLEU scores for the dataset.
    """
    bleu_scores = []
    smooth_fn = SmoothingFunction().method1

    for item in data:
        reference = item["reference"].split()
        generated = item["generated"].split()
        bleu_score = sentence_bleu([reference], generated, smoothing_function=smooth_fn)
        bleu_scores.append(bleu_score)

    return sum(bleu_scores) / len(bleu_scores)

# ROUGE Score Evaluation
def evaluate_rouge(data):
    """
    Compute ROUGE scores for the dataset.
    """
    rouge = evaluate.load("rouge")
    references = [item["reference"] for item in data]
    generated = [item["generated"] for item in data]

    return rouge.compute(predictions=generated, references=references)

# BERTScore Evaluation
def evaluate_bert(data):
    """
    Compute BERTScore for the dataset.
    """
    references = [item["reference"] for item in data]
    generated = [item["generated"] for item in data]

    P, R, F1 = score(generated, references, lang="en", verbose=True)
    return {"precision": P.mean().item(), "recall": R.mean().item(), "f1": F1.mean().item()}

# Main Evaluation Script
def main():
    # Filepath for the dataset
    filepath = "evalBASE.json"

    # Load and preprocess data
    data = load_data(filepath)
    data = preprocess_data(data)

    # Compute BLEU
    avg_bleu = evaluate_bleu(data)
    

    # Compute ROUGE
    rouge_scores = evaluate_rouge(data)
    

    # Compute BERTScore
    bert_scores = evaluate_bert(data)

    # Print Scores
    print(f"Average BLEU Score: {avg_bleu:.4f}")
    print(f"ROUGE Scores: {rouge_scores}")
    print(f"BERTScore: Precision: {bert_scores['precision']:.4f}, Recall: {bert_scores['recall']:.4f}, F1: {bert_scores['f1']:.4f}")

if __name__ == "__main__":
    main()
