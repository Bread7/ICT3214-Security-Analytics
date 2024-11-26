import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

base_model = "NousResearch/Llama-2-7b-chat-hf"
dolly_15K = "databricks/databricks-dolly-15k"
# new_model = "mistral-chat-dolly"
new_model = "llama-2-7b-chat-dolly"
model_id = "mistralai/Mistral-7B-v0.1"

dataset = load_dataset(dolly_15K, split="train")

print(f"Number of prompts: {len(dataset)}")
print(f"Column names are: {dataset.column_names}")


# def create_prompt(row):
#     prompt = f"Instruction: {row['instruction']}\\nContext: {row['context']}\\nResponse: {row['response']}"
#     return prompt


def create_prompt(row):
    return {
        "text": f"Instruction: {row['instruction']}\nContext: {row['context']}\nResponse: {row['response']}"
    }


print(dataset)
data = dataset.map(create_prompt)
# dataset["text"] = dataset.apply(create_prompt, axis=1)
# data = Dataset.from_pandas(dataset)

# Get type
compute_dtype = getattr(torch, "float16")

# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    task_type="CASUAL_LM",
)

# Define training arguments
args = TrainingArguments(
    output_dir="llama-dolly-7b",
    warmup_steps=1,
    num_train_epochs=10,  # adjust based on the data size
    per_device_train_batch_size=2,  # use 4 if you have more GPU RAM
    save_strategy="epoch",  # steps
    logging_steps=50,
    optim="paged_adamw_32bit",
    learning_rate=2.5e-5,
    fp16=True,
    seed=42,
    max_steps=500,
    save_steps=50,  # Save checkpoints every 50 steps
    do_eval=False,
)

# Create trainer
trainer = SFTTrainer(
    model=base_model,
    train_dataset=data,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=args,
    packing=False,
)

trainer.train()

trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)

prompt = " "

input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
outputs = new_model.generate(
    input_ids=input_ids, max_new_tokens=200, do_sample=True, top_p=0.9, temperature=0.1
)
result = tokenizer.batch_decode(
    outputs.detach().cpu().numpy(), skip_special_tokens=True
)[0]
print(result)
