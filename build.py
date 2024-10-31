import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

# Loading dataset
dataset = load_dataset("TokenBender/code_instructions_122k_alpaca_style", split="train")

# Format dataset


## Output right format for each row in dataset
# def create_text_row(instruction, output, text):
#     text_row = f"""<s>[INST] {instruction} here are the inputs {input} [/INST] \\n {output} </s>"""
#     return text_row
#
#
# ## Iterate all rows, format dataset and store in jsonl file
# def process_jsonl_file(output_file_path):
#     with open(output_file_path, "w") as output_jsonl_file:
#         for item in dataset:
#             json_object = {
#                 "text": create_text_row(
#                     item["instruction"], item["input"], item["output"]
#                 ),
#                 "instruction": item["instruction"],
#                 "input": item["input"],
#                 "output": item["output"],
#             }
#             output_jsonl_file.write(json.dumps(json_object) + "\\n")
#
#
# process_jsonl_file("./training_dataset.jsonl")

# Load training dataset
# train_dataset = load_dataset(
#     "json", data_files="./training_dataset.jsonl", split="train"
# )
train_dataset = dataset

# Model params
new_model = "newmistral"

## Lora params
lora_r = 64  # LoRA attention dimension
lora_alpha = 16  # Alpha parameter for LoRA scaling
lora_dropout = 0.1  # Dropout probability for LoRA layers

## bitsandbytes params
use_4bit = True  # 4-bit precision base model loading
bnb_4bit_compute_dtype = "float16"  # Compute dtype for 4-bit models
bnb_4bit_quant_type = "nf4"  # Quant type fp4/nf4
use_nested_quant = False  # Activate nested quantization for 4-bit models (double quant)

## Training args
output_dir = "./results"
num_train_epochs = 1  # Training epochs
fp16 = False
bf16 = False
per_device_train_batch_size = 1  # Batch size per GPU training
per_device_eval_batch_size = 1  # Batch size per GPU evaluation
gradient_accumulation_steps = 1  # Number of update steps to accumulate gradients for
gradient_checkpointing = True  # Enabled gradient checkpointing
max_grad_norm = 0.3  # Maximum gradient normal
learning_rate = 2e-4  # Initial learning rate
weight_decay = 0.001  # Apply to all layers except bias/LayerNorm weights
optim = "paged_adamw_32bit"  # Optimizer to use
lr_scheduler_type = "constant"  # Learning rate schedule
max_steps = -1  # Number of training steps
warmup_ratio = 0.03  # Ratio of steps for linear warmup (from 0 to learning rate)
group_by_length = True  # Group sequences into batches with same length
save_steps = 50  # Save at every update steps
logging_steps = 50  # Log every update steps

## SFT params
max_seq_length = None  # Maximum sequence length to use
packing = False  # Pack multiple short examples in the same input sequence to increase efficiency
device_map = {"": 0}  # Load entire model on GPU 0

# Load base model
model_name = "mistralai/Mistral-7B-Instruct-v0.1"

compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Fine tune with qLora and Supervised finetuning
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CASUAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
)

training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=100,  # the total number of training steps to perform
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard",
)

trainer = SFTTrainer(
    model=base_model,
    train_dataset=train_dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)

trainer.train()
trainer.model.save_pretrained(new_model)

# Infer fine-tuned model
eval_prompt = """print hello world in python"""
model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
model.eval()
with torch.no_grad():
    generated_code = tokenizer.decocde(
        model.generate(**model_input, max_new_tokens=256, pad_token_id=2)[0],
        skip_special_tokens=True,
    )
print(generated_code)

# Merge modeL
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)
merged_model = PeftModel.from_pretrained(base_model, new_model)
merged_model = merged_model.merge_and_unload()

merged_model.save_pretrained("merged_model", safe_serialization=True)
tokenizer.save_pretrained("merged_model")
