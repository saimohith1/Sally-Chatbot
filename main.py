import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# 1. Model and Tokenizer Setup
model_name = "Qwen/Qwen2-1.5B-Chat"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    # attn_implementation="flash_attention_2",  # Uncomment ONLY if flash-attn is installed
)

# 2. Prepare Dataset
dataset = load_dataset("csv", data_files="faq_qa_only.csv")

def format_example(example):
    return {"text": f"[INST] {example['question']} [/INST] {example['answer']}"}

dataset = dataset.map(format_example)

# 3. LoRA Config
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"],
)

# 4. SFTConfig (put ALL training/data args here)
sft_config = SFTConfig(
    output_dir="./qwen2-faq-finetuned",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    learning_rate=2e-4,
    bf16=True,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    logging_steps=10,
    save_steps=50,
    report_to="none",
    max_steps=300,  # Optional: Early stopping
    ddp_find_unused_parameters=False,
    max_seq_length=1024,         # <-- Put here!
    dataset_text_field="text",   # <-- Put here!
)

# 5. Trainer Setup (NO max_seq_length or dataset_text_field here)
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    peft_config=peft_config,
    processing_class=tokenizer,   # Correct way to pass tokenizer
    args=sft_config,              # Pass SFTConfig here
)

trainer.train()

# 6. Save Model
trainer.save_model("./qwen2-faq-finetuned")

# 7. Test Inference
from transformers import pipeline

chatbot = pipeline("text-generation", model="./qwen2-faq-finetuned", tokenizer=tokenizer)
response = chatbot("[INST] How do I track my order? [/INST]", max_new_tokens=128)
print(response[0]['generated_text'])
