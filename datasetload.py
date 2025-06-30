from datasets import load_dataset

dataset = load_dataset("csv", data_files="faq_qa_only.csv")

def format_example(example):
    prompt = f"[INST] {example['question']} [/INST] {example['answer']}"
    return {"text": prompt}

dataset = dataset.map(format_example)

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2-1.5B-Chat"  # Or another Qwen2 model

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype="auto",
    load_in_4bit=True  # For QLoRA/LoRA-style quantization
)

from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

from transformers import TrainingArguments

training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    output_dir="./qwen2-faq-finetuned",
    logging_steps=10,
    save_steps=50
)

from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
    args=training_args
)
trainer.train()

trainer.save_model("./qwen2-faq-finetuned")
