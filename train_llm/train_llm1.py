from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig, pipeline
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import wandb

path = #"File path of 'llm_on_jssp'/" 

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
dataset_path = path + "train_llm/dataset_llm1_5k.jsonl" 
wandb.init(project="jssp-llm", name="jssp_llm_ft")


dataset = load_dataset("json", data_files=dataset_path, split="train")

def formatting_func(example):
    return {
        "input": f"{example['instruction']}\n{example['input']}\nOutput:",
        "output": example['output']
    }

dataset = dataset.map(formatting_func)
ds = dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = ds["train"]
eval_dataset = ds["test"]

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="float16")
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

def tokenize_function(examples):
    # input과 output을 붙여 full_texts로
    full_texts = [
        f"{prompt} {target}" for prompt, target in zip(examples["input"], examples["output"])
    ]
    tokenized = tokenizer(
        full_texts,
        truncation=True,
        max_length=512,
        padding="max_length"
    )

    prompt_lens = [
        len(tokenizer(prompt, truncation=True, max_length=512)["input_ids"])
        for prompt in examples["input"]
    ]

    new_labels = []
    for input_ids, p_len in zip(tokenized["input_ids"], prompt_lens):
        input_ids = input_ids[:512]
        lbl = [-100]*p_len + input_ids[p_len:]
        lbl = lbl[:512]
        new_labels.append(lbl)

    tokenized["labels"] = new_labels
    return tokenized

train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    learning_rate=2e-4,
    eval_strategy="steps",
    eval_steps=20,
    save_strategy="steps",
    save_steps=20,
    report_to="wandb",
    logging_steps=10,
    output_dir=path+"/llm1_jssp_mistral7b_lora",
    fp16=True,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

trainer.train()
eval_results = trainer.evaluate()
print("=== Evaluation Results on Test Data ===")
print(eval_results)
trainer.save_model(path+"llm1_jssp_mistral7b_lora_final")
print("Fine-tuning complete and model saved!")



