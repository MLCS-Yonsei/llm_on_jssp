#############################################################################
# Pre-training gpt2 with 120k full dataset of Starjob                       #
# show answers to the same test prompts before training, and after training #
#############################################################################
import os
import json
import re
import torch
from datasets import Dataset, load_dataset
from transformers import (
    GPT2Tokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import get_peft_model, PrefixTuningConfig, TaskType

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
path = "as_you_would_like_it_to_be"

# === Load and format Starjob data ===
def format_example(example):
    #prompt_text = example.get("prompt_jobs_first", "")
    #json_prompt = json.dumps({"prompt": prompt_text}, separators=(",", ":"))
    return {
        #"text": json_prompt + "\nAnswer:",
        "text": example["prompt_jobs_first"] + "\nAnswer:",
        "labels": example["output"]
    }

with open(path + "/jssp_llm_format_120k.json", "r") as f:
    raw_data = json.load(f)

formatted_data = [format_example(d) for d in raw_data]
dataset = Dataset.from_list(formatted_data)
dataset = dataset.train_test_split(test_size=0.2)

# === Tokenization ===
tokenizer = GPT2Tokenizer.from_pretrained("gpt2", add_prefix_space=True)
tokenizer.pad_token = tokenizer.eos_token

# Tokenization with numerical stability
def tokenize(batch):
    return tokenizer(batch["text"], text_target=batch["labels"], padding="longest", truncation=True, max_length=1024)

tokenized = dataset.map(tokenize, batched=True)

# === Load Model with Prefix Tuning and Extended Position Embeddings ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

config = GPT2Config.from_pretrained("gpt2", n_positions=2048)
model = GPT2LMHeadModel.from_pretrained("gpt2", config=config, ignore_mismatched_sizes=True)
model.resize_token_embeddings(len(tokenizer))
model.to(device)


''' Initial efforts to try pre-fix tuning
base_model = GPT2LMHeadModel.from_pretrained("gpt2", config=config, ignore_mismatched_sizes=True)
base_model.resize_token_embeddings(len(tokenizer))

peft_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=20)
model = get_peft_model(base_model, peft_config)
model.to(device)
'''


# === Training Configuration with Early Stopping and Matching Save/Eval Strategy ===
training_args = TrainingArguments(
    output_dir=path + "/starjob_gpt2_full",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=10,
    weight_decay=0.01,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    save_total_limit=1,
    logging_steps=10,
    fp16=torch.cuda.is_available(),
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# === Train ===
trainer.train()

# === Evaluate ===
def evaluate(model, tokenizer, dataset_raw):
    model.eval()
    prompts = [ex["text"] for ex in dataset_raw]
    results = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=300, max_length=1024)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append(decoded)
    return results

def extract_makespan(text):
    match = re.search(r"Makespan:\s+(\d+(\.\d+)?)", text)
    return float(match.group(1)) if match else None

# Evaluate before and after training
pre_model = GPT2LMHeadModel.from_pretrained("gpt2", config=config, ignore_mismatched_sizes=True).to(device)
pre_results = evaluate(pre_model, tokenizer, formatted_data)
post_results = evaluate(model, tokenizer, formatted_data)
true_makespans = [extract_makespan(d["labels"]) for d in formatted_data]
pre_makespans = [extract_makespan(r) for r in pre_results]
post_makespans = [extract_makespan(r) for r in post_results]

pre_diff = [abs(p - t) if p is not None else None for p, t in zip(pre_makespans, true_makespans)]
post_diff = [abs(p - t) if p is not None else None for p, t in zip(post_makespans, true_makespans)]

valid_pre_diff = [x for x in pre_diff if x is not None]
valid_post_diff = [x for x in post_diff if x is not None]

'''
print("\n[Missing Makespan in pre_results]")
for i, text in enumerate(pre_results):
    makespan = extract_makespan(text)
    if makespan is None:
        print(f"\n🟥 Output {i} (No Makespan Found):")
        print(text[:10000])  # 너무 길면 500자까지만 출력
'''

# === Print sample outputs ===
print("\n[Sample Outputs]")
for i in range(3):
    print(f"\n--- Sample {i+1} ---")
    print("[Pre-train Result]:")
    print(pre_results[i][:10000])
    print("[Post-train Result]:")
    print(post_results[i][:10000])



print("\n[Performance Summary]")
print(f"Pre-train avg. makespan error: {sum(valid_pre_diff) / len(valid_pre_diff):.2f}")
print(f"Post-train avg. makespan error: {sum(valid_post_diff) / len(valid_post_diff):.2f}")

'''
print("\n[Performance Summary]")
print(f"Pre-train avg. makespan error: {sum(pre_diff) / len([x for x in pre_diff if x is not None]):.2f}")
print(f"Post-train avg. makespan error: {sum(post_diff) / len([x for x in post_diff if x is not None]):.2f}")
'''
