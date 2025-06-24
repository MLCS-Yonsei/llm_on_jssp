from transformers import logging
logging.set_verbosity_error()

import transformers.modeling_utils as _mutils
_mutils.ALL_PARALLEL_STYLES = []

from transformers.modeling_utils import PreTrainedModel
PreTrainedModel.post_init = lambda self: None

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, default_data_collator
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, Dataset, DatasetDict
import random
import json
import wandb


path = #"File path of 'llm_on_jssp'/" 
wandb.init(project="jssp-llm", name="llm2_final")


def load_jsonl(filename, split_ratio=0.1, seed=42):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
    random.seed(seed)
    random.shuffle(data)
    split = int(len(data) * (1 - split_ratio))
    train_data = data[:split]
    val_data = data[split:]
    return Dataset.from_list(train_data), Dataset.from_list(val_data)

train_dataset, val_dataset = load_jsonl(path + 'train_llm/dataset_llm2_1k.jsonl', split_ratio=0.1)



# 2. Model and tokenizer
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

# 3. LoRA 
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 4. Load model and prepare (4bit)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    load_in_4bit=True,
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)


def preprocess_batch(batch):
    text = batch['input'] + "\n" + batch['output']
    tokens = tokenizer(text, truncation=True, padding='max_length', max_length=512)
    tokens['labels'] = tokens['input_ids'].copy()
    return tokens

train_dataset = train_dataset.map(preprocess_batch)
val_dataset = val_dataset.map(preprocess_batch)



data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)


training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=200,
    eval_steps=100,                
    eval_strategy="steps",          
    save_strategy="steps",         
    load_best_model_at_end=True,   
    metric_for_best_model="eval_loss", 
    output_dir=path + "llm2_mistral7b-lora-struct2text",
    save_total_limit=1,
    report_to="wandb",
    remove_unused_columns=True,
    ddp_find_unused_parameters=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
)


trainer.train()

model.save_pretrained(path + "llm2_mistral7b-lora-struct2text")
tokenizer.save_pretrained(path + "llm2_mistral7b-lora-struct2text")

