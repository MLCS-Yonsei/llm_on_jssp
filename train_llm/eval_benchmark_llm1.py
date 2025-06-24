# ────────────────────────────────────────────────────────────────
# Avoiding Errors 
from transformers import logging
logging.set_verbosity_error()

import transformers.modeling_utils as _mutils
_mutils.ALL_PARALLEL_STYLES = []

from transformers.modeling_utils import PreTrainedModel
PreTrainedModel.post_init = lambda self: None

import importlib
import torch
try:
    t5_mod = importlib.import_module("transformers.models.t5.modeling_t5")
    t5_mod.T5LayerNorm = torch.nn.LayerNorm
except ImportError:
    pass
# ────────────────────────────────────────────────────────────────

import os
import json
import time
import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    pipeline,
)
import config

path = config.to_project_path

def load_llm_model(local_path):
    device_map = {"": 0} if torch.cuda.is_available() else None

    if os.path.exists(os.path.join(local_path, "adapter_config.json")):
        from peft import PeftModel, PeftConfig
        adapter_dir = local_path
        config = PeftConfig.from_pretrained(adapter_dir)

        # 1) load base model as single-device
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            device_map=None,
            trust_remote_code=True,
        )
        base_model.config.model_parallel = False

        # 2) apply PEFT adapter
        model = PeftModel.from_pretrained(
            base_model,
            adapter_dir,
            device_map=device_map,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model_name_or_path,
            use_fast=True,
            trust_remote_code=True,
        )
    else:
        # load original base model
        tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            local_path,
            device_map=device_map,
            trust_remote_code=True,
        )
        model.config.model_parallel = False

    # generate pipeline 
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=False,
    )
    return pipe

def load_seq2seq_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if device.type == "cuda" else -1,
        max_new_tokens=256,
    )
    return pipe

# ====== Performance Scores ======
def exact_match(pred, gt):
    return pred == gt

def matrix_match(pred, gt):
    try:
        return pred["matrix"] == gt["matrix"]
    except:
        return False

def label_match(pred, gt):
    try:
        return pred["label"] == gt["label"]
    except:
        return False

def parse_json_output(output):
    try:
        start = output.find("{")
        end = output.rfind("}")
        if start != -1 and end != -1:
            return json.loads(output[start : end + 1])
    except:
        return None
    return None


def benchmark(models, test_samples):
    result_rows = []
    for mname, model_fn in models.items():
        print(f"\n Evaluating model: {mname}")
        EM = MatrixOnly = LabelOnly = Fail = 0
        Times = []

        for sample in tqdm(test_samples, desc=mname):
            prompt = f"{sample['instruction']}\n{sample['input']}\nOutput:"
            gt = json.loads(sample["output"])
            t0 = time.time()
            try:
                output = model_fn(prompt)[0]["generated_text"]
            except Exception:
                output = ""
            t1 = time.time()

            Times.append(t1 - t0)
            pred = parse_json_output(output) if output else None
            if pred is None:
                Fail += 1
                continue
            if exact_match(pred, gt):
                EM += 1
            if matrix_match(pred, gt):
                MatrixOnly += 1
            if label_match(pred, gt):
                LabelOnly += 1

        total = len(test_samples)
        result_rows.append(
            {
                "Model": mname,
                "EM Acc": f"{EM/total:.2%}",
                "Matrix Only": f"{MatrixOnly/total:.2%}",
                "Label Only": f"{LabelOnly/total:.2%}",
                "Format Fail": Fail,
                "Avg Inference Time (s)": round(sum(Times) / total, 3),
            }
        )

    df = pd.DataFrame(result_rows)
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    finetuned_llm = load_llm_model(
        path + "/llm1_jssp_mistral7b_lora_final"
    )
    mistral_original = load_llm_model("mistralai/Mistral-7B-Instruct-v0.2")
    seq2seq = load_seq2seq_model("t5-base")

    
    with open(
        path + "/train_llm/test_data_llm1_100.jsonl"
    ) as f:
        test_samples = [json.loads(line) for line in f][:50]    # evaluate only 50 samples

    benchmark(
        {
            "finetuned_llm": lambda prompt: finetuned_llm(prompt),
            "mistral_original": lambda prompt: mistral_original(prompt),
            "seq2seq": lambda prompt: seq2seq(prompt),
        },
        test_samples,
    )
