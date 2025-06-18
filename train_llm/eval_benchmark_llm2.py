# ────────────────────────────────────────────────────────────────
# 1) 경고 메시지 억제
from transformers import logging
logging.set_verbosity_error()

# 2) transformers 내부 병렬화 검사 우회
import transformers.modeling_utils as _mutils
_mutils.ALL_PARALLEL_STYLES = []

# 3) 모든 PreTrainedModel.post_init 을 no-op 으로 덮어쓰기
from transformers.modeling_utils import PreTrainedModel
PreTrainedModel.post_init = lambda self: None

# 4) T5LayerNorm Apex 오류 회피용 패치
import importlib, torch
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
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    pipeline,
)

def load_llm_model(local_path):
    device_map = {"": 0} if torch.cuda.is_available() else None

    if os.path.exists(os.path.join(local_path, "adapter_config.json")):
        from peft import PeftModel, PeftConfig
        config = PeftConfig.from_pretrained(local_path)

        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            device_map=None,
            trust_remote_code=True,
        )
        base_model.config.model_parallel = False

        model = PeftModel.from_pretrained(
            base_model,
            local_path,
            device_map=device_map,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model_name_or_path,
            use_fast=True,
            trust_remote_code=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            local_path,
            device_map=device_map,
            trust_remote_code=True,
        )
        model.config.model_parallel = False

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=False,
    )

def load_seq2seq_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if device.type == "cuda" else -1,
        max_new_tokens=256,
    )



def convert_solution_for_llm(solution_json):
    sol = solution_json['solution']
    makespan = sol['makespan']
    filtered_schedule = []
    for job in sol['schedule']:
        filtered_job = [op for op in job if not (isinstance(op[0], int) and str(op[0]).startswith('9'))]
        filtered_schedule.append(filtered_job)
    solution_dict = {"solution": {"schedule": filtered_schedule, "makespan": makespan}}
    prompt = (
        'Convert the following job schedule into natural language:\n'
        f'{json.dumps(solution_dict)}\nAnswer:'
    )
    return prompt

def benchmark(models, test_samples):
    result_rows = []
    for name, fn in models.items():
        print(f"\n🔍 Evaluating model: {name}")
        EM = Fail = 0
        Times = []
        fail_examples = []

        for sample in tqdm(test_samples, desc=name):
            # ❗️ 여기만 바뀌었습니다: 
            # sample["input"] 은 JSON 문자열이므로 먼저 파싱한 뒤 변환 함수에 넘깁니다.
            sol_json = json.loads(sample["input"])
            prompt = convert_solution_for_llm(sol_json)

            t0 = time.time()
            out_list = fn(prompt)
            t1 = time.time()
            Times.append(t1 - t0)

            full = out_list[0].get("generated_text", "")

            # 모델이 붙여준 "Answer:" 이후 텍스트만 떼어냅니다.
            if "\nAnswer:" in full:
                pred_text = full.split("\nAnswer:", 1)[1].strip()
            else:
                parts = full.split("Answer:", 1)
                pred_text = parts[1].strip() if len(parts) > 1 else full.strip()

            gt_text = sample["output"].strip()

            if pred_text == gt_text:
                EM += 1
            else:
                Fail += 1
                if name == "llm2_struct2text" and len(fail_examples) < 2:
                    fail_examples.append((prompt, pred_text, gt_text))

        total = len(test_samples)
        result_rows.append({
            "Model": name,
            "Exact Text Match": f"{EM/total:.2%}",
            "Format Fail": Fail,
            "Avg Inference Time (s)": round(sum(Times)/total, 3),
        })

        if name == "llm2_struct2text":
            print("\n=== llm2_struct2text 실패 예시 ===")
            for prompt, pred, gt in fail_examples:
                print(">>> Prompt:")
                print(prompt)
                print(">>> Predicted:")
                print(pred)
                print(">>> Ground Truth:")
                print(gt)
                print("-"*40)

    df = pd.DataFrame(result_rows)
    print(df.to_string(index=False))
    return df

if __name__ == "__main__":
    # 모델 로드
    llm2 = load_llm_model("./medical/jssp_llm/llm2_mistral7b-lora-struct2text")
    mistral_original = load_llm_model("mistralai/Mistral-7B-Instruct-v0.2")
    seq2seq = load_seq2seq_model("t5-base")

    # 테스트 샘플 로드 (예: 50개)
    with open("./medical/jssp_llm/train_llm/test_data_llm2_100.jsonl") as f:
        samples = [json.loads(l) for l in f][:50]

    # 벤치마크 실행
    benchmark({
        "llm2_struct2text": lambda p: llm2(p),
        "mistral_original":    lambda p: mistral_original(p),
        "seq2seq":             lambda p: seq2seq(p),
    }, samples)
