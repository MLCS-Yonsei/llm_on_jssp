# ────────────────────────────────────────────────────────────────
# 1) 경고 메시지 억제
from transformers import logging
logging.set_verbosity_error()

# 2) transformers 내부 병렬화 검사 우회
import transformers.modeling_utils as _mutils
_mutils.ALL_PARALLEL_STYLES = []

# 3) 모든 PreTrainedModel.post_init 을 빈 함수로 덮어쓰기
from transformers.modeling_utils import PreTrainedModel
PreTrainedModel.post_init = lambda self: None

# 4) T5LayerNorm Apex 오류 회피용 패치
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

def load_llm_model(local_path):
    """LoRA 어댑터 모델과 일반 LLM을 모두 안전하게 로드합니다."""
    # 단일 GPU 또는 CPU만 사용하도록 설정
    device_map = {"": 0} if torch.cuda.is_available() else None

    # adapter_config.json 이 있으면 PEFT 방식
    if os.path.exists(os.path.join(local_path, "adapter_config.json")):
        from peft import PeftModel, PeftConfig
        adapter_dir = local_path
        config = PeftConfig.from_pretrained(adapter_dir)

        # 1) base model을 single-device 로드
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            device_map=None,
            trust_remote_code=True,
        )
        base_model.config.model_parallel = False

        # 2) PEFT 어댑터 적용 (device_map 지정 가능)
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
        # 일반 base model 로드
        tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            local_path,
            device_map=device_map,
            trust_remote_code=True,
        )
        model.config.model_parallel = False

    # pipeline 생성 (device 인자는 제거)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=False,
    )
    return pipe

def load_seq2seq_model(model_name):
    """Seq2Seq 모델을 빠르게 로드하고 단일 디바이스로 이동합니다."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # GPU가 있으면 GPU로, 없으면 CPU로
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

# ====== 평가 기준 함수들 ======
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

# ====== 벤치마크 실행 함수 ======
def benchmark(models, test_samples):
    result_rows = []
    for mname, model_fn in models.items():
        print(f"\n🔍 Evaluating model: {mname}")
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

# ====== 메인 실행 ======
if __name__ == "__main__":
    finetuned_llm = load_llm_model(
        "./medical/jssp_llm/llm1_jssp_mistral7b_lora_final"
    )
    mistral_original = load_llm_model("mistralai/Mistral-7B-Instruct-v0.2")
    seq2seq = load_seq2seq_model("t5-base")

    # 테스트 샘플을 5개로 제한하여 빠른 실행
    with open(
        "./medical/jssp_llm/train_llm/test_data_llm1_100.jsonl"
    ) as f:
        test_samples = [json.loads(line) for line in f][:50]

    benchmark(
        {
            "finetuned_llm": lambda prompt: finetuned_llm(prompt),
            "mistral_original": lambda prompt: mistral_original(prompt),
            "seq2seq": lambda prompt: seq2seq(prompt),
        },
        test_samples,
    )
