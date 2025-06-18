

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json
import torch

#path = 'C:/Users/djm06/Desktop/MLP/team_project/'
#path = './medical/jssp_llm/'

def llm_parse_input(full_prompt, al_path):
    path = al_path
    # LLM1 불러오기 (예시: ./llm1에 모델 저장)
    finetuned_model_path = path + "llm1_jssp_mistral7b_lora_final"
    tokenizer = AutoTokenizer.from_pretrained(finetuned_model_path)
    model = AutoModelForCausalLM.from_pretrained(finetuned_model_path, device_map="auto", trust_remote_code=True)

    # 파이프라인: 디코딩 컨트롤 세팅!
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=False,
        repetition_penalty=1.2,
        temperature=0.7,
        eos_token_id=tokenizer.eos_token_id,
    )

    #full_prompt =           ###인풋 불러오기기

    result = pipe(full_prompt)[0]['generated_text']

    print("\n=== 예시 입력 ===")
    print(full_prompt)
    print("\n=== LLM Prediction ===")
    print(result)

    # JSON 파싱
    
    try:
        # 마지막 } 기준으로 자름
        start_idx = result.find('{')
        end_idx = result.rfind('}')
        if start_idx != -1 and end_idx != -1:
            json_substr = result[start_idx:end_idx+1]
            parsed = json.loads(json_substr)
            print("\n=== Parsed Output ===")
            print(parsed)
        else:
            print("\n[Warning] JSON 중괄호 블록을 찾지 못함.")
    except Exception as e:
        print("\n[Warning] output 파싱 실패!", e)


    return parsed #json.loads(result) # 파싱된 json



def llm_generate_final_output(prompt, al_path):
    path = al_path
    # LLM2 불러오기 (예시: ./llm2에 모델 저장)
    model_path = path + "llm2_mistral7b-lora-struct2text"
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    #### 이거 수정 솔루션 들어올 때는 앞에 문장이 없음 ### 수정!!!
    #prompt = 'Convert the following job schedule into natural language:\n{"solution": {"schedule": [[(0, 0, 2), (1, 2, 5), (2, 5, 6)]], "makespan": 6}}\nAnswer:'

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=128)
    #print(tokenizer.decode(output[0], skip_special_tokens=True))

    return output
