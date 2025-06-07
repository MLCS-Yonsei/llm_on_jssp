##### 학습 완료된 llm들이 일하는 방 #####

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json

# LLM1 불러오기 (예시: ./llm1에 모델 저장)
tokenizer1 = AutoTokenizer.from_pretrained("./llm1")                    #### llm1 만들고 나서 이름 바꾸기
model1 = AutoModelForCausalLM.from_pretrained("./llm1")                 #### llm1 만들고 나서 이름 바꾸기 
llm1 = pipeline("text-generation", model=model1, tokenizer=tokenizer1)

# LLM2 불러오기 (예시: ./llm2에 모델 저장)
tokenizer2 = AutoTokenizer.from_pretrained("./llm2")                    #### llm2 만들고 나서 이름 바꾸기
model2 = AutoModelForCausalLM.from_pretrained("./llm2")                 #### llm2 만들고 나서 이름 바꾸기
llm2 = pipeline("text-generation", model=model2, tokenizer=tokenizer2)

# 사용 예시
def llm_parse_input(prompt, env_image):
    input_text = f"{prompt}\n[IMAGE]{env_image}"  # 이미지 경로/설명 등
    output = llm1(input_text, max_length=512)[0]["generated_text"]      #### 나중에 출력 형식 맞는지 확인하기
    return json.loads(output)

def llm_generate_final_output(solution_json):
    input_text = f"Schedule: {solution_json}"
    output = llm2(input_text, max_length=256)[0]["generated_text"]
    return output
