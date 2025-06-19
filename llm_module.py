from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json
import torch

path = #"File path of 'llm_on_jssp'/" 


def llm_parse_input(full_prompt, al_path):
    path = al_path
    # Load LLM1
    finetuned_model_path = path + "llm1_jssp_mistral7b_lora_final"
    tokenizer = AutoTokenizer.from_pretrained(finetuned_model_path)
    model = AutoModelForCausalLM.from_pretrained(finetuned_model_path, device_map="auto", trust_remote_code=True)

    
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

    
    result = pipe(full_prompt)[0]['generated_text']

    print("\n=== example sample ===")
    print(full_prompt)
    print("\n=== LLM Prediction ===")
    print(result)

    # JSON parsing
    try:
        start_idx = result.find('{')
        end_idx = result.rfind('}')
        if start_idx != -1 and end_idx != -1:
            json_substr = result[start_idx:end_idx+1]
            parsed = json.loads(json_substr)
            print("\n=== Parsed Output ===")
            print(parsed)
        else:
            print("\n[Warning] Can't find JSON {} block.")
    except Exception as e:
        print("\n[Warning] output parsing failed!", e)


    return parsed 



def llm_generate_final_output(prompt, al_path):
    path = al_path
    # Load LLM2
    model_path = path + "llm2_mistral7b-lora-struct2text"
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=128)
    #print(tokenizer.decode(output[0], skip_special_tokens=True))

    return output
