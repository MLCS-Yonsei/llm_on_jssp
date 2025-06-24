import os
import json
import torch
from peft import PeftModel
from huggingface_hub import login, HfFolder
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import config



# Get shared Hugging Face token
token = config.HF_TOKEN

# Login (required for private models)
login(token=token)
HfFolder.save_token(token)

def _clean_peft_config(peft_path: str):
    
    config_file = os.path.join(peft_path, "adapter_config.json")
    if not os.path.isfile(config_file):
        return
    with open(config_file, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    
    bad_keys = (
        "corda_config",
        "loftq_config",
        "eva_config",
        "runtime_config",
        "exclude_modules",
        "rank_pattern",
        "alpha_pattern",
        "modules_to_save",
        "megatron_config",
        "megatron_core",
        "revision",
        "lora_bias",
        "trainable_token_indices"
    )
    for key in bad_keys:
        cfg.pop(key, None)
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def llm_parse_input(full_prompt: str):
    """
    Use LLM1 (LoRA adapter + base model) to parse input and extract JSON from the generated text.
    """
    base_model_id = config.LLM1_BASE_MODEL_ID
    lora_model_path = config.LLM1_LORA_MODEL_PATH
    _clean_peft_config(lora_model_path)

    # Ensure offload directory exists
    offload_folder = os.path.join(config.to_project_path, 'offload')
    os.makedirs(offload_folder, exist_ok=True)

    # Tokenizer & Base model with offloading
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_auth_token=token)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        #device_map="auto",
        offload_folder=offload_folder,
        offload_state_dict=True,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        use_auth_token=token
    )

    # Apply LoRA adapter
    model = PeftModel.from_pretrained(base_model, lora_model_path, use_auth_token=token)

    # Pipeline for inference
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

    # Run inference
    result = pipe(full_prompt)[0]['generated_text']

    '''For Debugging
    print("\n=== input prompt ===")
    print(full_prompt)
    print("\n=== LLM Prediction ===")
    print(result)
    '''

    try:
        start_idx = result.find('{')
        end_idx = result.rfind('}')
        if start_idx != -1 and end_idx != -1:
            json_str = result[start_idx:end_idx+1]
            parsed = json.loads(json_str)
            print("\n=== Parsed Output ===")
            print(parsed)
        else:
            print("\n[Warning] Can't find JSON {} block.")
            parsed = None
    except Exception as e:
        print("\n[Warning] output parsing failed!", e)
        parsed = None

    return parsed


def llm_generate_final_output(prompt: str):
    """
    Use LLM2 (LoRA adapter + base model) to generate final structured text from prompt.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_id = config.LLM2_BASE_MODEL_ID
    lora_path = config.LLM2_LORA_MODEL_PATH
    _clean_peft_config(lora_path)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_id, use_auth_token=config.HF_TOKEN)

    # Prepare load kwargs
    load_kwargs = {'device_map': {'': device}, 'use_auth_token': config.HF_TOKEN}
    # If GPU available and bitsandbytes installed, use 4-bit quant
    if device=='cuda':
        try:
            bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
            load_kwargs['quantization_config'] = bnb_config
        except Exception:
            print('bitsandbytes not available, loading full precision on GPU')
            load_kwargs['torch_dtype'] = torch.float16
    else:
        load_kwargs['torch_dtype'] = torch.float32

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(base_id, **load_kwargs)

    # Apply LoRA adapter
    model = PeftModel.from_pretrained(base_model, lora_path, **{'device_map':{'':device}, 'use_auth_token':config.HF_TOKEN})

    # Generate output
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    outputs = model.generate(**inputs, max_new_tokens=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
