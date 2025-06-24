import os

# only thing you need to modify #

to_project_path          = "path_to_project_folder"     
which_prompt_do_you_want = "file_name_of_problem_prompt.json"             # no need to include path. only file name       
which_env_do_you_want    = "file_name_of_env_img.png"                     # no need to include path. only file name
HF_TOKEN                 = "your Hugging Face token with the access permission to mistralai/Mistral-7B-Instruct-v0.2"                  


#########################################################################################################################
#########################################################################################################################
# path to input
#PROJECT_ROOT = os.path.abspath(to_project_path)
path_to_input_prompt = os.path.join(to_project_path, "input", which_prompt_do_you_want)
path_to_input_image = os.path.join(to_project_path, "input", which_env_do_you_want)

# LLM1
LLM1_BASE_MODEL_ID   = "mistralai/Mistral-7B-Instruct-v0.2"
LLM1_LORA_MODEL_PATH = os.path.join(to_project_path, "llm1_jssp_mistral7b_lora_final")

# LLM2
LLM2_BASE_MODEL_ID   = "mistralai/Mistral-7B-Instruct-v0.2"
LLM2_LORA_MODEL_PATH = os.path.join(to_project_path, "llm2_mistral7b-lora-struct2text")
