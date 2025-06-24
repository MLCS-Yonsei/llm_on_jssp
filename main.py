from llm_module import llm_generate_final_output
from optimizer import solve_schedule
import json
from prompt_entrance import make_input_matrix
import os
import config


def convert_solution_for_llm(solution_json):
    sol = solution_json['solution']
    makespan = sol['makespan']
    filtered_schedule = []

    for job in sol['schedule']:
        filtered_job = [op for op in job if not (isinstance(op[0], int) and str(op[0]).startswith('9'))] 
        filtered_schedule.append(filtered_job)
    solution_dict = {"solution": {"schedule": filtered_schedule, "makespan": makespan}}
    prompt = f'Convert the following job schedule into natural language:\n{json.dumps(solution_dict)}\nAnswer:'
    return prompt

def extract_answer(llm_output):
    
    marker = "Answer:"
    idx = llm_output.find(marker)
    if idx != -1:
        return llm_output[idx + len(marker):].strip()
    else:
        return llm_output.strip()



if __name__ == '__main__':   

    path = config.to_project_path
    prompt = config.path_to_input_prompt
    env_image = config.path_to_input_image

    # Step 1: LLM processes prompt + image to structured JSON
    problem_json = make_input_matrix(prompt, env_image, al_path=path)

    # Step 2: Optimizer solves the structured problem and returns solution JSON
    solution_json = solve_schedule(problem_json)

    # Step 3: LLM evaluates and generates final prompt-like output
    llm_prompt = convert_solution_for_llm(solution_json)
    final_output = llm_generate_final_output(llm_prompt)

    # Step 4: Final parsing
    final_answer = extract_answer(final_output)

    # Print results
    print("\n\n[Final LLM Output]\n")
    print(final_answer)

