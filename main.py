# main.py
from llm_module import llm_generate_final_output
from optimizer import solve_schedule
import json
from prompt_entrance import make_input_matrix

### team_project 파일 경로 ###
path = 'C:/Users/djm06/Desktop/MLP/team_project/'
###                      ### 

def convert_solution_for_llm(solution_json):
    sol = solution_json['solution']
    makespan = sol['makespan']
    filtered_schedule = []
    for job in sol['schedule']:
        filtered_job = [op for op in job if not (isinstance(op[0], int) and str(op[0]).startswith('9'))] ## machine9일 경우가 고려안되어 있는 코드드
        filtered_schedule.append(filtered_job)
    solution_dict = {"solution": {"schedule": filtered_schedule, "makespan": makespan}}
    prompt = f'Convert the following job schedule into natural language:\n{json.dumps(solution_dict)}\nAnswer:'
    return prompt

def extract_answer(llm_output):
    """
    LLM 출력 문자열에서 'Answer:' 이후만 추출하여 반환
    """
    marker = "Answer:"
    idx = llm_output.find(marker)
    if idx != -1:
        return llm_output[idx + len(marker):].strip()
    else:
        # marker가 없는 경우 전체 반환 (예외 방지)
        return llm_output.strip()



if __name__ == '__main__':
    
    prompt = path + "input/problem_prompt.json"              # 데이터 만들어야 함
    env_image = path + "input/env_img.png"      # 알아먹기 쉬운 데이터로 다시 만들기

    # Step 1: LLM processes prompt + image to structured JSON
    # 출력 예시
    # {'matrix': [[[2, 2], [901, 26], [0, 3], [902, 31], [3, 4]], [[1, 4], [911, 25], [2, 2]]], 'label': 'best_makespan'}
    problem_json = make_input_matrix(prompt, env_image, al_path=path)

    # Step 2: Optimizer solves the structured problem and returns solution JSON
    # 출력 예시
    # {'selected_solver': 'tabu', 
    # 'solution': {'schedule': [[(2, 0, 2), (901, 2, 28), (0, 28, 31), (902, 31, 62), (3, 62, 66)], 
    #                           [(1, 0, 4), (911, 4, 29), (2, 29, 31)]], 'makespan': 66}}
    solution_json = solve_schedule(problem_json)

    # Step 3: LLM evaluates and generates final prompt-like output
    # 입력 예시
    #prompt = 'Convert the following job schedule into natural language:\n
    # {"solution": {"schedule": [[(2, 0, 2), (0, 28, 31), (3, 62, 66)], [(1, 0, 4), (2, 29, 31)]], 
    # "makespan": 66}}\nAnswer:' ->> 이게 llm_prompt
    # 출력 예시
    #Convert the following job schedule into natural language:
    #{"solution": {"schedule": [[(2, 0, 2), (0, 28, 31), (3, 62, 66)], [(1, 0, 4), (2, 29, 31)]], "makespan": 66}}
    #Answer: Job 1 is processed on machine 2 from 0 to 2, machine 0 from 28 to 31, machine 3 from 62 to 66. Job 2 is processed on machine 1 from 0 to 4, machine 2 from 29 to 31. The total makespan is 66.
    llm_prompt = convert_solution_for_llm(solution_json)
    final_output = llm_generate_final_output(llm_prompt, al_path=path)

    # 최종 파싱
    # 출력 예시
    # Job 1 is processed on machine 2 from 0 to 2, machine 0 from 28 to 31, machine 3 from 62 to 66. Job 2 is processed on machine 1 from 0 to 4, machine 2 from 29 to 31. The total makespan is 66.
    final_answer = extract_answer(final_output)

    # 결과 출력
    print("\n\n[Final LLM Output]\n")
    print(final_answer)

