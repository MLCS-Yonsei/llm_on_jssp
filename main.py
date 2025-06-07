# main.py
from llm_module import llm_parse_input, llm_generate_final_output
from optimizer import solve_schedule
import json

if __name__ == '__main__':
    prompt = "./input/prompt.json"              # 데이터 만들어야 함
    env_image = "./input/environment.jpg"      # 알아먹기 쉬운 데이터로 다시 만들기

    # Step 1: LLM processes prompt + image to structured JSON
    problem_json = llm_parse_input(prompt, env_image)

    # Step 2: Optimizer solves the structured problem and returns solution JSON
    solution_json = solve_schedule(problem_json)

    # Step 3: LLM evaluates and generates final prompt-like output
    final_output = llm_generate_final_output(solution_json)

    print("\n\n[Final LLM Output]\n")
    print(final_output)

