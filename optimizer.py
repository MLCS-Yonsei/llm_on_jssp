### 솔버 부서 ###
# main 함수(solve_schedule)이 새끼 솔버들을 일 시키고 전달받은 evaluation 기준에 따라서 최종 답을 반환 #
# 솔버 팀장의 인풋은 행렬과 평가 기준 #
# 솔버 팀장은 각 솔버에게 행렬을 전달하고, 각 솔버는 행렬을 처리하여 결과를 반환 #


from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import time
import random
from room_of_heuristic import solver_tabu, solver_ga, solver_sa, solver_aco, solver_grasp, solver_ortools


def solve_schedule(problem_json):
    problem = problem_json["matrix"]
    label = problem_json["label"]  # 0: fastest, 1: best makespan

    solvers = [
        solver_tabu,
        solver_ga,
        solver_sa,
        solver_aco,
        solver_grasp,
        solver_ortools,
    ]

    if label == 'fastest':
        # Return the solution from the fastest solver
        with ProcessPoolExecutor(max_workers=6) as executor:
            futures = {executor.submit(solver, problem): solver.__name__ for solver in solvers}
            done, pending = wait(futures, return_when=FIRST_COMPLETED)
            first_completed = done.pop()
            solver_name, solution = first_completed.result()
            for future in pending:
                future.cancel()

        return {
            "selected_solver": solver_name,
            "solution": solution
        }

    elif label == 'best_makespan':
        # Return the solution with the best makespan
        results = []
        with ProcessPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(solver, problem) for solver in solvers]
            for future in futures:
                solver_name, solution = future.result()
                results.append((solver_name, solution))
        print(results) ### 모든 답 출력력
        # Select the solution with the shortest makespan
        best_solver, best_solution = min(results, key=lambda x: x[1]["makespan"])
        return {
            "selected_solver": best_solver,
            "solution": best_solution
        }

    else:
        raise ValueError("Invalid label. Use 0 for fastest, 1 for best makespan.")



'''솔버 부서 잘 돌아가는지 확인'''
if __name__ == '__main__':
    # Example problem JSON
    '''
    problem_json = {
        # 4 jobs, 6 machines (each row for job, (machine_index, duration))
        "matrix": [
            [(0, 2), (1, 3), (2, 1)],
            [(3, 1), (4, 4), (5, 2)],
            [(1, 5), (5, 4), (0, 2)],
            [(1, 1), (2, 4), (3, 2)],
        ],
        "label": 'best makespan'  # 0: fastest, 1: best makespan
    }
    '''
    #problem_json = {'matrix': [[[2, 2], [901, 26], [0, 3], [902, 31], [3, 4]], [[1, 4], [911, 25], [2, 2]]], 
    #                'label': 'best_makespan'}
    problem_json = {'matrix': 
                    [[[900, 21], [2, 10], [901, 20], [0, 5], [902, 25], [3, 20], [903, 16], [5, 2], [904, 22], [1, 12]], 
                    [[910, 34], [1, 15], [911, 25], [2, 7], [912, 23], [6, 20]], 
                    [[920, 19], [6, 10], [921, 23], [2, 15], [922, 21], [5, 20]], 
                    [[930, 11], [0, 20], [931, 33], [4, 5], [932, 10], [1, 2], [933, 18], [3, 10]]], 
                    'label': 
                    'best_makespan'}

    solution = solve_schedule(problem_json)
    print("\n\n[Solution]\n")
    print(solution)
