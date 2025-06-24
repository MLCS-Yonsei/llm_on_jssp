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
        
        # Select the solution with the shortest makespan
        best_solver, best_solution = min(results, key=lambda x: x[1]["makespan"])
        return {
            "selected_solver": best_solver,
            "solution": best_solution
        }

    else:
        raise ValueError("Invalid label. Use 0 for fastest, 1 for best makespan.")

