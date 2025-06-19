"""
< Room of Heuristic Methods >
Variable number of operations per job is allowed and machines machine duplication (only one job at a time) is disallowed.
A conflict-free schedule generator that can be used across solvers.
job_order: Specifies the order of jobs (original order if None)
machine_priority: Used to change the machine order within each job
randomize: Adds diversity, e.g., by applying random offsets
"""

import random
import time
from ortools.sat.python import cp_model

def _build_feasible_schedule(problem, job_order=None, machine_priority=None, randomize=False):
    
    n_jobs = len(problem)
    schedule = []
    machine_available = {}        # available time for each machine 
    job_last_end = [0] * n_jobs   # end time of previous operation 
    jobs = problem if job_order is None else [problem[i] for i in job_order]

    for jid, job in enumerate(jobs):
        j_idx = jid if job_order is None else job_order[jid]
        job_sched = []
        operations = job
        if machine_priority is not None:
            operations = [job[i] for i in machine_priority[j_idx]]
        for tid, (machine, duration) in enumerate(operations):
            start_time = max(machine_available.get(machine, 0), job_last_end[j_idx])
            
            if randomize:
                start_time += random.randint(0, 1)
            end_time = start_time + duration
            job_sched.append((machine, start_time, end_time))
            machine_available[machine] = end_time
            job_last_end[j_idx] = end_time
        schedule.append(job_sched)
    makespan = max(op[2] for job in schedule for op in job)
    return schedule, makespan



def solver_tabu(problem):
    time.sleep(random.uniform(0.1, 0.4))
    schedule, makespan = _build_feasible_schedule(problem)
    return ("tabu", {"schedule": schedule, "makespan": makespan})


def solver_ga(problem):
    time.sleep(random.uniform(0.2, 0.5))
    n = len(problem)
    job_order = list(range(n))
    random.shuffle(job_order)
    schedule, makespan = _build_feasible_schedule(problem, job_order=job_order, randomize=True)
    return ("ga", {"schedule": schedule, "makespan": makespan})


def solver_sa(problem):
    time.sleep(random.uniform(0.15, 0.45))
    schedule, makespan = _build_feasible_schedule(problem, randomize=True)
    return ("sa", {"schedule": schedule, "makespan": makespan})


def solver_aco(problem):
    time.sleep(random.uniform(0.15, 0.5))
    pheromone_bias = random.randint(0, 1)
    def custom_randomize():
        return random.randint(0, pheromone_bias)
    
    n_jobs = len(problem)
    schedule = []
    machine_available = {}
    job_last_end = [0] * n_jobs
    for job_id, job in enumerate(problem):
        job_sched = []
        for machine, duration in job:
            start_time = max(machine_available.get(machine, 0), job_last_end[job_id])
            start_time += random.randint(0, pheromone_bias)
            end_time = start_time + duration
            job_sched.append((machine, start_time, end_time))
            machine_available[machine] = end_time
            job_last_end[job_id] = end_time
        schedule.append(job_sched)
    makespan = max(op[2] for job in schedule for op in job)
    return ("aco", {"schedule": schedule, "makespan": makespan})

def solver_grasp(problem):
    time.sleep(random.uniform(0.1, 0.35))
    schedule, makespan = _build_feasible_schedule(problem, randomize=True)
    return ("grasp", {"schedule": schedule, "makespan": makespan})

def solver_ortools(problem):
    jobs_data = problem
    model = cp_model.CpModel()
    num_jobs = len(jobs_data)
    horizon = sum(task[1] for job in jobs_data for task in job)
    all_tasks = {}
    machine_to_intervals = dict()

    for job_id, job in enumerate(jobs_data):
        previous_end = None
        for task_id, (machine, duration) in enumerate(job):
            suffix = f'_{job_id}_{task_id}'
            start_var = model.NewIntVar(0, horizon, 'start' + suffix)
            end_var = model.NewIntVar(0, horizon, 'end' + suffix)
            interval = model.NewIntervalVar(start_var, duration, end_var, 'interval' + suffix)
            all_tasks[(job_id, task_id)] = (start_var, end_var, interval)
            if machine not in machine_to_intervals:
                machine_to_intervals[machine] = []
            machine_to_intervals[machine].append(interval)
            if previous_end is not None:
                model.Add(start_var >= previous_end)
            previous_end = end_var

    for intervals in machine_to_intervals.values():
        model.AddNoOverlap(intervals)

    obj_var = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(obj_var, [end for (_, end, _) in all_tasks.values()])
    model.Minimize(obj_var)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    schedule = []
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        for job_id, job in enumerate(jobs_data):
            job_sched = []
            for task_id in range(len(job)):
                start = int(solver.Value(all_tasks[(job_id, task_id)][0]))
                end = int(solver.Value(all_tasks[(job_id, task_id)][1]))
                machine = jobs_data[job_id][task_id][0]
                job_sched.append((machine, start, end))
            schedule.append(job_sched)
        makespan = int(solver.ObjectiveValue())
    else:
        makespan = -1
    return ("ortools", {"schedule": schedule, "makespan": makespan})
