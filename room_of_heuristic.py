##### 휴리스틱 기법들을 모아둔 방 #####

import random
import time
from ortools.sat.python import cp_model

# 1. Tabu Search (의사코드 기반)
def solver_tabu(problem):
    time.sleep(random.uniform(0.1, 0.4))
    jobs = len(problem)
    machines = len(problem[0])
    # 무작위 순서로 simple schedule 생성
    start = 0
    schedule = []
    for i in range(jobs):
        job_sched = []
        t = start
        for j in range(machines):
            duration = problem[i][j][1]
            job_sched.append((problem[i][j][0], t, t+duration))
            t += duration
        schedule.append(job_sched)
    makespan = max(j[-1] for job in schedule for j in job)
    return ("tabu", {"schedule": schedule, "makespan": makespan})

# 2. Genetic Algorithm (의사코드 기반)
def solver_ga(problem):
    time.sleep(random.uniform(0.2, 0.5))
    jobs = len(problem)
    machines = len(problem[0])
    # 랜덤 크로스오버 시뮬
    start = 0
    schedule = []
    for i in range(jobs):
        job_sched = []
        t = start + random.randint(0, 2)
        for j in range(machines):
            duration = problem[i][j][1]
            job_sched.append((problem[i][j][0], t, t+duration))
            t += duration
        schedule.append(job_sched)
    makespan = max(j[-1] for job in schedule for j in job)
    return ("ga", {"schedule": schedule, "makespan": makespan})

# 3. Simulated Annealing (의사코드 기반)
def solver_sa(problem):
    time.sleep(random.uniform(0.15, 0.45))
    jobs = len(problem)
    machines = len(problem[0])
    start = random.randint(0, 1)
    schedule = []
    for i in range(jobs):
        job_sched = []
        t = start
        for j in range(machines):
            duration = problem[i][j][1]
            job_sched.append((problem[i][j][0], t, t+duration))
            t += duration + random.randint(0, 1)
        schedule.append(job_sched)
    makespan = max(j[-1] for job in schedule for j in job)
    return ("sa", {"schedule": schedule, "makespan": makespan})

# 4. Ant Colony Optimization (의사코드 기반)
def solver_aco(problem):
    time.sleep(random.uniform(0.15, 0.5))
    jobs = len(problem)
    machines = len(problem[0])
    schedule = []
    pheromone_bias = random.uniform(0, 1)
    for i in range(jobs):
        job_sched = []
        t = 0
        for j in range(machines):
            duration = problem[i][j][1]
            job_sched.append((problem[i][j][0], t, t+duration+int(pheromone_bias)))
            t += duration + int(pheromone_bias)
        schedule.append(job_sched)
    makespan = max(j[-1] for job in schedule for j in job)
    return ("aco", {"schedule": schedule, "makespan": makespan})

# 5. GRASP (의사코드 기반)
def solver_grasp(problem):
    time.sleep(random.uniform(0.1, 0.35))
    jobs = len(problem)
    machines = len(problem[0])
    schedule = []
    for i in range(jobs):
        job_sched = []
        t = random.randint(0, 2)
        for j in range(machines):
            duration = problem[i][j][1]
            job_sched.append((problem[i][j][0], t, t+duration))
            t += duration
        schedule.append(job_sched)
    makespan = max(j[-1] for job in schedule for j in job)
    return ("grasp", {"schedule": schedule, "makespan": makespan})

# 6. OR-Tools (실제 작동 예시)
def solver_ortools(problem):
    """
    OR-Tools로 JSSP를 푸는 함수. machine 번호가 연속/불연속/큰 값이어도 에러 없이 동작.
    입력: problem = [[(machine_id, duration), ...], ...]
    출력: ("ortools", {"schedule": [...], "makespan": ...})
    """
    jobs_data = problem  # [[(machine_id, duration), ...], ...]
    model = cp_model.CpModel()
    num_jobs = len(jobs_data)
    horizon = sum(task[1] for job in jobs_data for task in job)
    all_tasks = {}
    machine_to_intervals = dict()   # dict로 선언!

    for job_id, job in enumerate(jobs_data):
        previous_end = None
        for task_id, (machine, duration) in enumerate(job):
            suffix = f'_{job_id}_{task_id}'
            start_var = model.NewIntVar(0, horizon, 'start' + suffix)
            end_var = model.NewIntVar(0, horizon, 'end' + suffix)
            interval = model.NewIntervalVar(start_var, duration, end_var, 'interval' + suffix)
            all_tasks[(job_id, task_id)] = (start_var, end_var, interval)
            # dict에 머신별로 interval 모으기
            if machine not in machine_to_intervals:
                machine_to_intervals[machine] = []
            machine_to_intervals[machine].append(interval)
            if previous_end is not None:
                model.Add(start_var >= previous_end)
            previous_end = end_var

    # 모든 머신별 NoOverlap 제약 추가
    for intervals in machine_to_intervals.values():
        model.AddNoOverlap(intervals)

    obj_var = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(obj_var, [end for (_, end, _) in all_tasks.values()])
    model.Minimize(obj_var)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    schedule = []
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        for job_id in range(num_jobs):
            job_sched = []
            for task_id in range(len(jobs_data[job_id])):
                start = int(solver.Value(all_tasks[(job_id, task_id)][0]))
                end = int(solver.Value(all_tasks[(job_id, task_id)][1]))
                machine = jobs_data[job_id][task_id][0]
                job_sched.append((machine, start, end))
            schedule.append(job_sched)
        makespan = int(solver.ObjectiveValue())
    else:
        makespan = -1
    return ("ortools", {"schedule": schedule, "makespan": makespan})
