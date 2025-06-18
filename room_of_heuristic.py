##### 휴리스틱 기법들을 모아둔 방 #####
### job별 operation 개수 비정형 가능 ###
## 기계 중복 허용 안함 추가 ##

import random
import time
from ortools.sat.python import cp_model

def _build_feasible_schedule(problem, job_order=None, machine_priority=None, randomize=False):
    """
    공통적으로 쓸 수 있는 충돌 없는 스케줄 생성기.
    job_order: 작업 순서 지정용 (None이면 원래 순서)
    machine_priority: 각 job 내 machine 순서 바꿀 때 사용
    randomize: 랜덤 offset 주기 등, 다양성 부여용
    """
    n_jobs = len(problem)
    schedule = []
    machine_available = {}        # {machine: 사용가능시간}
    job_last_end = [0] * n_jobs   # 각 job에서 이전 작업 종료 시점
    jobs = problem if job_order is None else [problem[i] for i in job_order]

    for jid, job in enumerate(jobs):
        j_idx = jid if job_order is None else job_order[jid]
        job_sched = []
        operations = job
        if machine_priority is not None:
            operations = [job[i] for i in machine_priority[j_idx]]
        for tid, (machine, duration) in enumerate(operations):
            start_time = max(machine_available.get(machine, 0), job_last_end[j_idx])
            # 랜덤하게 delay나 bias를 추가하고 싶다면
            if randomize:
                start_time += random.randint(0, 1)
            end_time = start_time + duration
            job_sched.append((machine, start_time, end_time))
            machine_available[machine] = end_time
            job_last_end[j_idx] = end_time
        schedule.append(job_sched)
    makespan = max(op[2] for job in schedule for op in job)
    return schedule, makespan

# --------- 휴리스틱 함수들 ----------

def solver_tabu(problem):
    # 실제 Tabu Search 구현은 복잡하니, 여기선 feasible한 스케줄만 만듭니다
    time.sleep(random.uniform(0.1, 0.4))
    schedule, makespan = _build_feasible_schedule(problem)
    return ("tabu", {"schedule": schedule, "makespan": makespan})

def solver_ga(problem):
    # 임의의 job 순서와 약간의 랜덤성을 주어 feasible 스케줄 생성
    time.sleep(random.uniform(0.2, 0.5))
    n = len(problem)
    job_order = list(range(n))
    random.shuffle(job_order)
    schedule, makespan = _build_feasible_schedule(problem, job_order=job_order, randomize=True)
    return ("ga", {"schedule": schedule, "makespan": makespan})

def solver_sa(problem):
    # Simulated Annealing처럼 약간의 랜덤 offset을 주고 feasible 스케줄
    time.sleep(random.uniform(0.15, 0.45))
    schedule, makespan = _build_feasible_schedule(problem, randomize=True)
    return ("sa", {"schedule": schedule, "makespan": makespan})

def solver_aco(problem):
    # pheromone_bias를 시뮬레이션. random bias로 start_time에 영향
    time.sleep(random.uniform(0.15, 0.5))
    pheromone_bias = random.randint(0, 1)
    def custom_randomize():
        return random.randint(0, pheromone_bias)
    # 아래 코드처럼, bias를 start_time에 더해주는 느낌으로 구현
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
    # 랜덤 요소 + 탐욕적으로 기계/작업 가용시점에 배치
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
