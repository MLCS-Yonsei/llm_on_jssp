import random
import json

path = #"File path of 'llm_on_jssp'/" 

def generate_random_schedule(num_jobs, num_machines, min_ops=2, max_ops=4, min_dur=1, max_dur=10):
    
    schedule = []
    max_end_time = 0
    for _ in range(num_jobs):
        job_ops = []
        num_ops = random.randint(min_ops, max_ops)
        start_time = 0
        for _ in range(num_ops):
            machine = random.randint(0, num_machines - 1)
            duration = random.randint(min_dur, max_dur)
            end_time = start_time + duration
            job_ops.append((machine, start_time, end_time))
            start_time = end_time
        max_end_time = max(max_end_time, job_ops[-1][2])
        schedule.append(job_ops)
    makespan = max_end_time
    return {'solution': {'schedule': schedule, 'makespan': makespan}}

def schedule_to_natural_language_en(data):
    jobs = data['solution']['schedule']
    makespan = data['solution']['makespan']
    out = [f"The total makespan is {makespan}."]
    for j, job in enumerate(jobs, 1):
        steps = [f"machine {op[0]} from {op[1]} to {op[2]}" for op in job]
        out.append(f"Job {j} is processed on " + ", ".join(steps) + ".")
    return " ".join(out)


NUM_SAMPLES = 1000       # number of samples
NUM_JOBS = (2, 6)        # min~max of job_#
NUM_MACHINES = (2, 6)    # min~max of machine_#

with open(path + 'data_llm2_1k.jsonl', 'w', encoding='utf-8') as f:
    for _ in range(NUM_SAMPLES):
        nj = random.randint(*NUM_JOBS)
        nm = random.randint(*NUM_MACHINES)
        data = generate_random_schedule(nj, nm)
        input_str = json.dumps(data, ensure_ascii=False)
        output_str = schedule_to_natural_language_en(data)
        item = {"input": input_str, "output": output_str}
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
