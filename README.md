# 🤖 Job Shop Scheduling from Language: A Heuristic-based Framework with Large Language Models

> Team 5  
> Yujin Kim (2024321236), Yonghwa Seo (2025311450)

---

##  Project Overview

This project presents a novel framework that tackles the **Job Shop Scheduling Problem (JSSP)** using **Large Language Models (LLMs)**. By interpreting human language descriptions and combining them with environment information, the system generates optimized job schedules using a suite of heuristic algorithms.

---

##  Problem Statement

The **Job Shop Scheduling Problem (JSSP)** is a classic optimization task where:

- Multiple jobs must be processed through a set of machines.
- Each job consists of a sequence of operations with defined order.
- Each machine can process only one operation at a time.
- Different jobs may need the same machines, leading to conflicts.

Additionally, this project incorporates **agent movement time** and **collision avoidance** to reflect real-world robotic shop floors.


| Symbol         | Meaning                                       |
|----------------|-----------------------------------------------|
| 𝐽𝑖             | *i*‑th job (total *m* jobs)                    |
| 𝑂𝑖,𝑗           | *j*‑th operation of job 𝐽𝑖                      |
| 𝑀𝑘             | *k*‑th machine                                 |
| 𝐴𝑠             | *s*‑th agent or worker                         |
| 𝐶𝑖             | Completion time of job 𝐽𝑖                      |
| 𝐶ₘₐₓ          | Makespan = max(𝐶𝑖)                             |
| 𝑇𝑠,𝑥,𝑦        | Time for agent *s* to move from 𝑀ₓ to 𝑀ᵧ        |
| 𝐷ₜ             | Time delay due to the *t*‑th collision         |

---

##  Method

###  LLM-Based Pipeline

This project is structured to accept **natural language job descriptions as input** and produce **natural language responses as output**, enabling seamless interaction between human-readable tasks and scheduling solutions.

To achieve this, two Large Language Models (LLMs) are utilized:

### 1. LLM1 (From Language to Matrix)

- **Purpose**: Converts natural language prompts into structured job matrices
- **Input**: Natural language job description (prompt)
- **Output**: JSON object containing:
  - Matrix format of the job schedule
  - Label information (e.g., best solver criteria)
- **Dataset**:
  - 5000 synthetic samples
  - Randomly generated combinations of:
    - Number of jobs and operations
    - Machine assignments
    - Processing durations
   
### 2. Integrated Optimizer

### 3. LLM2 (From Matrix to Language)

- **Purpose**: Converts the matrix-form solution (from solver output) back into natural language
- **Input**: JSON-format schedule matrix
- **Output**: Natural language description of the optimized schedule
- **Dataset**:
  - 1000 samples
  - Pairs of matrix schedule and fixed-format human-readable explanation

---

##  4. Simulation (Lifelong Multi-Agent Pathfinding Policy)
## License
This project is licensed under the MIT License.  
Some portions of the code are adapted from [AIRI-Institute/learn-to-follow](https://github.com/AIRI-Institute/learn-to-follow) under the same license.  
See the [LICENSE](.learn_follow/LICENSE) file for details.

<p align="center">
  <img src="./docker/image.png" alt="Lifelong MAPF Example" width="300"/>
</p>

## Policy Description

The core policy implemented in this project is based on the **Follower** algorithm proposed in  
["Learn to Follow: Decentralized Lifelong Multi-agent Pathfinding via Planning and Learning"](https://arxiv.org/pdf/2310.01207).

**Key features:**
- **Decentralized planning:**  
  Each agent plans its path independently using a global planner, then continually replans based on observed conflicts.
- **Lifelong pathfinding:**  
  Agents are repeatedly assigned new goals in a persistent, dynamic environment, requiring both reactivity and long-term strategy.
- **Local conflict resolution:**  
  A local reinforcement learning (RL) policy resolves collisions and deadlocks online, complementing the global planner.
- **Scalability and robustness:**  
  The combined global planning and local RL allow the system to handle large numbers of agents and complex, cluttered maps.

**Illustration:**  
The figure above visualizes a typical lifelong multi-agent scenario.  
- Each circle represents an agent's goal.  
- The red area shows the local observation used by the RL policy.  
- Agents must coordinate their motion to achieve all goals efficiently, avoiding obstacles and each other in real-time.

For more details, see the original [paper](https://arxiv.org/pdf/2310.01207).

---

**Project Modifications and Scope** 
Job Scheduling with Wait Times:
In this simulation, each agent is given a fixed sequence of targets (jobs) to visit. Upon arriving at each target, the agent waits for a specified processing (wait) time, simulating job execution as in job-shop scheduling (JSSP). This setup allows for realistic modeling of both movement and task processing phases.

**Simulation-Based Evaluation** 
This project does not introduce a new scheduling algorithm. Instead, it evaluates how well predefined job schedules and wait times are executed in a multi-agent environment that uses the decentralized “Follower” policy for pathfinding and conflict resolution.

**Path and Schedule Consistency Analysis** 
The simulation records both the planned job sequences and the actual agent trajectories. By comparing the scheduled plans with the realized execution—including delays, waiting times, and conflicts—it quantifies the effects of path conflicts and congestion on overall task completion.

**Motivation for Path-aware Scheduling** 
Discrepancies between scheduled and actual task completions observed in the simulation motivate further research into scheduling approaches that jointly consider movement, congestion, and task timing. This simulation serves as a baseline for such future work.

## Summary:
This project provides a simulation framework for analyzing the interplay between job scheduling and multi-agent pathfinding, highlighting the challenges of schedule fidelity in dynamic, congested environments.

---

## Installation:

# 1. Create and activate the Conda environment (Python 3.8)
conda create -n mlp_jssp_project python=3.8
conda activate mlp_jssp_project

# 2. Install required Python packages
pip install -r docker/requirements.txt

---

## Map, Goal Sequences, and Wait Sequences

### Map

- The map configuration file is stored at:  
  `env/test-mapz.yaml`
- You can specify the map file by editing the parser argument in your code:

```python
parser.add_argument('--map_name', type=str, default='mlp_test', help='Map name (default: %(default)s)')
```

### Goal Sequences & Wait Sequences

- Edit these variables in test.py, specifically in the create_custom_env function:

```python
    goal_sequences = [
        [(0, 13), (0, 7), (0, 18), (10, 12), (0, 24)],
        [(0, 24), (0, 13), (10, 0)],
        [(10, 0), (0, 18), (10, 12)],
        [(0, 7), (10, 24), (0, 24), (0, 18)],
    ]

    wait_sequences = [
        [10, 5, 20, 2, 12],
        [15, 7, 20],
        [10, 15, 20],
        [20, 5, 2, 10],
    ]

    agents_start_pos = [(7, 0), (7, 1), (7, 2), (7, 3)]
```

- goal_sequences: The list of goals (targets) each agent will visit in order.

- wait_sequences: The wait time at each corresponding goal for each agent.

- agents_start_pos: The starting positions for each agent.

---


#### Baseline Policy : Lifelong Multi-Agent Pathfinding Policy
#### License
This project is licensed under the MIT License.  
Some portions of the code are adapted from [AIRI-Institute/learn-to-follow](https://github.com/AIRI-Institute/learn-to-follow) under the same license.  
See the [LICENSE](.learn_follow/LICENSE) file for details.

<p align="center">
  <img src=".sim/docker/image.png" alt="Lifelong MAPF Example" width="300"/>
</p>

#### Policy Description

The core policy implemented in this project is based on the **Follower** algorithm proposed in  
["Learn to Follow: Decentralized Lifelong Multi-agent Pathfinding via Planning and Learning"](https://arxiv.org/pdf/2310.01207).

**Key features:**
- **Decentralized planning:**  
  Each agent plans its path independently using a global planner, then continually replans based on observed conflicts.
- **Lifelong pathfinding:**  
  Agents are repeatedly assigned new goals in a persistent, dynamic environment, requiring both reactivity and long-term strategy.
- **Local conflict resolution:**  
  A local reinforcement learning (RL) policy resolves collisions and deadlocks online, complementing the global planner.
- **Scalability and robustness:**  
  The combined global planning and local RL allow the system to handle large numbers of agents and complex, cluttered maps.

**Illustration:**  
The figure above visualizes a typical lifelong multi-agent scenario.  
- Each circle represents an agent's goal.  
- The red area shows the local observation used by the RL policy.  
- Agents must coordinate their motion to achieve all goals efficiently, avoiding obstacles and each other in real-time.

For more details, see the original [paper](https://arxiv.org/pdf/2310.01207).

---

**Project Modifications and Scope** 
Job Scheduling with Wait Times:
In this simulation, each agent is given a fixed sequence of targets (jobs) to visit. Upon arriving at each target, the agent waits for a specified processing (wait) time, simulating job execution as in job-shop scheduling (JSSP). This setup allows for realistic modeling of both movement and task processing phases.

**Simulation-Based Evaluation** 
This project does not introduce a new scheduling algorithm. Instead, it evaluates how well predefined job schedules and wait times are executed in a multi-agent environment that uses the decentralized “Follower” policy for pathfinding and conflict resolution.

**Path and Schedule Consistency Analysis** 
The simulation records both the planned job sequences and the actual agent trajectories. By comparing the scheduled plans with the realized execution—including delays, waiting times, and conflicts—it quantifies the effects of path conflicts and congestion on overall task completion.

**Motivation for Path-aware Scheduling** 
Discrepancies between scheduled and actual task completions observed in the simulation motivate further research into scheduling approaches that jointly consider movement, congestion, and task timing. This simulation serves as a baseline for such future work.

#### Summary:
This project provides a simulation framework for analyzing the interplay between job scheduling and multi-agent pathfinding, highlighting the challenges of schedule fidelity in dynamic, congested environments.

---

### Installation:

#### 1. Create and activate the Conda environment (Python 3.8)
conda create -n mlp_jssp_project python=3.8
conda activate mlp_jssp_project

##### 2. Install required Python packages
pip install -r docker/requirements.txt

---

### Map, Goal Sequences, and Wait Sequences

#### Map

- The map configuration file is stored at:  
  `env/test-mapz.yaml`
- You can specify the map file by editing the parser argument in your code:

```python
parser.add_argument('--map_name', type=str, default='mlp_test', help='Map name (default: %(default)s)')
```

#### Goal Sequences & Wait Sequences

- Edit these variables in test.py, specifically in the create_custom_env function:

```python
    goal_sequences = [
        [(0, 13), (0, 7), (0, 18), (10, 12), (0, 24)],
        [(0, 24), (0, 13), (10, 0)],
        [(10, 0), (0, 18), (10, 12)],
        [(0, 7), (10, 24), (0, 24), (0, 18)],
    ]

    wait_sequences = [
        [10, 5, 20, 2, 12],
        [15, 7, 20],
        [10, 15, 20],
        [20, 5, 2, 10],
    ]

    agents_start_pos = [(7, 0), (7, 1), (7, 2), (7, 3)]
```

- goal_sequences: The list of goals (targets) each agent will visit in order.

- wait_sequences: The wait time at each corresponding goal for each agent.

- agents_start_pos: The starting positions for each agent.

---

##  Results 

### 1. LLM Conversion Accuracy

| Model             | EM Acc | Format Fail | Avg Inference Time |
|------------------|--------|-------------|--------------------|
| Fine-tuned LLM   | 36.00% | 32 / 50     | 7.597 s            |
| Mistral-original | 0.00%  | 50 / 50     | 5.066 s            |
| Seq2seq baseline | 0.00%  | 50 / 50     | 0.024 s            |

### 2. Schedule Optimization Example

Job Matrix:
```json
[
  [[2,10],[0,5],[3,20],[5,2],[1,12]],
  [[1,15],[2,7],[6,20]],
  [[6,10],[2,15],[5,20]],
  [[0,20],[4,5],[1,2],[3,10]]
]

