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

1. **Input**:  
   - Natural language job description (prompt)  
   - PNG image of environment (machine layout)

2. **LLM1**:
   - Parses prompt and environment image  
   - Converts to matrix (job structure + movement times)

3. **Heuristic Solvers**:  
   - Tabu Search, Genetic Algorithm, Simulated Annealing  
   - OR-Tools, GRASP, Ant Colony Optimization  
   - Returns best schedule based on makespan or speed

4. **LLM2**:  
   - Converts optimized matrix back to natural language

---

##  Dataset

- **LLM1 Dataset**:  
  - 5000 synthetic prompts with varied syntax/structure  
  - Generated using combinations of phrases for job, machine, and time

- **LLM2 Dataset**:  
  - 1000 samples pairing matrix input with fixed-form natural language output

Example prompt:


---

##  Simulation

- Modified **Pogema simulator** to reflect:
  - Agent movement
  - Job sequences
  - Collision avoidance
  - Decentralized policy using **Learn to Follow**

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

