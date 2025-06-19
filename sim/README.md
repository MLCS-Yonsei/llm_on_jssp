
# Lifelong Multi-Agent Pathfinding Policy
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
