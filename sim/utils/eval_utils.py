def run_episode(env, algo):
    """
    Runs an episode in the environment using the given algorithm.

    Args:
        env: The environment to run the episode in.
        algo: The algorithm used for action selection.

    Returns:
        ResultsHolder: Object containing the results of the episode.
    """
    algo.reset_states()
    results_holder = ResultsHolder()
    obs, _  = env.reset(seed=env.grid_config.seed)
    step_count = 0
    while True:
        obs, rew, dones, tr, infos = env.step(algo.act(obs))
        # print(""""______________________ """)
        # for agent_idx, agent_obs in enumerate(obs):
        #     pos = agent_obs['xy']         # 현재 위치
        #     target = agent_obs['target_xy']  # 목표 위치
        #     print(f"Agent {agent_idx}: Position {pos}, Target {target}")
        final_infos = infos
        results_holder.after_step(infos)

        if all(dones) or all(tr):
            break
        step_count +=1
    print(env.goal_sequences)
    print(f"total step : {step_count }")
    # ---- 마지막에 goal_times, wait_times 출력 ----
    print("==== 에이전트별 목표 이동 시간 및 대기 시간 ====")
    for agent_idx, info in enumerate(final_infos):
        goal_times = info.get("goal_times", [])
        wait_times = info.get("wait_times", [])
        print(f"Agent {agent_idx}:")
        print(f"  Goal별 이동 step 수   : {goal_times}")
        print(f"  Goal별 대기(wait) step: {wait_times}")
        print("-" * 40)

    return results_holder.get_final()

class ResultsHolder:
    """
    Holds and manages the results obtained during an episode.

    """

    def __init__(self):
        """
        Initializes an instance of ResultsHolder.
        """
        self.results = dict()

    def after_step(self, infos):
        """
        Updates the results with the metrics from the given information.

        Args:
            infos (List[dict]): List of dictionaries containing information about the episode.

        """
        if 'metrics' in infos[0]:
            self.results.update(**infos[0]['metrics'])

    def get_final(self):
        """
        Returns the final results obtained during the episode.

        Returns:
            dict: The final results.

        """
        return self.results

    def __repr__(self):
        """
        Returns a string representation of the ResultsHolder.

        Returns:
            str: The string representation of the ResultsHolder.

        """
        return str(self.get_final())
