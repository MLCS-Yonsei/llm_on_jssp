import gymnasium
import numpy as np
from gymnasium.spaces import Box, Dict

class CustomGoalSequenceWrapper(gymnasium.Wrapper):
    def __init__(self, env, agent_goal_sequences):
        super().__init__(env)
        self.agent_goal_sequences = agent_goal_sequences
        self.agent_goal_indices = [0 for _ in agent_goal_sequences]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.agent_goal_indices = [0 for _ in self.agent_goal_sequences]
        # 첫 목표를 observation과 내부 환경 모두에 반영
        for i, seq in enumerate(self.agent_goal_sequences):
            obs[i]['target_xy'] = seq[0]
            # Pogema 내부 goal 변수 수정
            if hasattr(self.env, "agents"):
                self.env.agents[i].goal = seq[0]
            elif hasattr(self.env, "grid") and hasattr(self.env.grid, "goals"):
                self.env.grid.goals[i] = seq[0]
        return obs, info

    def step(self, actions):
        obs, rewards, dones, info = self.env.step(actions)
        for i, ob in enumerate(obs):
            # 목표 도달 체크: finished 플래그 or 좌표 비교
            goal_reached = ob.get('finished', ob['xy'] == ob['target_xy'])
            if goal_reached:
                self.agent_goal_indices[i] += 1
                if self.agent_goal_indices[i] < len(self.agent_goal_sequences[i]):
                    next_goal = self.agent_goal_sequences[i][self.agent_goal_indices[i]]
                    ob['target_xy'] = next_goal
                    # Pogema 내부 goal 변수까지 수정
                    if hasattr(self.env, "agents"):
                        self.env.agents[i].goal = next_goal
                    elif hasattr(self.env, "grid") and hasattr(self.env.grid, "goals"):
                        self.env.grid.goals[i] = next_goal
        return obs, rewards, dones, info
    
def convert_obs(obs):
    obs_channels = [obs['obstacles'], obs['agents']]
    obs_image = np.stack(obs_channels, axis=0)  # (2, 11, 11)
    obs_new = {'obs': obs_image}
    return obs_new

class ObsReformatWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # observation_space도 'obs'만 가지도록 새로 정의
        shape = (2, 11, 11)  # [채널수, 높이, 너비] - 반드시 네 환경에 맞게!
        self.observation_space = Dict({'obs': Box(low=0.0, high=1.0, shape=shape, dtype=np.float32)})

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = [convert_obs(o) for o in obs]
        return obs, info

    def step(self, actions):
        obs, rewards, dones, info = self.env.step(actions)
        obs = [convert_obs(o) for o in obs]
        return obs, rewards, dones, info