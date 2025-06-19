import gymnasium

import gymnasium
from follower_cpp.CustomGoalSequenceWrapper import CustomGoalSequenceWrapper

class ProvideMapWrapper(gymnasium.Wrapper):
    def reset(self, **kwargs):
        observations, infos = self.env.reset(seed=self.env.grid_config.seed)
        global_obstacles = self.get_global_obstacles()
        global_agents_xy = self.get_global_agents_xy()
        for idx, obs in enumerate(observations):
            obs['global_obstacles'] = global_obstacles
            obs['global_agent_xy'] = global_agents_xy[idx]
        print("[ProvideMapWrapper.reset] type(observations):", type(observations))
        print("[ProvideMapWrapper.reset] type(observations[0]):", type(observations[0]))
        print("[ProvideMapWrapper.reset] observations[0] keys:", list(observations[0].keys()))
        if 'obs' in observations[0]:
            print("[ProvideMapWrapper.reset] observations[0]['obs'] shape:", observations[0]['obs'].shape)
        return observations, infos

def follower_cpp_preprocessor(env, algo_config):
    env = ProvideMapWrapper(env)
    agent_goal_sequences = [
        [(2,2), (3,5), (4,4)],
        [(8,1), (1,7), (3,8)],
        [(6,5), (7,7), (0,0)]
    ]
    env = CustomGoalSequenceWrapper(env, agent_goal_sequences)
    return env

# class ProvideMapWrapper(gymnasium.Wrapper):
#     def reset(self, **kwargs):
#         observations, infos = self.env.reset(seed=self.env.grid_config.seed)
#         global_obstacles = self.get_global_obstacles()
#         global_agents_xy = self.get_global_agents_xy()
#         for idx, obs in enumerate(observations):
#             obs['global_obstacles'] = global_obstacles
#             obs['global_agent_xy'] = global_agents_xy[idx]
#         return observations, infos


# def follower_cpp_preprocessor(env, algo_config):
#     env = ProvideMapWrapper(env)
#     return env
