import argparse

from env.create_env import create_env_base
from env.custom_maps import MAPS_REGISTRY
from utils.eval_utils import run_episode
from follower.training_config import EnvironmentMazes
from follower.inference import FollowerInferenceConfig, FollowerInference
from follower.preprocessing import follower_preprocessor
from follower_cpp.inference import FollowerConfigCPP, FollowerInferenceCPP
from follower_cpp.preprocessing import follower_cpp_preprocessor
import warnings

def create_custom_env(cfg):
    env_cfg = EnvironmentMazes(with_animation=cfg.animation)
    
    # machine 0 : ( 0, 7)
    # machine 1 : ( 0,24)
    # machine 2 : ( 0,13)
    # machine 3 : ( 0,18)
    # machine 4 : (10,24)
    # machine 5 : (10,12)
    # machine 6 : (10, 0)
    
    goal_sequences = [
    [( 0,13), ( 0, 7), ( 0, 18),  (10,12),( 0,24)],
    [( 0,24), ( 0,13), (10, 0) ],  
    [(10, 0), ( 0,18), (10,12) ],
    [( 0, 7), (10,24), ( 0,24), ( 0,18)],  
    ]
    wait_sequences = [
        [10,  5,  20, 2, 12],  
        [15,  7,  20], 
        [10, 15,  20],
        [20,  5,  2, 10], 
    ]
    agents_start_pos= [(7,0), (7, 1), (7, 2), (7, 3)] 

    print("angents number",len(goal_sequences))
    env_cfg.grid_config.num_agents =len(goal_sequences)#3 # cfg.num_agents
    env_cfg.grid_config.map_name = cfg.map_name
    env_cfg.grid_config.seed = cfg.seed
    env_cfg.grid_config.max_episode_steps = cfg.max_episode_steps
    #env_cfg.grid_config.targets_xy= [( 0,24), (10, 0), (10,12), ( 0,18)] # dummy target
    # return create_env_base(env_cfg)
    return create_env_base(env_cfg,
                           goal_sequences=goal_sequences,
                           wait_sequences=wait_sequences,
                           agents_start_pos=agents_start_pos)


def run_follower(env):
    follower_cfg = FollowerInferenceConfig()
    algo = FollowerInference(follower_cfg)

    env = follower_preprocessor(env, follower_cfg)

    return run_episode(env, algo)


def run_follower_cpp(env):
    follower_cfg = FollowerConfigCPP(path_to_weights='model/follower-lite', num_threads=6)
    algo = FollowerInferenceCPP(follower_cfg)

    env = follower_cpp_preprocessor(env, follower_cfg)

    return run_episode(env, algo)


def main():
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description='Follower Inference Script')
    parser.add_argument('--animation', action='store_false', help='Enable animation (default: %(default)s)')
    parser.add_argument('--num_agents', type=int, default=3, help='Number of agents (default: %(default)d)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed (default: %(default)d)')
    # parser.add_argument('--map_name', type=str, default='wfi_warehouse', help='Map name (default: %(default)s)')
    parser.add_argument('--map_name', type=str, default='mlp_test', help='Map name (default: %(default)s)')
    parser.add_argument('--max_episode_steps', type=int, default=512,
                        help='Maximum episode steps (default: %(default)d)')
    parser.add_argument('--show_map_names', action='store_true', help='Shows names of all available maps')

    parser.add_argument('--algorithm', type=str, choices=['Follower', 'FollowerLite'], default='Follower',
                        help='Algorithm to use: "Follower" or "FollowerLite" (default: "Follower")')

    args = parser.parse_args()

    if args.show_map_names:
        for map_ in MAPS_REGISTRY:
            print(map_)
        return

    if args.algorithm == 'FollowerLite':
        print(run_follower_cpp(create_custom_env(args)))
    else:  # Default to 'Follower'
        print(run_follower(create_custom_env(args)))

if __name__ == '__main__':
    main()
