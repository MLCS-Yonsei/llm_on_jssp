from typing import Optional

import numpy as np
import gymnasium
from gymnasium.error import ResetNeeded

from pogema.grid import Grid, GridLifeLong
from pogema.grid_config import GridConfig
from pogema.wrappers.metrics import LifeLongAverageThroughputMetric, NonDisappearEpLengthMetric, \
    NonDisappearCSRMetric, NonDisappearISRMetric, EpLengthMetric, ISRMetric, CSRMetric, SumOfCostsAndMakespanMetric
from pogema.wrappers.multi_time_limit import MultiTimeLimit
from pogema.generator import generate_new_target, generate_from_possible_targets
from pogema.wrappers.persistence import PersistentWrapper


class ActionsSampler:
    """
    Samples the random actions for the given number of agents using the given seed.
    """

    def __init__(self, num_actions, seed=42):
        self._num_actions = num_actions
        self._rnd = None
        self.update_seed(seed)

    def update_seed(self, seed=None):
        self._rnd = np.random.default_rng(seed)

    def sample_actions(self, dim=1):
        return self._rnd.integers(self._num_actions, size=dim)


class PogemaBase(gymnasium.Env):
    """
    Abstract class of the Pogema environment.
    """
    metadata = {"render_modes": ["ansi"], }

    def step(self, action):
        raise NotImplementedError

    def reset(self, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None, ):
        raise NotImplementedError

    def __init__(self, grid_config: GridConfig = GridConfig()):
        # noinspection PyTypeChecker
        self.grid: Grid = None
        self.grid_config = grid_config

        self.action_space: gymnasium.spaces.Discrete = gymnasium.spaces.Discrete(len(self.grid_config.MOVES))
        self._multi_action_sampler = ActionsSampler(self.action_space.n, seed=self.grid_config.seed)

    def _get_agents_obs(self, agent_id=0):
        """
        Returns the observation of the agent with the given id.
        :param agent_id:
        :return:
        """
        return np.concatenate([
            self.grid.get_obstacles_for_agent(agent_id)[None],
            self.grid.get_positions(agent_id)[None],
            self.grid.get_square_target(agent_id)[None]
        ])

    def check_reset(self):
        """
        Checks if the reset needed.
        :return:
        """
        if self.grid is None:
            raise ResetNeeded("Please reset environment first!")

    def render(self, mode='human'):
        """
        Renders the environment using ascii graphics.
        :param mode:
        :return:
        """
        self.check_reset()
        return self.grid.render(mode=mode)

    def sample_actions(self):
        """
        Samples the random actions for the given number of agents.
        :return:
        """
        return self._multi_action_sampler.sample_actions(dim=self.grid_config.num_agents)

    def get_num_agents(self):
        """
        Returns the number of agents in the environment.
        :return:
        """
        return self.grid_config.num_agents


class Pogema(PogemaBase):
    def __init__(self, grid_config=GridConfig(num_agents=2)):
        super().__init__(grid_config)
        self.was_on_goal = None
        full_size = self.grid_config.obs_radius * 2 + 1
        if self.grid_config.observation_type == 'default':
            self.observation_space = gymnasium.spaces.Box(-1.0, 1.0, shape=(3, full_size, full_size))
        elif self.grid_config.observation_type == 'POMAPF':
            self.observation_space: gymnasium.spaces.Dict = gymnasium.spaces.Dict(
                obstacles=gymnasium.spaces.Box(0.0, 1.0, shape=(full_size, full_size)),
                agents=gymnasium.spaces.Box(0.0, 1.0, shape=(full_size, full_size)),
                xy=gymnasium.spaces.Box(low=-1024, high=1024, shape=(2,), dtype=int),
                target_xy=gymnasium.spaces.Box(low=-1024, high=1024, shape=(2,), dtype=int),
            )
        elif self.grid_config.observation_type == 'MAPF':
            self.observation_space: gymnasium.spaces.Dict = gymnasium.spaces.Dict(
                obstacles=gymnasium.spaces.Box(0.0, 1.0, shape=(full_size, full_size)),
                agents=gymnasium.spaces.Box(0.0, 1.0, shape=(full_size, full_size)),
                xy=gymnasium.spaces.Box(low=-1024, high=1024, shape=(2,), dtype=int),
                target_xy=gymnasium.spaces.Box(low=-1024, high=1024, shape=(2,), dtype=int),
                # global_obstacles=None, # todo define shapes of global state variables
                # global_xy=None,
                # global_target_xy=None,
            )
        else:
            raise ValueError(f"Unknown observation type: {self.grid.config.observation_type}")

    def step(self, action: list):
        assert len(action) == self.grid_config.num_agents
        rewards = []

        terminated = []

        self.move_agents(action)
        self.update_was_on_goal()

        for agent_idx in range(self.grid_config.num_agents):

            on_goal = self.grid.on_goal(agent_idx)
            if on_goal and self.grid.is_active[agent_idx]:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
            terminated.append(on_goal)

        for agent_idx in range(self.grid_config.num_agents):
            if self.grid.on_goal(agent_idx):
                self.grid.hide_agent(agent_idx)
                self.grid.is_active[agent_idx] = False

        infos = self._get_infos()

        observations = self._obs()
        truncated = [False] * self.grid_config.num_agents
        return observations, rewards, terminated, truncated, infos

    def _initialize_grid(self):
        self.grid: Grid = Grid(grid_config=self.grid_config)

    def update_was_on_goal(self):
        self.was_on_goal = [self.grid.on_goal(agent_idx) and self.grid.is_active[agent_idx]
                            for agent_idx in range(self.grid_config.num_agents)]

    def reset(self, seed: Optional[int] = None, return_info: bool = True, options: Optional[dict] = None, ):
        self._initialize_grid()
        self.update_was_on_goal()

        if seed is not None:
            self.grid.seed = seed

        if return_info:
            return self._obs(), self._get_infos()
        return self._obs()

    def _obs(self):
        if self.grid_config.observation_type == 'default':
            return [self._get_agents_obs(index) for index in range(self.grid_config.num_agents)]
        elif self.grid_config.observation_type == 'POMAPF':
            return self._pomapf_obs()

        elif self.grid_config.observation_type == 'MAPF':
            results = self._pomapf_obs()
            global_obstacles = self.grid.get_obstacles()
            global_agents_xy = self.grid.get_agents_xy()
            global_targets_xy = self.grid.get_targets_xy()

            for agent_idx in range(self.grid_config.num_agents):
                result = results[agent_idx]
                result.update(global_obstacles=global_obstacles)
                result['global_xy'] = global_agents_xy[agent_idx]
                result['global_target_xy'] = global_targets_xy[agent_idx]

            return results
        else:
            raise ValueError(f"Unknown observation type: {self.grid.config.observation_type}")

    def _pomapf_obs(self):
        results = []
        agents_xy_relative = self.grid.get_agents_xy_relative()
        targets_xy_relative = self.grid.get_targets_xy_relative()

        for agent_idx in range(self.grid_config.num_agents):
            result = {'obstacles': self.grid.get_obstacles_for_agent(agent_idx),
                      'agents': self.grid.get_positions(agent_idx),
                      'xy': agents_xy_relative[agent_idx],
                      'target_xy': targets_xy_relative[agent_idx]}

            results.append(result)
        return results

    def _get_infos(self):
        infos = [dict() for _ in range(self.grid_config.num_agents)]
        for agent_idx in range(self.grid_config.num_agents):
            infos[agent_idx]['is_active'] = self.grid.is_active[agent_idx]

        return infos


    def _revert_action(self, agent_idx, used_cells, cell, actions):
        actions[agent_idx] = 0
        used_cells[cell].remove(agent_idx)
        new_cell = self.grid.positions_xy[agent_idx]
        if new_cell in used_cells and len(used_cells[new_cell]) > 0:
            used_cells[new_cell].append(agent_idx)
            return self._revert_action(used_cells[new_cell][0], used_cells, new_cell, actions)
        else:
            used_cells.setdefault(new_cell, []).append(agent_idx)
        return actions, used_cells

    def move_agents(self, actions):
        if self.grid.config.collision_system == 'priority':
            for agent_idx in range(self.grid_config.num_agents):
                if self.grid.is_active[agent_idx]:
                    self.grid.move(agent_idx, actions[agent_idx])
        elif self.grid.config.collision_system == 'block_both':
            used_cells = {}
            agents_xy = self.grid.get_agents_xy()
            for agent_idx, (x, y) in enumerate(agents_xy):
                if self.grid.is_active[agent_idx]:
                    dx, dy = self.grid_config.MOVES[actions[agent_idx]]
                    used_cells[x + dx, y + dy] = 'blocked' if (x + dx, y + dy) in used_cells else 'visited'
                    used_cells[x, y] = 'blocked'
            for agent_idx in range(self.grid_config.num_agents):
                if self.grid.is_active[agent_idx]:
                    x, y = agents_xy[agent_idx]
                    dx, dy = self.grid_config.MOVES[actions[agent_idx]]
                    if used_cells.get((x + dx, y + dy), None) != 'blocked':
                        self.grid.move(agent_idx, actions[agent_idx])
        elif self.grid.config.collision_system == 'soft':
            used_cells = dict()
            used_edges = dict()
            agents_xy = self.grid.get_agents_xy()
            for agent_idx, (x, y) in enumerate(agents_xy):
                if self.grid.is_active[agent_idx]:
                    dx, dy = self.grid.config.MOVES[actions[agent_idx]]
                    used_cells.setdefault((x + dx, y + dy), []).append(agent_idx)
                    used_edges[x, y, x + dx, y + dy] = [agent_idx]
                    if dx != 0 or dy != 0:
                        used_edges.setdefault((x + dx, y + dy, x, y), []).append(agent_idx)
            for agent_idx, (x, y) in enumerate(agents_xy):
                if self.grid.is_active[agent_idx]:
                    dx, dy = self.grid.config.MOVES[actions[agent_idx]]
                    if len(used_edges[x, y, x + dx, y + dy]) > 1:
                        used_cells[x + dx, y + dy].remove(agent_idx)
                        used_cells.setdefault((x, y), []).append(agent_idx)
                        actions[agent_idx] = 0
            for agent_idx in reversed(range(len(agents_xy))):
                x, y = agents_xy[agent_idx]
                if self.grid.is_active[agent_idx]:
                    dx, dy = self.grid.config.MOVES[actions[agent_idx]]
                    if len(used_cells[x + dx, y + dy]) > 1 or self.grid.has_obstacle(x + dx, y + dy):
                        actions, used_cells = self._revert_action(agent_idx, used_cells, (x + dx, y + dy), actions)
            for agent_idx in range(self.grid_config.num_agents):
                if self.grid.is_active[agent_idx]:
                    self.grid.move_without_checks(agent_idx, actions[agent_idx])
        else:
            raise ValueError('Unknown collision system: {}'.format(self.grid.config.collision_system))

    def get_agents_xy_relative(self):
        return self.grid.get_agents_xy_relative()

    def get_targets_xy_relative(self):
        return self.grid.get_targets_xy_relative()

    def get_obstacles(self, ignore_borders=False):
        return self.grid.get_obstacles(ignore_borders=ignore_borders)

    def get_agents_xy(self, only_active=False, ignore_borders=False):
        return self.grid.get_agents_xy(only_active=only_active, ignore_borders=ignore_borders)

    def get_targets_xy(self, only_active=False, ignore_borders=False):
        return self.grid.get_targets_xy(only_active=only_active, ignore_borders=ignore_borders)

    def get_state(self, ignore_borders=False, as_dict=False):
        return self.grid.get_state(ignore_borders=ignore_borders, as_dict=as_dict)


# class PogemaLifeLong(Pogema):
#     def __init__(self, grid_config=GridConfig(num_agents=2)):
#         super().__init__(grid_config)

#     def _initialize_grid(self):
#         self.grid: GridLifeLong = GridLifeLong(grid_config=self.grid_config)

#         main_rng = np.random.default_rng(self.grid_config.seed)
#         seeds = main_rng.integers(np.iinfo(np.int32).max, size=self.grid_config.num_agents)
#         self.random_generators = [np.random.default_rng(seed) for seed in seeds]

#     def _generate_new_target(self, agent_idx):
#         if self.grid_config.possible_targets_xy is not None:
#             new_goal = generate_from_possible_targets(self.random_generators[agent_idx], 
#                                                      self.grid_config.possible_targets_xy, 
#                                                      self.grid.positions_xy[agent_idx])
#             return (new_goal[0] + self.grid_config.obs_radius, new_goal[1] + self.grid_config.obs_radius)
#         else:
#             return generate_new_target(self.random_generators[agent_idx],
#                                     self.grid.point_to_component,
#                                     self.grid.component_to_points,
#                                     self.grid.positions_xy[agent_idx])

#     def step(self, action: list):
#         assert len(action) == self.grid_config.num_agents
#         rewards = []

#         infos = [dict() for _ in range(self.grid_config.num_agents)]

#         self.move_agents(action)
#         self.update_was_on_goal()

#         for agent_idx in range(self.grid_config.num_agents):
#             on_goal = self.grid.on_goal(agent_idx)
#             if on_goal and self.grid.is_active[agent_idx]:
#                 rewards.append(1.0)
#             else:
#                 rewards.append(0.0)

#             if self.grid.on_goal(agent_idx):
#                 self.grid.finishes_xy[agent_idx] = self._generate_new_target(agent_idx)

#         for agent_idx in range(self.grid_config.num_agents):
#             infos[agent_idx]['is_active'] = self.grid.is_active[agent_idx]

#         obs = self._obs()

#         terminated = [False] * self.grid_config.num_agents
#         truncated = [False] * self.grid_config.num_agents
#         return obs, rewards, terminated, truncated, infos
class PogemaLifeLong(Pogema):
    def __init__(self, grid_config=GridConfig(num_agents=2), goal_sequences=None, wait_steps=None):
        """
        :param grid_config: 환경 설정
        :param goal_sequences: 각 에이전트별 순차 목표 리스트 [[(x1,y1), (x2,y2), ...], ...]
        :param wait_steps: 목표 도달 후 대기할 스텝 수
        """
        super().__init__(grid_config)

        # 목표 시퀀스 및 인덱스 초기화
        self.goal_sequences = goal_sequences

        if goal_sequences is not None:
            self.goal_indices = [0] * self.grid_config.num_agents
        else:
            self.goal_indices = None

        # 대기 시간 설정
        self.wait_steps = 3
        self.wait_sequences =wait_steps
        self.wait_counters = [0] * self.grid_config.num_agents

    def to_grid_coords(self,x, y, obs_radius):
        return x + obs_radius, y + obs_radius

    def _initialize_grid(self):
        self.grid: GridLifeLong = GridLifeLong(grid_config=self.grid_config)
    
        main_rng = np.random.default_rng(self.grid_config.seed)
        seeds = main_rng.integers(np.iinfo(np.int32).max, size=self.grid_config.num_agents)
        self.random_generators = [np.random.default_rng(seed) for seed in seeds]
    
        if self.goal_sequences is not None:
            obs_radius = self.grid_config.obs_radius
            for agent_idx in range(self.grid_config.num_agents):
                x, y = self.goal_sequences[agent_idx][0]
                xg, yg = self.to_grid_coords(x, y, obs_radius)
                self.grid.finishes_xy[agent_idx] = (xg, yg)
        


    def _generate_new_target(self, agent_idx):
        # 1. goal_sequences에 목표가 있으면 그걸 우선 사용
        if self.goal_sequences is not None:
            idx = self.goal_indices[agent_idx]
            seq = self.goal_sequences[agent_idx]
            if idx < len(seq):
                # obs_radius 만큼 offset 적용 (좌표 변환)
                x, y = seq[idx]
                xg, yg = x + self.grid_config.obs_radius, y + self.grid_config.obs_radius
                return (xg, yg)
            else:
                x, y = seq[-1]
                xg, yg = x + self.grid_config.obs_radius, y + self.grid_config.obs_radius
                return (xg, yg)

        # 2. goal_sequences가 없을 때만 기존 랜덤 목표 생성
        if self.grid_config.possible_targets_xy is not None:
            new_goal = generate_from_possible_targets(
                self.random_generators[agent_idx],
                self.grid_config.possible_targets_xy,
                self.grid.positions_xy[agent_idx]
            )
            xg, yg = new_goal[0] + self.grid_config.obs_radius, new_goal[1] + self.grid_config.obs_radius
            return (xg, yg)
        else:
            return generate_new_target(
                self.random_generators[agent_idx],
                self.grid.point_to_component,
                self.grid.component_to_points,
                self.grid.positions_xy[agent_idx]
            )

    def reset(self, seed: Optional[int] = None, return_info: bool = True, options: Optional[dict] = None, ):
        self._initialize_grid()
        self.update_was_on_goal()
    
        # --- 타임 스탯 초기화 ---
        self.goal_reach_steps = [0] * self.grid_config.num_agents  # 골 도달 후 카운트용
        self.goal_times = [[] for _ in range(self.grid_config.num_agents)]  # 목표 간 이동 시간 저장
        self.wait_times = [[] for _ in range(self.grid_config.num_agents)]  # 각 골에서 wait한 시간 저장
        self.last_goal_step = [0] * self.grid_config.num_agents
        self.current_step = 0
    
        if seed is not None:
            self.grid.seed = seed
    
        if return_info:
            return self._obs(), self._get_infos()
        return self._obs()
    
    def step(self, action: list):
        assert len(action) == self.grid_config.num_agents
        rewards = []
        infos = [{} for _ in range(self.grid_config.num_agents)]
        terminated = [False] * self.grid_config.num_agents
        truncated = [False] * self.grid_config.num_agents
    
        # --- 전체 step 카운트 ---
        self.current_step += 1
    
        # 1. 목표 다 소진 or 대기 중이면 무조건 정지(stay)로 action 덮어쓰기
        for agent_idx in range(self.grid_config.num_agents):
            # 목표 다 소진
            if (
                self.goal_sequences is not None and 
                self.goal_indices[agent_idx] >= len(self.goal_sequences[agent_idx])
            ):
                action[agent_idx] = 0  # stay
                terminated[agent_idx] = True
                self.grid.is_active[agent_idx] = False
                continue
            
            idx = self.goal_indices[agent_idx]
            if self.wait_sequences is not None:
                if idx < len(self.wait_sequences[agent_idx]):
                    wait_steps = self.wait_sequences[agent_idx][idx]
                else:
                    wait_steps = 0
            else:
                wait_steps = self.wait_steps if self.wait_steps is not None else 0
    
            if 0 < self.wait_counters[agent_idx] <= wait_steps:
                action[agent_idx] = 0
    
        # 2. 실제 이동
        self.move_agents(action)
        self.update_was_on_goal()
    
        # 3. 리워드 및 상태 갱신
        for agent_idx in range(self.grid_config.num_agents):
            idx = self.goal_indices[agent_idx]
            # 목표 시퀀스 끝났으면 그냥 대기, 리워드는 0
            if (
                self.goal_sequences is not None and 
                idx >= len(self.goal_sequences[agent_idx])
            ):
                # 마지막 목표에 도달한 시점에서 남아있는 wait time도 정리(딱 한번만 실행)
                # 추가 기록 방지: goal_times/wait_times가 목표 개수와 맞지 않을 경우만 정리
                if len(self.goal_times[agent_idx]) < len(self.goal_sequences[agent_idx]):
                    move_time = self.current_step - self.last_goal_step[agent_idx] - 1
                    if move_time >= 0:
                        self.goal_times[agent_idx].append(move_time)
                    # 마지막 wait 기록(이미 wait했을 때만)
                    if len(self.wait_times[agent_idx]) < len(self.goal_sequences[agent_idx]):
                        self.wait_times[agent_idx].append(self.wait_counters[agent_idx])
                rewards.append(0.0)
                self.wait_counters[agent_idx] = 0
                terminated[agent_idx] = True
                self.grid.is_active[agent_idx] = False
                continue
            
            on_goal = self.grid.on_goal(agent_idx)
    
            if self.wait_sequences is not None:
                if idx < len(self.wait_sequences[agent_idx]):
                    wait_steps = self.wait_sequences[agent_idx][idx]
                else:
                    wait_steps = 0
            else:
                wait_steps = self.wait_steps if self.wait_steps is not None else 0
    
            if on_goal and self.grid.is_active[agent_idx]:
                if self.wait_counters[agent_idx] == 0:
                    # 처음 골 도착한 순간 (대기 시작)
                    move_time = self.current_step - self.last_goal_step[agent_idx] - 1
                    if move_time >= 0:
                        self.goal_times[agent_idx].append(move_time)
                    self.last_goal_step[agent_idx] = self.current_step - 1
                if self.wait_counters[agent_idx] < wait_steps:
                    self.wait_counters[agent_idx] += 1
                    rewards.append(1.0)
                else:
                    self.wait_times[agent_idx].append(self.wait_counters[agent_idx])
                    self.wait_counters[agent_idx] = 0
                    self.goal_indices[agent_idx] += 1
                    next_target = self._generate_new_target(agent_idx)
                    self.grid.finishes_xy[agent_idx] = next_target
                    rewards.append(1.0)
            elif on_goal:
                rewards.append(0.0)
            else:
                self.wait_counters[agent_idx] = 0
                rewards.append(0.0)
    
        # info에 통계 저장
        for agent_idx in range(self.grid_config.num_agents):
            infos[agent_idx]['is_active'] = self.grid.is_active[agent_idx]
            # 길이 맞추기(혹시 마지막에 하나 더 추가되는 것 방지)
            infos[agent_idx]['goal_times'] = self.goal_times[agent_idx][:len(self.goal_sequences[agent_idx])]
            infos[agent_idx]['wait_times'] = self.wait_times[agent_idx][:len(self.goal_sequences[agent_idx])]
    
        obs = self._obs()
        return obs, rewards, terminated, truncated, infos

class PogemaCoopFinish(Pogema):
    def __init__(self, grid_config=GridConfig(num_agents=2)):
        super().__init__(grid_config)
        self.num_agents = self.grid_config.num_agents
        self.is_multiagent = True

    def _initialize_grid(self):
        self.grid: Grid = Grid(grid_config=self.grid_config)

    def step(self, action: list):
        assert len(action) == self.grid_config.num_agents

        infos = [dict() for _ in range(self.grid_config.num_agents)]

        self.move_agents(action)
        self.update_was_on_goal()

        is_task_solved = all(self.was_on_goal)
        for agent_idx in range(self.grid_config.num_agents):
            infos[agent_idx]['is_active'] = self.grid.is_active[agent_idx]

        obs = self._obs()

        terminated = [is_task_solved] * self.grid_config.num_agents
        truncated = [False] * self.grid_config.num_agents
        rewards = [1.0 if is_task_solved else 0.0 for _ in range(self.grid_config.num_agents)]
        return obs, rewards, terminated, truncated, infos


def _make_pogema(grid_config, goal_sequences, wait_steps):

    if grid_config.on_target == 'restart':
        env = PogemaLifeLong(grid_config=grid_config,goal_sequences=goal_sequences, wait_steps=wait_steps)
    elif grid_config.on_target == 'nothing':
        env = PogemaCoopFinish(grid_config=grid_config)
    elif grid_config.on_target == 'finish':
        env = Pogema(grid_config=grid_config)
    else:
        raise KeyError(f'Unknown on_target option: {grid_config.on_target}')

    env = MultiTimeLimit(env, grid_config.max_episode_steps)

    if env.grid_config.persistent:
        env = PersistentWrapper(env)
    else:
        # adding metrics wrappers
        if grid_config.on_target == 'restart':
            env = LifeLongAverageThroughputMetric(env)
        elif grid_config.on_target == 'nothing':
            env = NonDisappearISRMetric(env)
            env = NonDisappearCSRMetric(env)
            env = NonDisappearEpLengthMetric(env)
            env = SumOfCostsAndMakespanMetric(env)
        elif grid_config.on_target == 'finish':
            env = ISRMetric(env)
            env = CSRMetric(env)
            env = EpLengthMetric(env)
        else:
            raise KeyError(f'Unknown on_target option: {grid_config.on_target}')

    return env
