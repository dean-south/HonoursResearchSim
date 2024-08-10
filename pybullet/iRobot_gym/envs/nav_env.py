import math
from math import sin, cos, pi
import gym
from gym import spaces
import pybullet as p
import random
import numpy as np
from .scenarios import SimpleNavScenario
from .camera import CameraController


class SimpleNavEnv(gym.Env):

    def __init__(self, scenario):
        self._scenario = scenario
        self._initialized = False
        self._time = 0.0

        self.action_space = spaces.Discrete(2)

        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=float)

        self.path = []

        self.observation = []
        if self._scenario.agent.task_name == 'reward_rapprochement_goal':
            self._RewardFunction = RewardRapprochementGoal(
                self._scenario.agent.task_param)
        elif self._scenario.agent.task_name == 'reward_binary_goal_based':
            self._RewardFunction = RewardBinaryGoalBased(
                self._scenario.agent.task_param)
        elif self._scenario.agent.task_name == 'no_reward':
            self._RewardFunction = NoReward(
                self._scenario.agent.task_param)
        elif self._scenario.agent.task_name == 'reward_displacement':
            self._RewardFunction = RewardDisplacement(
                self._scenario.agent.task_param)
        elif self._scenario.agent.task_name == 'reward_straight_navigation':
            self._RewardFunction = RewardNavStr(
                self._scenario.agent.task_param, self._scenario.world.state(), self._scenario.agent.id, self)
            
            self.get_start_pose = self.random_start_pose

            self.get_path = self.get_nav_bend_path
            
            
        # Your existing initialization code here
        
        self.camera_controller = CameraController()

    @ property
    def scenario(self):
        return self._scenario

    def step(self, action):
        state = self._scenario.world.state()
        # self.observation, _ = self._scenario.agent.step(action=action)
        self.observation = self.get_world_observation()
        done = self._RewardFunction.done(self._scenario.agent.id, state, self.path[0])
        reward = self._RewardFunction.reward(
            self._scenario.agent.id, state, self.path[0])
        self._time = self._scenario.world.update(
            agent_id=self._scenario.agent.id)
        self.update_camera()

        current_cell = self.pos2cell(*state[self._scenario.agent.id]['pose'][:2])

        if current_cell == self.path[0]:
            self.path.pop(0)

        return self.observation, reward, done, state[self._scenario.agent.id]

    def reset(self):
        if not self._initialized:
            self._scenario.world.init()
            self._initialized = True
        else:
            self._scenario.world.reset()
        # obs = self._scenario.agent.reset(
        #     self._scenario.world._get_starting_position(self._scenario.agent))
        self._scenario.agent.reset(self.get_start_pose())
        self._scenario.world.update(agent_id=self._scenario.agent.id)
        self._RewardFunction.reset()

        self.path = self.get_path()


        
        # Reset camera to initial position
        self.camera_controller = CameraController()
        p.resetDebugVisualizerCamera(
            self.camera_controller.camera_distance,
            self.camera_controller.camera_yaw,
            self.camera_controller.camera_pitch,
            self.camera_controller.camera_target_position
        )

    def render(self, **kwargs):
        return self._scenario.world.render(agent_id=self._scenario.agent.id, **kwargs)

    def seed(self, seed=None):
        self._scenario.world.seed(seed)

    def get_laserranges(self):
        return self._scenario.agent._vehicle.observe()
    
    def update_camera(self):
        self.camera_controller.update()

    def get_world_observation(self):

        state = self._scenario.world.state()[self._scenario.agent.id] 
        
        if len(self.path):
            obs = [*self.normalise_pos(*state['pose'][:2]),
                    self.normalise_theta(state['pose'][-1]),
                    *self.normalise_v(*state['velocity'][:2]),
                    self.normalise_v_theta(state['velocity'][-1]),
                    *self.one_hot_cell(self.path[0])]
        else:
            obs = [*self.normalise_pos(*state['pose'][:2]),
                    self.normalise_theta(state['pose'][-1]),
                    *self.normalise_v(*state['velocity'][:2]),
                    self.normalise_v_theta(state['velocity'][-1]),
                    *np.zeros(32)]

        return obs

    def get_nav_bend_path(self):
        curr_cell = self.get_start_cell()

        cell_path = []

        for i in range(2):

            direction = random.randint(0,3)

            d_x = int(sin(direction * pi/2))
            d_y = int(cos(direction * pi/2))

            dist = random.randint(1,15)

            des_cell = [int(curr_cell[0] + d_x*dist), int(curr_cell[1] + d_y*dist)]

            if des_cell[0] < 0:
                des_cell[0] = 0
            elif des_cell[1] < 0:
                des_cell[1] = 0
            elif des_cell[0] > 15:
                des_cell[0] = 15
            elif des_cell[1] > 15:
                des_cell[1] = 15

            cell_path.append(des_cell)

            curr_cell = des_cell

        return cell_path

    def get_start_cell(self):
        x,y,z = self._scenario.world._get_starting_position(self._scenario.agent)[0]

        cell_x = 0
        cell_y = 0

        for i in range(16):
            if x >= -8 + i and x < -8 + (i+1):
                cell_x = i

            if y >= -8 + i and y < -8 + (i+1):
                cell_y = i

        return [cell_x, cell_y]

    @staticmethod
    def random_start_pose():

        cell_x = random.randint(0,15)
        cell_y = random.randint(0,15)

        start_pos = [i for i in range(-8,8,1)]
        directions = [[0,0,1,1], [0,0,0,1], [0,0,-1,1], [0,0,0,-1]]

        start_x = start_pos[cell_x] + random.uniform(0.168, 0.832)
        start_y = start_pos[cell_y] + random.uniform(0.168, 0.832)

        d = random.randint(0,3)
        orientation = directions[d]

        return [[start_x, start_y, 0], orientation]
    
    def one_hot_cell(self, cell):
        cell_x = np.zeros(16)
        cell_y = np.zeros(16)

        cell_x[cell[0]] = 1
        cell_y[cell[1]] = 1

        return [*cell_x, *cell_y]
    
    @staticmethod
    def normalise_pos(x,y):
        x = (x + 8)/(16)
        y = (y + 8)/(16)

        return [x,y]
    
    @staticmethod
    def normalise_theta(theta):
        theta = (theta + pi)/(2*pi) 

        return theta

    @staticmethod
    def normalise_v(v_x,v_y):
        v_x = (v_x + 100)/(200)
        v_y = (v_y + 100)/(200)

        return [v_x, v_y]

    @staticmethod
    def normalise_v_theta(v_theta):
        return (v_theta + 25)/(50)

    @staticmethod
    def pos2cell(x, y):
        cell_x = 0
        cell_y = 0

        for i in range(16):
            if x >= -8 + i and x < -8 + (i+1):
                cell_x = i

            if y >= -8 + i and y < -8 + (i+1):
                cell_y = i

        return [cell_x, cell_y]

class RewardNavStr:
    "Reward for training robot to go straight"

    def __init__(self, param, state, agent_id, env) -> None:
        
        self._time_limit = param['time_limit']
        self.prev_state = state[agent_id]
        self.env = env
        self.agent_id = agent_id

    def reward(self, _agent_id, _state, desired_cell):
        reward = 0

        state = _state[_agent_id]

        x,y = state['pose'][:2]
        prev_x, prev_y = self.prev_state['pose'][:2]

        current_cell = self.pos2cell(x,y)
        prev_cell = self.pos2cell(prev_x, prev_y)

        if current_cell == desired_cell:
            reward += 200
        elif prev_cell != current_cell and \
            np.linalg.norm([prev_cell, desired_cell]) > np.linalg.norm([current_cell, desired_cell]):
            reward += 10
        elif prev_cell != current_cell and \
            np.linalg.norm([prev_cell, desired_cell]) < np.linalg.norm([current_cell, desired_cell]):
            reward += -100
        else:
            reward -= 1

        self.prev_state = state

        return reward

    def done(self, _agent_id, _state, desired_cell):
        done = 0

        state = _state[_agent_id]
        x,y = state['pose'][:2]
        current_cell = self.pos2cell(x,y)

        prev_x, prev_y = self.prev_state['pose'][:2]
        prev_cell = self.pos2cell(prev_x, prev_y)

        if prev_cell != current_cell and \
            np.linalg.norm([prev_cell, desired_cell]) < np.linalg.norm([current_cell, desired_cell]):
            done = 1
        
        laserRanges = self.env.get_laserranges()
        for r in laserRanges:
            if r < 0.2 and r > 0.1:                           
                done = True
                if self._verbose:
                    print("collision detected")
                rew = -100

        return done

    def reset(self):
        self.env._scenario.world.state()[self.agent_id]

    def pos2cell(self, x, y):
        
        x = self.unnormalize(x)
        y = self.unnormalize(y)

        cell_x = 0
        cell_y = 0

        for i in range(16):
            if x >= -8 + i and x < -8 + (i+1):
                cell_x = i

            if y >= -8 + i and y < -8 + (i+1):
                cell_y = i

        return [cell_x, cell_y]


class NoReward:
    """ No reward"""

    def __init__(self, param):
        self._time_limit = param['time_limit']
        self._goal_size_detection = param['goal_size_detection']

    def reward(self, _agent_id, _state, _action):
        return 0

    def done(self, agent_id, state):
        agent_state = state[agent_id]
        if agent_state['dist_obj'] < self._goal_size_detection:
            return True
        return self._time_limit < agent_state['time'] and self._time_limit > 0

    def reset(self):
        pass


class RewardBinaryGoalBased:
    """ Reward of 1 is given when close enough to the goal. """

    def __init__(self, param):
        self._time_limit = param['time_limit']
        self._goal_size_detection = param['goal_size_detection']

    def reward(self, agent_id, state, _action):
        agent_state = state[agent_id]
        if agent_state['dist_obj'] < self._goal_size_detection:
            return 1
        return 0

    def done(self, agent_id, state):
        agent_state = state[agent_id]
        if agent_state['dist_obj'] < self._goal_size_detection:
            return True
        return self._time_limit < agent_state['time'] and self._time_limit > 0

    def reset(self):
        pass


class RewardDisplacement:
    """ Reward = distance to previous position"""

    def __init__(self, param):
        self._time_limit = param['time_limit']
        self._goal_size_detection = param['goal_size_detection']
        self._last_stored_pos = None

    def reward(self, agent_id, state, action):
        agent_state = state[agent_id]
        pose = agent_state['pose']
        if self._last_stored_pos is None:
            self._last_stored_pos = pose
        reward = math.sqrt((pose[0] - self._last_stored_pos[0])**2 +
                           (pose[1] - self._last_stored_pos[1])**2)
        self._last_stored_pos = pose
        return reward

    def done(self, agent_id, state):
        agent_state = state[agent_id]
        if agent_state['dist_obj'] < self._goal_size_detection:
            return True
        return self._time_limit < agent_state['time'] and self._time_limit > 0

    def reset(self):
        self._last_stored_pos = None


class RewardRapprochementGoal:
    """ Reward when you reduce the distance to the goal"""

    def __init__(self, param):
        self._time_limit = param['time_limit']
        self._goal_size_detection = param['goal_size_detection']
        self._last_stored_progress = None

    def reward(self, agent_id, state, _action):
        agent_state = state[agent_id]
        progress = agent_state['dist_obj']
        if self._last_stored_progress is None:
            self._last_stored_progress = progress
        reward = (-1)*(progress - self._last_stored_progress)
        self._last_stored_progress = progress
        return reward

    def done(self, agent_id, state):
        agent_state = state[agent_id]
        if agent_state['dist_obj'] < self._goal_size_detection:
            return True
        return (self._time_limit < agent_state['time'] and self._time_limit > 0)

    def reset(self):
        self._last_stored_progress = None
