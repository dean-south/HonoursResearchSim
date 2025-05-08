import math
from math import sin, cos, pi, atan, sqrt, acos, asin
import gym
from gym import spaces
import pybullet as p
import random
import numpy as np
from numpy.linalg import norm
from numpy import dot
from .scenarios import SimpleNavScenario
from .camera import CameraController
from iRobot_gym.envs.training_env import set_constrained_env
from collections import deque



def one_hot_cell(self, cell):
    cell_x = np.zeros(self.maze_size)
    cell_y = np.zeros(self.maze_size)

    cell_x[cell[0]] = 1
    cell_y[cell[1]] = 1

    return [*cell_x, *cell_y]

fix_path_tasks = ['2018apec', 'training_env', 'straight_env', '2014japan', '2010robotic', 'zigzag']

class SimpleNavEnv(gym.Env):

    def __init__(self, scenario, render_mode='rgb_array'):
        if scenario.agent.task_name == 'cl_const' or scenario.agent.task_name == 'cl_move':
            self.maze_id = 0
            self.maze_size = 16
            self.wall_prob = 0.45
            self.show_walls = True
            self.alt_goal = False if scenario.agent.task_name == 'cl_const' else True
            self.maze = set_constrained_env(self.maze_size, self.wall_prob, self.show_walls)
        else:
            self.maze_size = 16

        super().__init__()
        self._scenario = scenario
        self._initialized = False
        self._time = 1
        self.Return = 0
        self.dist_tol = 0.1 #Make robot have 8 lasers decrease in reset
        self.render_mode = render_mode

        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=float)

        self.observation_space = spaces.Box(low=0, high=1, shape=(12,), dtype=float)

        self.path = []

        self.task_name = self._scenario.agent.task_name

        self.total_returns = deque(maxlen=10)
        self.total_times = deque(maxlen=10)

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
                self._scenario.agent.task_param, self)
        
            self.get_start_pose = self.random_start_pose
            self.get_path = self.get_nav_bend_path
        elif self._scenario.agent.task_name == 'carry_on':
            self._RewardFunction = RewardCarryOn(
                self._scenario.agent.task_param, self)

            self.get_start_pose = self.carry_on_start_pose
            self.get_path = self.get_carry_on_path
        elif self._scenario.agent.task_name == '2018apec':
            self._RewardFunction = RewardCarryOn(
                self._scenario.agent.task_param, self)
            self.get_start_pose = self.get_2018_apec_start_pose
            self.get_path = self.get_2018_apec_path
        elif self._scenario.agent.task_name == '2014japan':
            self._RewardFunction = RewardCarryOn(
                self._scenario.agent.task_param, self)
            self.get_start_pose = self.get_2018_apec_start_pose
            self.get_path = self.get_2014_japan_path
        elif self._scenario.agent.task_name == '2010robotic':
            self._RewardFunction = RewardCarryOn(
                self._scenario.agent.task_param, self)
            self.get_start_pose = self.get_2018_apec_start_pose
            self.get_path = self.get_2010_robotic_path
        elif self._scenario.agent.task_name == 'zigzag':
            self._RewardFunction = RewardCarryOn(
                self._scenario.agent.task_param, self)
            self.get_start_pose = self.get_2018_apec_start_pose
            self.get_path = self.get_zigzag_path
        elif self._scenario.agent.task_name == 'training_env':
            self._RewardFunction = RewardCarryOn(
                self._scenario.agent.task_param, self)
            self.get_start_pose = self.get_2018_apec_start_pose
            self.get_path = self.get_training_env_path
        elif self._scenario.agent.task_name == 'straight_env':
            self._RewardFunction = RewardCarryOn(
                self._scenario.agent.task_param, self)
            self.reverse_path = False
            self.get_start_pose = self.get_straight_start_pose
            self.get_path = self.get_straight_env_path
        elif any(self._scenario.agent.task_name == name for name in ['cl', 'cl_const', 'cl_move']):
            self._RewardFunction = RewardCarryOn(
                self._scenario.agent.task_param, self)
            self.get_start_pose = self.get_cl_start_pose
            self.get_path = self.get_cl_path
                    
        self.camera_controller = CameraController()


    @ property
    def scenario(self):
        return self._scenario

    def step(self, action):
        # self.observation, _ = self._scenario.agent.step(action=action)
        self._scenario.agent.step(action=action)
        state = self._scenario.world.state()
        self.observation = self.get_world_observation()

        reward = self._RewardFunction.reward(self._scenario.agent.id, state)

        self.Return += reward

        dist = self.get_pose(self.path, state[self._scenario.agent.id])[0]*self.maze_size*sqrt(2)
       
        # current_cell = self.pos2cell(*state[self._scenario.agent.id]['pose'][:2])
        done = self._RewardFunction.done(self._scenario.agent.id, state)
        if dist < self.dist_tol:
            self.path = np.delete(self.path,0,axis=0)
            if len(self.path) == 0:
                self.path = self.get_path()
                if any(self._scenario.agent.task_name == task for task in fix_path_tasks):
                    done = True

       
        self._time +=1
        self._scenario.world.update(
            agent_id=self._scenario.agent.id)
        self.update_camera()

        
        return self.observation, reward, done, state[self._scenario.agent.id]

    def reset(self, **kwards):
        if not self._initialized:
            self._scenario.world.init()
            self._initialized = True
            if not any(self._scenario.agent.task_name == task for task in fix_path_tasks):
                self._scenario.agent.reset(self.random_start_pose())
            else:
                self._scenario.agent.reset(self.get_start_pose())
        else:

            self.total_returns.append(self.Return/self._time)
            self.total_times.append(self._time)

            if self._scenario.agent.task_name == 'cl_const':

                # if np.mean(self.total_returns) > 0.6 and np.mean(self.total_times) > self._scenario.agent.task_param['time_limit']*0.8:

                #     print(f'Average reward per time:{np.mean(self.total_returns)}, Average Time per episode:{np.mean(self.total_times)}')

                #     self.total_returns = deque(maxlen=10)
                #     self.total_times = deque(maxlen=10)
                    
                #     self.wall_prob += 0.125 * self.show_walls if self.wall_prob < 0.5 else 0
                #     self.show_walls = True
                #     self.maze_size = 16

                #     print(f'Maze Size:{self.maze_size} Wall Probability:{self.wall_prob}, Show Walls:{self.show_walls}')

                #     p.removeBody(self.maze_id)
                #     self.maze = set_constrained_env(self.maze_size, self.wall_prob, self.show_walls)
                #     self.maze_id = p.loadURDF("pybullet/models/scenes/cl/cl.urdf")

                p.removeBody(self.maze_id)
                self.maze = set_constrained_env(self.maze_size, self.wall_prob, self.show_walls)
                self.maze_id = p.loadURDF("pybullet/models/scenes/cl/cl.urdf")
                
            self._scenario.world.reset()
            self._scenario.agent.reset(self.get_start_pose())
        
        self._scenario.world.update(agent_id=self._scenario.agent.id)
        self._RewardFunction.reset()

        self.path = self.get_path()

        self._time = 1
        self.Return = 0


        
        # Reset camera to initial position
        self.camera_controller = CameraController()
        p.resetDebugVisualizerCamera(
            self.camera_controller.camera_distance,
            self.camera_controller.camera_yaw,
            self.camera_controller.camera_pitch,
            self.camera_controller.camera_target_position
        )

        return np.array(self.get_world_observation())

    def render(self, **kwargs):
        return self._scenario.world.render(agent_id=self._scenario.agent.id, **kwargs)

    def seed(self, seed=None):
        self._scenario.world.seed(seed)

    def get_laserranges(self):
        return self._scenario.agent._vehicle.observe()
    
    def get_pose_env(self):
        return self._scenario.world.state()[self._scenario.agent.id]['pose']

    def get_pose(self, path, state):

        cell = path[0]

        pos = state['pose'][:2]
        theta = state['pose'][-1]

        target = cell + np.array([-(self.maze_size/2 - 0.5),-(self.maze_size/2 - 0.5)])

        pos_ = np.linalg.norm([pos - target])/(self.maze_size*sqrt(2))

        if (pos[0]-target[0] == 0):
            theta_ = pi/2 if target[1] > pos[1] else -pi/2
        elif (pos[1]-target[1] == 0):
            theta_ = 0 if target[0] > pos[0] else pi
        else:
            theta_ = atan((pos[1]-target[1])/(pos[0]-target[0]))

            if pos[0] > target[0]:
                theta_ += pi if pos[1] < target[1] else -pi

        phi = theta_ - theta

        return pos_, phi
    
    def update_camera(self):
        self.camera_controller.update()

    def get_world_observation(self):

        state = self._scenario.world.state()[self._scenario.agent.id] 
        laserRanges = self.get_laserranges()
        # laserRanges = laserRanges[range(0,len(laserRanges)//2,2)]

        pose = self.get_pose(self.path, state)
        
        obs = [ pose[0],
                self.normalise_theta(pose[1]),
                self.normalise_v(state['velocity'][0]),
                self.normalise_v_theta(state['velocity'][-1]),
                *laserRanges/0.5,
                # dist_to_obj,
                # angle_to_obj,
                ]

        return obs

    def get_nav_bend_path(self):
        curr_cell = self.get_start_cell()

        cell_path = []

        for i in range(2):

            direction = random.randint(0,3)

            d_x = int(sin(direction * pi/2))
            d_y = int(cos(direction * pi/2))

            dist = random.randint(1,self.maze_size - 1)

            des_cell = [int(curr_cell[0] + d_x*dist), int(curr_cell[1] + d_y*dist)]

            if des_cell[0] < 0:
                des_cell[0] = 0
            elif des_cell[1] < 0:
                des_cell[1] = 0
            elif des_cell[0] > self.maze_size - 1:
                des_cell[0] = self.maze_size - 1
            elif des_cell[1] > self.maze_size - 1:
                des_cell[1] = self.maze_size - 1

            cell_path.append(des_cell)

            curr_cell = des_cell

        return np.array(cell_path)

    def get_start_cell(self):
        x,y,z = self._scenario.world._get_starting_position(self._scenario.agent)[0]

        cell_x = 0
        cell_y = 0

        for i in range(self.maze_size):
            if x >= -self.maze_size//2 + i and x < -self.maze_size//2 + (i+1):
                cell_x = i

            if y >= -self.maze_size//2 + i and y < -self.maze_size//2 + (i+1):
                cell_y = i

        return [cell_x, cell_y]

    def random_start_pose(self):

        cell_x = random.randint(0,self.maze_size-1)
        cell_y = random.randint(0,self.maze_size-1)

        start_pos = [i for i in range(-self.maze_size//2,self.maze_size//2,1)]
        directions = [[0,0,1,1], [0,0,0,1], [0,0,-1,1], [0,0,0,-1]]

        start_x = start_pos[cell_x] + random.uniform(0.168, 0.832)
        start_y = start_pos[cell_y] + random.uniform(0.168, 0.832)

        d = random.randint(0,3)
        orientation = directions[d]

        return [[start_x, start_y, 0], orientation]

    def carry_on_start_pose(self):
        if self._RewardFunction.reset_pose:
            # print('poes')
            self._RewardFunction.reset_pose = False
            return self.random_start_pose()
        else:
            
            state = self._scenario.world.state()[self._scenario.agent.id]

            pos = state['pose'][:3]

            orientation = p.getQuaternionFromEuler(state['pose'][3:])

        return [pos, orientation]
    
    def get_carry_on_path(self):

        state = self._scenario.world.state()[self._scenario.agent.id] 
        curr_cell = self.pos2cell(*state['pose'][:2])

        cell_path = []

        direction = random.randint(0,3)

        d_x = int(sin(direction * pi/2))
        d_y = int(cos(direction * pi/2))

        dist = random.randint(1,self.maze_size-1)

        des_cell = [int(curr_cell[0] + d_x*dist), int(curr_cell[1] + d_y*dist)]

        if des_cell[0] < 0:
            des_cell[0] = 0
        elif des_cell[1] < 0:
            des_cell[1] = 0
        elif des_cell[0] > self.maze_size - 1:
            des_cell[0] = self.maze_size - 1
        elif des_cell[1] > self.maze_size - 1:
            des_cell[1] = self.maze_size - 1

        cell_path.append(des_cell)

        return np.array(cell_path)

    def get_2018_apec_start_pose(self):
        pos = [-7.5,-7.5,0]
        oritentation = p.getQuaternionFromEuler([0,0,pi/2])

        return [pos, oritentation]
    
    def get_2018_apec_path(self):

        cell_path = np.array([
            [0,15], [15,15], [15,13], [2,13], [2,1], [14,1],
            [14,3], [13,3], [13,2], [12,2], [12,3], [11,3],
            [11,2], [10,2], [10,5], [9,5], [9,4], [8,4],
            [8,3], [6,3], [6,4], [7,4], [7,5], [8,5],
            [8,6], [6,6], [6,7], [7,7]
        ])

        return cell_path
    
    def get_2014_japan_path(self):

        cell_path = np.array([
            [0,15],[1,15],[1,0],[2,0],[2,15],[15,15],
            [15,13],[14,13],[14,14],[4,14],[4,10],[3,10],
            [3,9],[8,9],[8,10],[9,10],[9,11],[7,11],
            [7,12],[12,12],[12,11],[11,11],[11,10],[10,10],
            [10,9],[9,9],[9,8],[8,8]
        ])

        return cell_path
    
    def get_2010_robotic_path(self):

        cell_path = np.array([
            [0,14],[6,14],[6,15],[14,15],[14,13],
            [15,13],[15,11],[14,11],[14,10],[15,10],
            [15,4],[13,4],[13,5],[12,5],[12,6],
            [11,6],[11,8],[12,8],[12,9],[10,9],
            [10,6],[9,6],[9,5],[7,5],[7,7]
        ])

        return cell_path
    
    def get_zigzag_path(self):

        cell_path = np.array([
            [0,1],[1,1],[1,0],[2,0],[2,1],[3,1],
            [3,0],[4,0],[4,1],[5,1],[5,0],[6,0],
            [6,1],[7,1],[7,0],[8,0],[8,1],[9,1],
            [9,0],[10,0],[10,1],[11,1],[11,0],[12,0],
            [12,1],[13,1],[13,0],[14,0],[14,1],[15,1],
            [15,2],[14,2],[14,3],[15,3],[15,4],[14,4],
            [14,5],[15,5],[15,6],[14,6],[14,7],[15,7],
            [15,8],[14,8],[14,9],[15,9],[15,10],[14,10],
            [14,11],[15,11],[15,12],[14,12],[14,13],[15,13],
            [15,14],[14,14],[14,15],[13,15],[13,14],[12,14],
            [12,15],[11,15],[11,14],[10,14],[10,15],[9,15],
            [9,14],[8,14],[8,15],[7,15],[7,14],[6,14],
            [6,15],[5,15],[5,14],[4,14],[4,15],[3,15],
            [3,14],[2,14],[2,15],[1,15],[1,14],[0,14],
            [0,13],[1,13],[1,12],[0,12],[0,11],[1,11],
            [1,10],[0,10],[0,9],[1,9],[1,8],[0,8],
            [0,7],[1,7],[1,6],[0,6],[0,5],[1,5],
            [1,4],[0,4],[0,3],[1,3],[1,2],[2,2],
            [2,3],[3,3],[3,2],[4,2],[4,3],[5,3],
            [5,2],[6,2],[6,3],[7,3],[7,2],[8,2],
            [8,3],[9,3],[9,2],[10,2],[10,3],[11,3],
            [11,2],[12,2],[12,3],[13,3],[13,4],[12,4],
            [12,5],[13,5],[13,6],[12,6],[12,7],[13,7],
            [13,8],[12,8],[12,9],[13,9],[13,10],[12,10],
            [12,11],[13,11],[13,12],[12,12],[12,13],[11,13],
            [11,12],[10,12],[10,13],[9,13],[9,12],[8,12],
            [8,13],[7,13],[7,12],[6,12],[6,13],[5,13],
            [5,12],[4,12],[4,13],[3,13],[3,12],[2,12],
            [2,11],[3,11],[3,10],[2,10],[2,9],[3,9],
            [3,8],[2,8],[2,7],[3,7],[3,6],[2,6],
            [2,5],[3,5],[3,4],[4,4],[4,5],[5,5],
            [5,4],[6,4],[6,5],[7,5],[7,4],[8,4],
            [8,5],[9,5],[9,4],[10,4],[10,5],[11,5],
            [11,6],[10,6],[10,7],[11,7],[11,8],[10,8],
            [10,9],[11,9],[11,10],[10,10],[10,11],[9,11],
            [9,10],[8,10],[8,11],[7,11],[7,10],[6,10],
            [6,11],[5,11],[5,10],[4,10],[4,9],[5,9],
            [5,8],[4,8],[4,7],[5,7],[5,6],[6,6], 
            [6,9],[9,9],[9,6],[7,6],[7,7]
        ])

        return cell_path

    def get_training_env_path(self):

        self.reverse_path = random.randint(0,1)

        cell_path = np.array([
            [0,0], [0,15], [1,15], [1,1], [2,1], [2,15], [3,15], [3,1], [4,1],
            [4,15], [5,15], [5,0], [4,0], [4,15], [5,15] ,[5,14], [6,14],
            [6,15], [7,15], [7,14], [8,14], [8,15], [9,15], [9,14], [10,14],
            [10,15], [11,15], [11,14], [12,14], [12,15], [13,15], [13,14], [14,14],
            [14,15], [15,15], [15,13], [14,13], [14,12], [13,12], [13,11], [12,11],
            [12,10], [11,10], [11,9], [10,9], [10,8], [9,8], [9,7], [8,7],
            [8,6], [7,6], [7,4], [8,4], [8,5], [9,5], [9,6], [10,6],
            [10,7], [11,7], [11,8], [12,8], [12,9], [13,9], [13,10], [14,10],
            [14,11], [15,11], [15,9], [14,9], [14,8], [13,8], [13,7],
            [12,7], [12,6], [11,6], [11,5], [10,5], [10,4], [9,4], [9,3],
            [8,3], [8,2], [7,2], [7,1], [14,1], [14,9], [15,9], [15,0], [0,0]
        ])

        if self.reverse_path:
            cell_path = np.flip(cell_path, axis=0)

        return cell_path[1:]
    
    def get_straight_start_pose(self):
        
        self.reverse_path = random.randint(0,1)
        
        pos = [-7.5,-7.5,0] if not self.reverse_path else [7.5,-7.5,0]
        
        directions = [[0,0,1,1], [0,0,0,1], [0,0,-1,1], [0,0,0,-1]]
        d = random.randint(0,3)
        orientation = directions[d]

        return [pos, orientation]

    def get_straight_env_path(self):
        cell_path = np.array([
            [0,0], [0,15], [1,15], [1,0],
            [2,0], [2,15], [3,15], [3,0],
            [4,0], [4,15], [5,15], [5,0],
            [6,0], [6,15], [7,15], [7,0],
            [8,0], [8,15], [9,15], [9,0],
            [10,0], [10,15], [11,15], [11,0],
            [12,0], [12,15], [13,15], [13,0],
            [14,0], [14,15], [15,15], [15,0],
        ])

        if self.reverse_path:
            cell_path = np.flip(cell_path, axis=0)

        return cell_path[1:]
    
    def get_cl_start_pose(self):

        if self.show_walls:
            maze_walls = np.flip(self.maze,0)
            valid_start = False
            
            while not valid_start:

                i = random.randint(0,self.maze_size*2-2)
                j = random.randint(0,self.maze_size*2-2)

                i -= i%2
                j -= j%2

                if i > 0:
                    valid_start = not maze_walls[i-1,j]
                    if valid_start:
                        break
                if i < self.maze_size*2-2:
                    valid_start = not maze_walls[i+1,j]
                    if valid_start:
                        break
                if j > 0:
                    valid_start = not maze_walls[i,j-1]
                    if valid_start:
                        break
                if j < self.maze_size*2-2:
                    valid_start = not maze_walls[i,j+1]
                    if valid_start:
                        break
            
            x = -((self.maze_size-1)//2) + j//2
            y = (self.maze_size-1)//2 - i//2

            x = x - 0.5
            # y = y - np.sign(y)*0.5 if y != 0 else y + 0.5
            y += 0.5

            pos = [x,y,0]

        else:
            pos = [0.5,0.5,0]

        directions = [[0,0,1,1], [0,0,0,1], [0,0,-1,1], [0,0,0,-1]]
        d = random.randint(0,3)
        orientation = directions[d]

        return [pos, orientation]



    def get_cl_path(self):
        state = self._scenario.world.state()[self._scenario.agent.id] 
        curr_cell = self.pos2cell(*state['pose'][:2])

        cell_path = []

        while True:

            direction = random.randint(0,3)

            d_x = int(sin(direction * pi/2))
            d_y = int(cos(direction * pi/2))

            dist = random.randint(1,self.maze_size-1)

            des_cell = [int(curr_cell[0] + d_x*dist), int(curr_cell[1] + d_y*dist)]


            if des_cell[0] < 0:
                des_cell[0] = 0
            elif des_cell[0] > self.maze_size-1:
                des_cell[0] = self.maze_size-1
                
            if des_cell[1] < 0:
                des_cell[1] = 0
            
            elif des_cell[1] > self.maze_size-1:
                des_cell[1] = self.maze_size-1

            if self.show_walls:
                if abs(d_x) > 0:
                    for x in range(curr_cell[0], des_cell[0], d_x):
                        if self.maze[2*des_cell[1], 2*x + d_x]:
                            des_cell[0] = random.randint(min(curr_cell[0]+d_x,x), max(curr_cell[0]+d_x,x))
                            break
                elif abs(d_y) > 0:
                    for y in range(curr_cell[1], des_cell[1], d_y):
                        if self.maze[2*y + d_y, 2*des_cell[0]]:
                            des_cell[1] = random.randint(min(curr_cell[1]+d_y,y), max(curr_cell[1]+d_y,y)) 
                            break

            if self.alt_goal:
                des_cell += [random.random()/2.5 - 0.2, random.random()/2.5 - 0.2]

            if des_cell != curr_cell:
                break

        cell_path.append(des_cell)


        return np.array(cell_path)
    
    def normalise_pos(self,x,y):
        x = (x + self.maze_size//2)/(self.maze_size)
        y = (y + self.maze_size//2)/(self.maze_size)

        return [x,y]
    
    @staticmethod
    def normalise_theta(theta):
        theta = (theta + pi)/(2*pi) 

        return theta

    @staticmethod
    def normalise_v(v_x):
        v_x = ((v_x + 0.62)/(1.24))  

        return v_x

    @staticmethod
    def normalise_v_theta(v_theta):
        return (v_theta + 5)/(10)

    def pos2cell(self, x, y):
        cell_x = 0
        cell_y = 0

        for i in range(self.maze_size):
            if x >= -self.maze_size//2 + i and x < -self.maze_size//2 + (i+1):
                cell_x = i

            if y >= -self.maze_size//2 + i and y < -self.maze_size//2 + (i+1):
                cell_y = i

        return [cell_x, cell_y]
    
    def robot_collision(self):
        laserRanges = self.get_laserranges()

        if round(np.min(laserRanges),2) <= 0.2 and round(np.min(laserRanges),2) > 0.14:
            return True
        
        return False
    
    def get_obj_dist(self):
        laserRanges = self.get_laserranges()
        laserRanges = laserRanges[range(0,len(laserRanges)//2,2)]

        return 0.5 - np.min(laserRanges)

    def get_laser_obs_space(self):    
        laserRanges = self.get_laserranges()
        laserRanges = laserRanges[range(0,len(laserRanges)//2,2)]

        dist = (0.5 - np.min(laserRanges))/(0.3)
        angle = np.argmin(laserRanges)/10 if dist > 0 else -1

        return dist, angle

    
class RewardCarryOn:
    "End episode when you reach goal cell. Start next episode immidiately"

    def __init__(self, param, env) -> None:
        
        self._time_limit = param['time_limit']
        self.env = env
        self.reset_pose = False

    def reward(self, _agent_id, _state):
        state = _state[_agent_id]

        curr_pos = state['pose'][:2]

        dist, phi = self.env.get_pose(self.env.path, state)

        v_x = state['velocity'][0]

        # reward = - distance to goal - relative orientation to goal + forward velocity - wall proximity
        reward = - 1*abs(phi)/pi + 1*(self.env.normalise_v(v_x)*2-1) - 1*self.env.get_obj_dist()/(0.3)

        # if (len(self.env.path) and sum(current_cell == self.env.path[0])>1) or not len(self.env.path):
        #     reward = 150
        if dist < self.env.dist_tol/(self.env.maze_size*sqrt(2)):
            reward += 200

        elif self.env.robot_collision():
            reward += -300

                
        return reward

    def done(self, _agent_id, _state):
        done = False

        state = _state[_agent_id]
        x,y = state['pose'][:2]
        current_cell = self.pos2cell(x,y)

        # if sum(current_cell == self.env.path[0])>1:
        #     done = True
        # elif prev_cell != current_cell and \
        #     np.linalg.norm([prev_cell, self.env.path[0]]) < np.linalg.norm([current_cell, self.env.path[0]]):
        #     done = True
        #     self.reset_pose = True

        #     # print("went to the wrong cell")
        if self.env._time % self._time_limit == 0:
            self.reset_pose = True
            done = True
            # print('time ran out')
        elif self.env.robot_collision():                        
            done = True
            self.reset_pose = True


        return done
    
    def reset(self):
        # self.prev_state = self.env._scenario.world.state()[self.env._scenario.agent.id]
        pass


    def pos2cell(self, x, y):

        cell_x = 0
        cell_y = 0

        for i in range(self.env.maze_size):
            if x >= -self.env.maze_size//2 + i and x < -self.env.maze_size//2 + (i+1):
                cell_x = i

            if y >= -self.env.maze_size//2 + i and y < -self.env.maze_size//2 + (i+1):
                cell_y = i

        return [cell_x, cell_y]


class RewardNavStr:
    "Reward for training robot to go straight"

    def __init__(self, param, env) -> None:
        
        self._time_limit = param['time_limit']
        self.prev_state = None
        self.env = env

    def reward(self, _agent_id, _state):
        if self.prev_state is None:
            self.prev_state = self.env._scenario.world.state()[self.env._scenario.agent.id]

        reward = -1

        state = _state[_agent_id]

        x,y = state['pose'][:2]
        prev_x, prev_y = self.prev_state['pose'][:2]

        current_cell = self.pos2cell(x,y)
        prev_cell = self.pos2cell(prev_x, prev_y)

        if (len(self.env.path) and current_cell == self.env.path[0]) or not len(self.env.path):
            reward += 20
        elif len(self.env.path) and prev_cell != current_cell and \
            np.linalg.norm([prev_cell, self.env.path[0]]) > np.linalg.norm([current_cell, self.env.path[0]]):
            reward = 10
        elif len(self.env.path) and prev_cell != current_cell and \
            np.linalg.norm([prev_cell, self.env.path[0]]) < np.linalg.norm([current_cell, self.env.path[0]]):
            reward = -10
        else:
            laserRanges = self.env.get_laserranges()
            for r in laserRanges:
                if r < 0.19 and r > 0.14:                           
                    reward = -10
                    break
            
        return reward

    def done(self, _agent_id, _state):
        if self.prev_state is None:
            self.prev_state = self.env._scenario.world.state()[self.env._scenario.agent.id]

        done = False

        state = _state[_agent_id]
        x,y = state['pose'][:2]
        current_cell = self.pos2cell(x,y)

        prev_x, prev_y = self.prev_state['pose'][:2]
        prev_cell = self.pos2cell(prev_x, prev_y)

        self.prev_state = state

        if current_cell == self.env.path[0]:
            done = True
        elif prev_cell != current_cell and \
            np.linalg.norm([prev_cell, self.env.path[0]]) < np.linalg.norm([current_cell, self.env.path[0]]):
            done = True
            print("went to the wrong cell")

        else:
        
            laserRanges = self.env.get_laserranges()
            for r in laserRanges:
                if r < 0.19 and r > 0.14:                           
                    done = True
                    # print(f'crashed {r=}')
                    break

        return done

    def reset(self):
        self.prev_state = None

    def pos2cell(self, x, y):

        cell_x = 0
        cell_y = 0

        for i in range(self.maze_size):
            if x >= -self.maze_size//2 + i and x < -self.maze_size//2 + (i+1):
                cell_x = i

            if y >= -self.maze_size//2 + i and y < -self.maze_size//2 + (i+1):
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
