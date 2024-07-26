import os
import sys
import time
import csv
import argparse
import gym
import pybullet as p
import numpy as np
import random
from math import sin, cos, pi

from iRobot_gym.envs import SimpleNavEnv
from controllers.follow_wall import FollowWallController
from controllers.forward import ForwardController
from controllers.rulebased import RuleBasedController
from controllers.braitenberg import BraitenbergController
from controllers.novelty_ctr import NoveltyController
from controllers.RL_ctr import Agent
from controllers.blank_ctr import BlankController




class SimEnv():
    """This is the main class that runs the PyBullet simulation daccording to the arguments.

    Attributes:
        _env: The actual environnment.
        _sleep_time: float, representing the sleep time between each step.
        _ctr: string, the name of the controller.
        _verbose: bool, activate verbose or not.
        _episodes: int, amount of training episodes.
        _model_name: str, name of model being trained
        _i: int, iterator for steps.
        _liste_position: list of tuples, values to save in the csv file.
    """

    def __init__(self):
        self._env = gym.make(args.env+str('-v0'))
        self._sleep_time = args.sleep_time
        self._ctr = args.ctr
        self._verbose = args.verbose
        self._episodes = args.episodes
        self._model_name = args.model_name
        self._i = -1
        self._liste_position = []
        self._env.reset()
        self.train = args.train
        self.test_mode = args.test_mode
        


        # initialize controllers
        if self._ctr == "forward":
            self._controller = ForwardController(
                self._env, verbose=self._verbose)
        elif self._ctr == "wall":
            self._controller = FollowWallController(
                self._env, verbose=self._verbose)
        elif self._ctr == "rule":
            self._controller = RuleBasedController(
                self._env, verbose=self._verbose)
        elif self._ctr == "braitenberg":
            self._controller = BraitenbergController(
                self._env, verbose=self._verbose)
        elif self._ctr == "novelty":
            self._controller = NoveltyController(
                self._env, args.file_name, verbose=self._verbose)
        elif self._ctr == "RL":
            self._controller = Agent(alpha=0.01, beta=0.01, input_dims=18, tau=0.05, env=self._env, 
                                     batch_size=100, layer1_size=400, layer2_size=300, n_actions=2,
                                     model_name=args.model_name)                
            
            if not self.test_mode:

                if args.load_model != '':
                    self._controller.load_models(args.load_model, args.model_version)

                path = os.path.join(os.getcwd() + '/saved_models', args.model_name)
                os.mkdir(path)


        elif self._ctr == "blank":
            self._controller = BlankController()
        else:
            print("\nNo controller named", self._ctr)
            sys.exit()

    def _movement(self, action, nbr=1):
        for _ in range(nbr):
            obs, rew, done, info = self._env.step(action)

            print(self._i, end='\r')
            self._i += 1

            if self._i > 0:
                rew = self._controller.get_reward()

                self._controller.is_command_complete()

                obs = self.get_state_command(info, self._controller.get_command(), 
                                         self._controller.get_next_command())
            
            
                if self._controller.get_next_command() == -1 or self._controller.wrong_cell:
                    self._controller.wrong_cell = False
                    done = True

            time.sleep(self._sleep_time)

        return obs, rew, done, info
    
    def get_state_command(self, state, command, next_command):
        state_command = np.zeros(self._controller.actor.input_dims)

        state_command[:2] = state['pose'][:2] # position x,y
        state_command[2] = state['pose'][-1] # orientation
        state_command[3:5] = state['velocity'][:2] # velocity x,y
        state_command[5] = state['velocity'][-1] #angular velocity

        com = np.zeros(6)
        next_com = np.zeros(6)

        com[command] = 1
        if next_command != -1:
            next_com[next_command] = 1

        state_command[6:12] = com
        state_command[12:18] = next_com

        return state_command
    
    def get_cell_path(self, start_cell):
        cell_path = [start_cell]

        orientation = 0

        for c in self._controller.commands:
            if c == 0:
                cell_path.append(cell_path[-1])
            elif c == 1:
                cell_path.append([cell_path[-1][0] + sin(pi/2*orientation), cell_path[-1][1] + cos(pi/2*orientation)])
            elif c == 2:
                orientation = (orientation + 3) % 4
            elif c == 3:
                orientation = (orientation + 1) % 4

        return cell_path


    def pose_to_cell(self, pos):
        x,y = pos

        cell_x = 0
        cell_y = 0

        for i in range(16):
            if x >= -8 + i and x < -8 + (i+1):
                cell_x = i

            if y >= -8 + i and y < -8 + (i+1):
                cell_y = i

        return [cell_x, cell_y]

    def random_start_pose(self):

        cell_x = random.randint(0,15)
        cell_y = random.randint(0,15)

        start_pos = [i for i in range(-8,8,1)]
        directions = [[0,0,1,1], [0,0,0,1], [0,0,-1,1], [0,0,0,-1]]

        start_x = start_pos[cell_x] + random.uniform(0.168, 0.832)
        start_y = start_pos[cell_y] + random.uniform(0.168, 0.832)

        d = random.randint(0,3)
        orientation = directions[d]

        return [start_x, start_y, 0], orientation


    def start(self):
        """Forward the simulation until its complete."""  

        best_score = -1000000
        score_history = []

        kill_sim = False

        for e in range(self._episodes):
            print(f"Starting Epoch {e}")

            self._done = 0
            score = 0
            self._i = -1
            self._env.reset()

            if self.train == 'stay':
                self._controller.commands = [0,0]
                start_pos, start_ori = self.random_start_pose()
                p.resetBasePositionAndOrientation(3, start_pos, start_ori)

            self._obs, self._rew, self._done, self._info = self._movement([0, 0])

            self._controller.current_cell = self.pose_to_cell(start_pos[:2])
            self._controller.cell_path = self.get_cell_path(self._controller.current_cell)

            self._controller.set_state_command(self.get_state_command(self._info, self._controller.get_command(),
                                                                      self._controller.get_next_command()))


            while not self._done:

                try:
                    action = self._controller.get_action()
                    state_command_, self._rew, self._done, self._info = self._movement(action)

                    self._controller.current_cell = self.pose_to_cell(state_command_[:2])

                    # laserRanges = self._env.get_laserranges()
                    # for r in laserRanges:
                    #     if r < 0.2 and r > 0.1:                           
                    #         p.resetBasePositionAndOrientation(3, [-7.5,-7.5, 0.0], [0, 0, 1, 1])
                    #         self._done = 1
                    #         if self._verbose:
                    #             print("collision detected")
                    #         self._rew = -2000

                    if not self.test_mode:
                        self._controller.remember(self._controller.state_command, action, self._rew, state_command_, self._done)
                        self._controller.learn()

                    score += self._rew
                    self._controller.set_state_command(state_command_)

                    # print(self._info['pose'])
                    self._controller.reset()

                    if self._i == 1500:
                        self._done = 1
                        break

                except KeyboardInterrupt:
                    print(' The simulation was forcibly stopped.')
                    kill_sim = True
                    break

            if kill_sim:
                break
            
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])

            if avg_score > best_score:
                best_score = avg_score
                if not self.test_mode:
                    self._controller.save_models(e)


            print(f"episode {e}, score {score}, average score {avg_score}")
            # p.resetBasePositionAndOrientation(3, [-9.3, -9.25, 0.0], [0, 0, 1, 1])

            
        print("training is completed")
        print("Number of steps:", self._i)
        print("Simulation time:", self._info['time'], "s\n")
        self._env.close()

    def save_score(self, scores):
        cwd = os.getcwd()
        data_path = os.path.join(cwd, 'Data')

        with open(data_path + f'/{self._model_name}') as file:
            writer = csv.writer(file)
            writer.writerow(scores)


    def create_graph(self):
        pass

    def save_result(self):
        """Save the simulation data in a csv file in the folder
        corresponding to the controller and name it accordingly.
        """
        base_path = os.path.dirname(os.path.abspath(__file__))
        path = f'{base_path}/../results/{args.env}/bullet_{args.ctr}_'
        i = 1
        if os.path.exists(path+str(i)+".csv"):
            while os.path.exists(path+str(i)+".csv"):
                i += 1
        with open(path+str(i)+".csv", 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["steps", "x", "y", "z", "roll", "pitch", "yaw",
                             "distance_to_obj", "laser"])
            writer.writerows(self._liste_position)


def main():
    if args.model_name == args.load_model:
        print(f'Error model name must be different to load model name, {args.model_name} == {args.load_model}')
        return
    sim_env = SimEnv()
    sim_env.start()
    if args.save_res:
        sim_env.save_result()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Launch pybullet simulation run.')
    parser.add_argument('--env', type=str, default="2018apec",
                        help='environnement: kitchen, maze_hard, race_track')
    parser.add_argument('--ctr', type=str, default="RL",
                        help='controller: wall, rule, braitenberg, novelty, RL')
    parser.add_argument('--sleep_time', type=float, default=0.001,
                        help='sleeping time between each step')
    parser.add_argument('--save_res', type=bool, default=False,
                        help='save the result in a csv file: True or False')
    parser.add_argument('--verbose', type=bool, default=False,
                        help='verbose for controller: True or False')
    parser.add_argument('--file_name', type=str,
                        default='NoveltyFitness/9/maze_nsfit9-gen38-p0', help='file name of the invidual to load if ctr=novelty')
    parser.add_argument('--episodes', type=int, default=1000000,
                        help='how many training episodes')
    parser.add_argument('--model_name', type=str, default='model',
                        help='name of the model being trained')
    parser.add_argument('--test_mode', type=bool, default=False,
                        help="set true or false for test mode")
    parser.add_argument('--load_model', type=str, default='',
                        help="give name of pre-trained model to load")
    parser.add_argument('--model_version', type=int, default=0,
                        help="if load_model isn't blank, state which version of the model you want to load")
    parser.add_argument('--train', type=str, default='stay',
                        help='state what you want the model to train')
    args = parser.parse_args()
    main()
