import os
import sys
import time
import csv
import argparse
import gym
import pybullet as p
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import wandb
import torch as th
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import TD3, HerReplayBuffer, PPO, SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.td3.policies import MlpPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.td3.policies import MlpPolicy
from stable_baselines3.common.callbacks import CheckpointCallback
from math import sin, cos, pi
from sb3_contrib import RecurrentPPO, TQC


from iRobot_gym.envs import SimpleNavEnv
from controllers.follow_wall import FollowWallController
from controllers.forward import ForwardController
from controllers.rulebased import RuleBasedController
from controllers.braitenberg import BraitenbergController
from controllers.novelty_ctr import NoveltyController
from controllers.RL_ctr import Agent
from controllers.blank_ctr import BlankController


class CustomMlpPolicy(MlpPolicy):
                def _build_mlp_extractor(self):
                    super()._build_mlp_extractor()
                    
                    # Modify the actor network
                    last_layer_size = self.actor.latent_pi.shape[1]
                    self.actor.mu = th.nn.Sequential(
                        th.nn.Linear(last_layer_size, self.action_space.shape[0]),
                        th.nn.Tanh()  # Add Tanh activation function
                    )

def linear_schedule(initial_value):
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
    the current learning rate depending on remaining progress
    (starts with initial_value and decreases to 0).
    """
    def schedule(progress_remaining):
        return progress_remaining * initial_value
    return schedule
            
def exponential_schedule(initial_value, decay_rate=0.99):
    def schedule(progress_remaining):
        return initial_value * (decay_rate ** (1 - progress_remaining))
    return schedule

load_func = {'sb3':TD3.load, 
            'ppo':PPO.load,
            'sac':SAC.load,
            'recppo': RecurrentPPO.load,
            'tqc':TQC.load}


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
        self._env = gym.make(args.env+str('-v0'), render_mode="rgb_array")
        self._sleep_time = args.sleep_time
        self._ctr = args.ctr
        self._verbose = args.verbose
        self._episodes = args.episodes
        self._model_name = args.model_name
        self._i = -1
        self._liste_position = []
        # self._env.reset()
        self.train = args.train
        self.test_mode = args.test_mode
        

        sb3_models = ['sb3', "sac", 'ppo', 'recppo', 'tqc']


        # initialize controllers
        if self._ctr == "RL":
            noise = 0.1 if not self.test_mode else 0
            self._controller = Agent(alpha=0.001, beta=0.001, input_dims=38, tau=0.001, env=self._env, 
                                     batch_size=1024, layer1_size=128, layer2_size=128, n_actions=2,
                                     model_name=self._model_name, update_actor_interval=10, noise=noise, warmup=1024)                
            
            if not self.test_mode:

                if args.load_model != '':
                    self._controller.load_models(args.load_model, args.model_version)

                path = os.path.join(os.getcwd() + '/pybullet/saved_models', args.model_name)
                os.mkdir(path)
        elif self._ctr == "sb3":
            n_actions = self._env.action_space.shape[-1]
            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.25 * np.ones(n_actions))

            if not self.test_mode:
                config = {
                    "policy_type": "MlpPolicy",
                    "total_timesteps": 10000000,
                    "env_name": "Blank-v0",
                }

                self.run = wandb.init(
                    project="Go slow",
                    config=config,
                    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                    save_code=True,  # optional
                )

            
            
           
            initial_learning_rate = 0.001


            self.model = TD3(
                'MlpPolicy',# CustomMlpPolicy ,
                self._env, 
                action_noise=action_noise, 
                verbose=1, 
                tensorboard_log=f"runs/{self._model_name}",
                tau=0.005,
                batch_size=256,
                policy_delay=10,
                gamma=0.99,
                learning_rate=exponential_schedule(initial_learning_rate, decay_rate=0.5),
                )
        elif self._ctr == 'ppo':

            if not self.test_mode:
                config = {
                    "policy_type": "MlpPolicy",
                    "total_timesteps": 10000000,
                    "env_name": "Empty-v0",
                }

                self.run = wandb.init(
                        project="PPO Maze",
                        config=config,
                        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                        save_code=True,  # optional
                    )
                
            initial_learning_rate = 0.0003
                
            self.model = PPO(
                'MlpPolicy', # CustomMlpPolicy,
                self._env,
                verbose=1,
                tensorboard_log=f"runs/{self._model_name}",
                learning_rate=exponential_schedule(initial_learning_rate, decay_rate=0.5)
            )
        
        elif self._ctr == 'sac':
            if not self.test_mode:
                config = {
                    "policy_type": "MlpPolicy",
                    "total_timesteps": 1000000,
                    "env_name": "Empty-v0",
                }

                self.run = wandb.init(
                        project="SAC Maze",
                        config=config,
                        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                        save_code=True,  # optional
                    )
                
            self.model = SAC(
                'MlpPolicy', # CustomMlpPolicy,
                self._env,
                verbose=1,
                tensorboard_log=f"runs/{self._model_name}"
            )
                

        elif self._ctr == 'recppo':
            if not self.test_mode:
                config = {
                    "policy_type": "MlpLstmPolicy",
                    "total_timesteps": 10000000,
                    "env_name": "Empty-v0",
                }

                self.run = wandb.init(
                        project="PPO Maze",
                        config=config,
                        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                        save_code=True,  # optional
                    )

            initial_learning_rate = 0.0003


            self.model = RecurrentPPO(
                'MlpLstmPolicy', # CustomMlpPolicy,
                self._env,
                verbose=1,
                tensorboard_log=f"runs/{self._model_name}",
                learning_rate=exponential_schedule(initial_learning_rate, decay_rate=0.5),
            )

        elif self._ctr == 'tqc':
            if not self.test_mode:
                config = {
                    "policy_type": "MlpPolicy",
                    "total_timesteps": 1000000,
                    "env_name": "Empty-v0",
                }

                self.run = wandb.init(
                        project="SAC Maze",
                        config=config,
                        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                        save_code=True,  # optional
                    )
                
            self.model = TQC(
                'MlpPolicy', # CustomMlpPolicy,
                self._env,
                verbose=1,
                tensorboard_log=f"runs/{self._model_name}"
            )


        else:
            print("\nNo controller named", self._ctr)
            sys.exit()

        if args.load_model is not None and any(self._ctr == ctr for ctr in sb3_models):
            print("Loading Model")
            self.model = load_func[self._ctr](f'models/{args.load_model}/model',env=self._env, verbose=1)

    def _movement(self, action):
    
        state, rew, done, info = self._env.step(action)

        # print(self._i, end='\r')
        self._i += 1

        # time.sleep(self._sleep_time)

        return state, rew, done, info


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
    
    def start(self):
        """Forward the simulation until its complete."""  

        best_score = -10000000
        score_history = []
        avg_score_history = []
        avg_reward_history = []

        kill_sim = False

        for e in range(self._episodes if self.test_mode else 1):
            self._done = 0
            score = 0
            self._i = -1
            obs = self._env.reset()      

            pose = self._env.get_pose()

            if self.test_mode:
                print(f'Starting Episode {e}, starting cell: {self.pose_to_cell(pose[:2])}, goal cell: {self._env.path}, ')

            while not self._done:

                try:
                    if self._ctr == "RL":
                        action = self._controller.get_action()

                        state_, self._rew, self._done, self._info = self._movement(action)
                    
                        if not self.test_mode:
                            self._controller.remember(obs, action, self._rew, state_, self._done)
                            self._controller.learn()

                      
                        score += self._rew
                        obs = state_

                        # print(self._info['pose'])
                        self._controller.reset()
                    elif self._ctr == 'sb3':

                        if not self.test_mode:
                            self.model.learn(total_timesteps=5000000, log_interval=10, 
                                    callback=WandbCallback(
                                        gradient_save_freq=10,
                                        model_save_freq=5000,
                                        model_save_path=f"models/{self._model_name}",
                                        verbose=2,
                                    ),
                                )
                            self.run.finish()
                        else:
                            # print(state[:6])
                            action, _ = self.model.predict(obs)

                            # action = [1,1]

                            # print(f'{action=} {action[0]=}')

                            obs, rew, self._done, info = self._movement(action)
                    
                    elif self._ctr == 'ppo' or self._ctr == 'sac' or self._ctr == 'recppo' or self._ctr == 'tqc':
                        if not self.test_mode:

                            # Create the wandb callback
                            wandb_callback = WandbCallback(
                                gradient_save_freq=10,
                                model_save_freq=5000,
                                model_save_path=f"models/{self.run.id}",
                                verbose=2,
                            )

                            # Create a checkpoint callback to save the model
                            checkpoint_callback = CheckpointCallback(
                                save_freq=10000,  # Save every 10000 steps
                                save_path=f"./models/{self.run.id}/",
                                name_prefix="rl_model"
                            )


                            self.model.learn(total_timesteps=10000000, log_interval=10, 
                                    callback=wandb_callback
                                )
                            self.run.finish()
                        else:
                            # print(state[:6])
                            action, _ = self.model.predict(obs)

                            # print(f'{obs=} \n {action=}')

                            action = [1,1]

                            # print(f'{action=} {action[0]=}')

                            obs, rew, self._done, info = self._movement(action)
                       

                except KeyboardInterrupt:
                    print(' The simulation was forcibly stopped.')
                    if not self.test_mode:
                        self.model.save(f"models/{self.run.id}/model")
                    kill_sim = True
                    break

            if kill_sim:
                break
            
            # score_history.append(score)
            # avg_score = np.mean(score_history[-100:])
            # avg_score_history.append(avg_score)
            # avg_reward_history.append(score/self._i)

            # self.save_scores(score_history, avg_score_history, avg_reward_history)
            # self.create_graph()

            # if avg_score > best_score and self._ctr == "RL":
            #     best_score = avg_score
            #     if not self.test_mode:
            #         self._controller.save_models(e)
            # elif e % 500 == 0 and not self.test_mode:
            #     self._controller.save_models(e)
            # p.resetBasePositionAndOrientation(3, [-9.3, -9.25, 0.0], [0, 0, 1, 1])

        self._env.close()

    def save_scores(self, scores, avg_scores, avg_rewards):
        cwd = os.getcwd()
        data_path = os.path.join(cwd, 'pybullet/Data')

        episodes = len(scores)

        data = [[e, scores[e], avg_scores[e], avg_rewards[e]] for e in range(episodes)]

        with open(data_path + f'/{self._model_name}.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(["episodes", "scores", 'avg_scores', 'avg_rewards'])
            writer.writerows(data)


    def create_graph(self):
        cwd = os.getcwd()
        data_path = os.path.join(cwd, 'pybullet/Data')
        graph_path = os.path.join(cwd, 'pybullet/Graphs')

        data = pd.read_csv(data_path + f'/{self._model_name}.csv')

        # Extract the y-values
        y_scores = data['scores'].values
        y_avg_scores = data['avg_scores'].values
        y_avg_rewards = data['avg_rewards'].values

        # Generate x-values
        x = data['episodes'].values

        # Create a line graph
        plt.plot(x, y_scores, label='Episode Score')
        plt.plot(x, y_avg_scores, label='Rolling Average Score')
        plt.plot(x, y_avg_rewards, label="Average Reward per Episode")

        # Add title and labels
        plt.title(f'{self._model_name} Score Graph')
        plt.xlabel('Episode')
        plt.ylabel('Score/Reward')

        plt.legend()

        # Save the graph to a file
        plt.savefig(f'{graph_path}/{self._model_name}.png')

        # Close the plot to free up memory
        plt.close()
        

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
    parser.add_argument('--env', type=str, default="blank_gui",
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
    parser.add_argument('--episodes', type=int, default=1000,
                        help='how many training episodes')
    parser.add_argument('--model_name', type=str, default='model',
                        help='name of the model being trained')
    parser.add_argument('--test_mode', type=bool, default=False,
                        help="set true or false for test mode")
    parser.add_argument('--load_model', type=str, default=None,
                        help="give name of pre-trained model to load")
    parser.add_argument('--model_version', type=int, default=0,
                        help="if load_model isn't blank, state which version of the model you want to load")
    parser.add_argument('--train', type=str, default='nav',
                        help='state what you want the model to train')
    args = parser.parse_args()
    main()
