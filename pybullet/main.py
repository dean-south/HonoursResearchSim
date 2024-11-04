import os
import sys
import argparse
import gym as gym
import numpy as np
import wandb
import json
import csv
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import TD3, HerReplayBuffer, PPO, SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.td3.policies import MlpPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.ppo import MlpPolicy
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

load_func = {'td3':TD3.load, 
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
        

        self.sb3_models = ['td3', "sac", 'ppo', 'recppo', 'tqc']


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
        elif self._ctr == "td3":
            n_actions = self._env.action_space.shape[-1]
            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))

            if not self.test_mode:
                config = {
                    "policy_type": "MlpPolicy",
                    "total_timesteps": 10000000,
                    "env_name": "Blank-v0",
                }

                self.run = wandb.init(
                    project="TD3 Maze",
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
                # learning_rate=exponential_schedule(initial_learning_rate, decay_rate=0.5),
                learning_starts=0
                )
        elif self._ctr == 'ppo':

            if not self.test_mode:
                config = {
                    "policy_type": "MlpPolicy",
                    "total_timesteps": 1000000,
                    "env_name": "Empty-v0",
                }

                self.run = wandb.init(
                        project="PPO Maze",
                        config=config,
                        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                        save_code=True,  # optional
                    )
                
            initial_learning_rate = 0.0003

            policy_kwargs = dict(net_arch=[128, 128])
                
            self.model = PPO(
                'MlpPolicy', # CustomMlpPolicy,
                self._env,
                verbose=1,
                tensorboard_log=f"runs/{self._model_name}",
                policy_kwargs=policy_kwargs
            )
        
        elif self._ctr == 'sac':
            n_actions = self._env.action_space.shape[-1]
            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.05 * np.ones(n_actions))

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
                
            initial_learning_rate = 0.0003

            self.model = SAC(
                'MlpPolicy', # CustomMlpPolicy,
                self._env,
                verbose=1,
                tensorboard_log=f"runs/{self._model_name}",
                # action_noise=action_noise
                learning_rate=exponential_schedule(initial_learning_rate, decay_rate=0.5),
                # learning_starts=0,
                # target_update_interval=10
            )
                

        elif self._ctr == 'recppo':
            if not self.test_mode:
                config = {
                    "policy_type": "MlpLstmPolicy",
                    "total_timesteps": 1000000,
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

        if args.load_model is not None and any(self._ctr == ctr for ctr in self.sb3_models):
            print("Loading Model")
            self.model.set_parameters(f'models/{args.load_model}/model')
            # self.model = load_func[self._ctr](f'models/{args.load_model}/model',env=self._env, verbose=1)

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

        velocity_history = np.array([])
        episode_lenghts = np.array([])
        episode_completeness = np.array([])
        num_crashes = 0
        num_timeouts = 0
        num_complete = 0

        kill_sim = False

        for e in range(self._episodes if self.test_mode else 1):
            self._done = 0
            score = 0
            self._i = -1
            obs = self._env.reset()

            path_len = len(self._env.path)

            goals_reached = 0     

            pose = self._env.get_pose()
            goal_cell = self._env.path[0]

            if self.test_mode:
                print(f'Starting Episode {e}, starting cell: {self.pose_to_cell(pose[:2])}, goal cell: {goal_cell}, ')

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
                    
                    elif any(self._ctr == ctr for ctr in self.sb3_models):
                        if not self.test_mode:

                            # Create the wandb callback
                            wandb_callback = WandbCallback(
                                gradient_save_freq=10,
                                model_save_freq=5000,
                                model_save_path=f"models/{self.run.id}",
                                verbose=2,
                            )

                            # Create a checkpoint callback to save the model
                            # checkpoint_callback = CheckpointCallback(
                            #     save_freq=10000,  # Save every 10000 steps
                            #     save_path=f"./models/{self.run.id}/",
                            #     name_prefix="rl_model"
                            # )


                            self.model.learn(total_timesteps=4000000, log_interval=10, 

                                    callback=wandb_callback
                                )
                            self.run.finish()
                        else:
                            # print(state[:6])
                            action, _ = self.model.predict(obs)

                            # print(f'{obs=} \n {action=}')

                            # action = [1,1]

                            # print(f'{action=} {action[0]=}')

                            obs, rew, self._done, info = self._movement(action)

                            velocity_history = np.append(velocity_history, info['velocity'][0])

                            if obs[0]*16*np.sqrt(2) < 0.3:
                                goals_reached += 1
                                # goal_cell = self._env.path[0]
                                # pose = self._env.get_pose()
                                # print(f'starting cell: {self.pose_to_cell(pose[:2])}, goal cell: {goal_cell}')  

                except KeyboardInterrupt:
                    print(' The simulation was forcibly stopped.')
                    if not self.test_mode:
                        self.model.save(f"models/{self.run.id}/model")
                    kill_sim = True
                    break

            
            if self.test_mode:
                episode_lenghts = np.append(episode_lenghts, self._i)   
                episode_completeness = np.append(episode_completeness, goals_reached/path_len)
                if self._env.robot_collision():
                    num_crashes += 1
                elif goals_reached/path_len == 1:
                    num_complete += 1
                else:
                    num_timeouts += 1

            if kill_sim:
                break
            
        if self.test_mode and not kill_sim:
            print(f'velocity mean: {np.mean(velocity_history)}, velocity variance: {np.var(velocity_history)}')
            print(f'avg episode length: {np.mean(episode_lenghts)}')
            print(f'avg episode completeness: {np.mean(episode_completeness)}')
            print(f'number of completes: {num_complete}, number of crashes {num_crashes}, number of timeouts {num_timeouts}')

            data = {"velocity mean": np.mean(velocity_history),
                    'velocity variance': np.var(velocity_history),
                    'avg episode length': np.mean(episode_lenghts),
                    'avg episode completeness': np.mean(episode_completeness),
                    'number of completes, crashes, timemouts':[num_complete, num_crashes, num_timeouts]}
            

            with open(f"data/{self._env.task_name}/{self._env.task_name}_{self._ctr}_{args.load_model}_{self._model_name}.json", 'w') as outfile:
                json.dump(data, outfile)

        self._env.close()


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
    parser.add_argument('--episodes', type=int, default=10,
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
