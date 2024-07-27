import numpy as np
import torch as T
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import os
import random
from math import acos, asin, pi


class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions) -> None:
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done
        self.reward_memory[index] = reward
        self.action_memory[index] = action

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ =self.new_state_memory[batch]
        action = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, action, rewards, states_, dones
    

class CriticNetwork(nn.Module):
    
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir='tmp/td3') -> None:
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.name +'_td3')

        self.fc1 = nn.Linear(self.input_dims + self.n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)

        self.optimiser = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        q1_action_value = self.fc1(T.cat([state, action], dim=1))
        q1_action_value = F.relu(q1_action_value)
        q1_action_value = self.fc2(q1_action_value)
        q1_action_value = F.relu(q1_action_value)

        q1 = self.q1(q1_action_value)

        return q1
    
    def save_checkpoint(self, save_path):
        T.save(self.state_dict(), save_path + f'/{self.name}')

    def load_checkpoint(self, load_path):
        self.load_state_dict(T.load(load_path + f'/{self.name}'))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir='models/test') -> None:
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims =fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimiser = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = T.tanh(self.mu(prob))

        return mu
    
    def save_checkpoint(self, save_path):
        T.save(self.state_dict(), save_path + f'/{self.name}')

    def load_checkpoint(self, load_path):
        self.load_state_dict(T.load(load_path + f'/{self.name}'))


class Agent():
    def __init__(self, alpha, beta, input_dims, tau, env,
                gamma=0.99, update_actor_interval=2, warmup=1000, n_actions=2, max_size=1000000,
                layer1_size=400, layer2_size=300, batch_size=100, noise=0.1, verbose=False, model_name='model') -> None:
        
        self.env = env
        self.verbose = verbose
        self.gamma = gamma
        self.tau = tau
        self.max_action = 1
        self.min_action = -1
        self.memory = ReplayBuffer(max_size,input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.time_step = 0
        self.warmup = warmup
        self.n_actions = n_actions
        self.update_actor_iter = update_actor_interval
        self.model_name = model_name
        self.cell_path = []
        self.wrong_cell = 0
        self.current_cell = []
        self.desired_cell = []
        self.prev_cell = []
        self.state = []

        self.actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size, self.n_actions, name='actor')

        self.critic_1 = CriticNetwork(beta, input_dims, layer1_size, layer2_size, self.n_actions, name='critic_1')
        self.critic_2 = CriticNetwork(beta, input_dims, layer1_size, layer2_size, self.n_actions, name='critic_2')

        self.target_actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size, self.n_actions, name='target_actor')
        self.target_critic_1 = CriticNetwork(beta, input_dims, layer1_size, layer2_size, self.n_actions, name='target_critic_1')
        self.target_critic_2 = CriticNetwork(beta, input_dims, layer1_size, layer2_size, self.n_actions, name='target_critic_2')

        self.noise = noise
        self.update_network_parameters(tau=1)

    def get_action(self):   

        if self.time_step < self.warmup:
            mu = T.tensor(np.random.normal(scale=self.noise, size=(self.n_actions,))).to(self.actor.device)
        else:
            state = T.tensor(self.state, dtype=T.float).to(self.actor.device)
            mu = self.actor.forward(state).to(self.actor.device)

            if self.current_cell == self.desired_cell:
                r = random.random()

                if r < 0.1:
                    return np.array([0,0])

        mu_prime = mu + T.tensor(np.random.normal(scale=self.noise), dtype=T.float).to(self.actor.device)

        mu_prime = T.clamp(mu_prime, self.min_action, self.max_action)
        self.time_step += 1
        
        return mu_prime.cpu().detach().numpy()
    

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state,action,reward,new_state,done)


    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
             return
        
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        state = T.tensor(state, dtype=T.float).to(self.critic_1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic_1.device)
        reward = T.tensor(reward, dtype=T.float).to(self.critic_1.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.critic_1.device)
        done = T.tensor(done).to(self.critic_1.device)

        target_actions = self.target_actor.forward(state_)
        target_actions = target_actions + T.clamp(T.tensor(np.random.normal(scale=0.2)), -0.5, 0.5)
        target_actions = T.clamp(target_actions, self.min_action, self.max_action)

        q1_ = self.target_critic_1.forward(state_, target_actions)
        q2_ = self.target_critic_2.forward(state_, target_actions)

        q1 = self.critic_1.forward(state, action)
        q2 = self.critic_2.forward(state, action)

        q1_[done] = 0.0
        q2_[done] = 0.0

        q1_ = q1_.view(-1)
        q2_ = q2_.view(-1)

        critic_value_ = T.min(q1_, q2_)

        target = reward + self.gamma*critic_value_
        target = target.view(self.batch_size, 1)

        self.critic_1.optimiser.zero_grad()
        self.critic_2.optimiser.zero_grad()

        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)
        critic_loss = q1_loss + q2_loss
        critic_loss.backward()
        self.critic_1.optimiser.step()
        self.critic_2.optimiser.step()

        self.learn_step_cntr += 1

        if self.learn_step_cntr % self.update_actor_iter != 0:
            return

        self.actor.optimiser.zero_grad()
        actor_q1_loss = self.critic_1.forward(state, self.actor.forward(state))
        actor_loss = -T.mean(actor_q1_loss)
        actor_loss.backward()
        self.actor.optimiser.step()

        self.update_network_parameters()


    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()

        critic_1 = dict(critic_1_params)
        critic_2 = dict(critic_2_params)
        actor = dict(actor_params)
        target_actor = dict(target_actor_params)
        target_critic_1 = dict(target_critic_1_params)
        target_critic_2 = dict(target_critic_2_params)

        for name in critic_1:
            critic_1[name] = tau * critic_1[name].clone() + (1 - tau) * target_critic_1[name].clone()

        for name in critic_2:
            critic_2[name] = tau * critic_1[name].clone() + (1 - tau) * target_critic_2[name].clone()

        for name in actor:
            actor[name] = tau * actor[name].clone() + (1 - tau) * target_actor[name].clone()

        self.target_critic_1.load_state_dict(critic_1)
        self.target_critic_2.load_state_dict(critic_2)
        self.actor.load_state_dict(actor)

    def desired_orientation(self):
        unit_v = (np.array(self.desired_cell) - np.array(self.current_cell))/np.linalg.norm([self.desired_cell, self.current_cell])

        return acos(unit_v[0]) + asin(unit_v[1])
    
    def get_reward(self):
        reward = 0

        # print(self.prev_cell, self.current_cell, self.desired_cell)

        if self.current_cell == self.desired_cell:

            if abs(self.state[0]) + abs(self.state[1]) + abs(self.state[2]) == 0:
                return 1000
            elif abs(self.state[0]) + abs(self.state[1]) + abs(self.state[2]) < 0.5:
                return min(750, 0.001/abs(self.state[0]) + abs(self.state[1]) + abs(self.state[2]))
            
            reward = - (abs(self.state[0]) + abs(self.state[1]) + abs(self.state[2]))
            
        elif self.prev_cell != self.current_cell \
            and np.linalg.norm([self.prev_cell,self.desired_cell]) > np.linalg.norm([self.current_cell, self.desired_cell]):
            reward = 100
        elif self.prev_cell != self.current_cell \
            and np.linalg.norm([self.prev_cell,self.desired_cell]) < np.linalg.norm([self.current_cell, self.desired_cell]):
            self.wrong_cell = True
            reward = -200
        elif self.desired_orientation() - pi/8 > self.state[2] and self.desired_orientation() + pi/8 < self.state[2]:
            reward = 1
        else:        
            reward = -1

        return reward


    def reset(self):
        pass

    def save_models(self, episode):

        print('... saving model ...')

        model_e = self.model_name + f'_{episode}'
        path = os.getcwd() + '/saved_models/' + self.model_name

        save_dir_path = os.path.join(path, model_e)

        os.mkdir(save_dir_path)

        self.actor.save_checkpoint(save_dir_path)
        self.target_actor.save_checkpoint(save_dir_path)
        self.critic_1.save_checkpoint(save_dir_path)
        self.target_critic_1.save_checkpoint(save_dir_path)
        self.critic_2.save_checkpoint(save_dir_path)
        self.target_critic_2.save_checkpoint(save_dir_path)


    def load_models(self, load_model_name, model_version):

        print('... loading model ...')

        model_e = load_model_name + f'_{model_version}'
        path = os.getcwd() + '/saved_models/' + load_model_name

        load_dir_path = os.path.join(path, model_e)

        self.actor.load_checkpoint(load_dir_path)
        self.target_actor.load_checkpoint(load_dir_path)
        self.critic_1.load_checkpoint(load_dir_path)
        self.target_critic_1.load_checkpoint(load_dir_path)
        self.critic_2.load_checkpoint(load_dir_path)
        self.target_critic_2.load_checkpoint(load_dir_path)


