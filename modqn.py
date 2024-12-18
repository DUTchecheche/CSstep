import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from env import get_valid_actions, get_fps, get_fps_list
import random


class Memory(object):
    def __init__(self, capacity, state_dim, action_dim, reward_dim):
        self.capacity = capacity
        self.dim = state_dim + action_dim + 1 + reward_dim
        # the dimensionality of step_index is 1
        self.data = np.zeros((self.capacity, self.dim))
        self.pointer = 0

    def reset(self):
        self.data = np.zeros((self.capacity, self.dim))
        self.pointer = 0

    def store(self, s, a, step, r):
        index = self.pointer % self.capacity
        self.data[index] = np.concatenate((s, a, np.array([step]), r), axis=0)
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory not full'
        indexes = np.random.choice(self.capacity, size=n)
        return self.data[indexes], indexes


class Memory_smiles(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = [i for i in range(self.capacity)]
        self.pointer = 0

    def reset(self):
        self.data = [i for i in range(self.capacity)]
        self.pointer = 0

    def store(self, a):
        index = self.pointer % self.capacity
        self.data[index] = a
        self.pointer += 1

    def sample(self, indexes):
        return [self.data[i] for i in indexes]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1025, 512)
        # molecular fingerprint + step_index =1024 + 1
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 16)
        self.q = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.q(x)
    # Q network


class MODQN(object):
    def __init__(self, reward_list, lr, lr_decrease, target_iter, state_dim, action_dim, gamma, epsilon, epsilon_increase, double_q):
        self.reward_dim = len(reward_list)
        self.lr = lr
        self.lr_decrease = lr_decrease
        self.target_iter = target_iter
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon_max = epsilon
        self.epsilon_increase = epsilon_increase
        self.epsilon = 0.5 if epsilon_increase != 0.0 else epsilon
        self.loss_function = nn.SmoothL1Loss()
        self.step_counter = 0
        self.double_q = double_q
        for i in range(self.reward_dim):
            exec('self.eval_Q_{}=Net()'.format(i))
            exec('self.target_Q_{}=Net()'.format(i))
            exec('for p in self.target_Q_{}.parameters(): p.requires_grad = False'.format(i))
            exec('self.optimizer_{} = torch.optim.Adam(self.eval_Q_{}.parameters(), lr=self.lr)'.format(i, i))
            # Multi-agent strategy

    def choose_action(self, state, current_step):
        actions = get_valid_actions(state,
                                    atom_types=['C', 'O', 'N'],
                                    allow_removal=True,
                                    allow_no_modification=True,
                                    allowed_ring_sizes=(5, 6),
                                    allow_bonds_between_rings=False)
        random.shuffle(actions)
        # shuffle actions to prevent torch.argmax from only taking the one that appears first when the Q values of some actions are the same
        state_temp = get_fps(smiles=state, length=1024)
        if np.random.uniform() < self.epsilon:
            actions_fps = torch.FloatTensor(np.array(get_fps_list(actions, length=1024)))
            current_step = torch.full((actions_fps.size(0), 1), current_step)
            value_list = []
            for i in range(self.reward_dim):
                exec('value_list.append(self.eval_Q_{}.forward(torch.cat([actions_fps,current_step],dim=1)))'.format(i))
            # output format: tensor: batch_size*output_dim=1
            value = sum(value_list)
            # value, _ = torch.max(torch.cat(value_list, dim=1), dim=1)
            action = actions[torch.argmax(value)]
        else:
            action = np.random.choice(actions)
        action_fps = get_fps(action, length=1024)
        return action, state_temp, action_fps
        # return smiles, fps, fps

    def learn(self, data, data_smi, max_step):
        if self.step_counter % self.target_iter == 0:
            with torch.no_grad():
                for i in range(self.reward_dim):
                    exec('self.target_Q_{}.load_state_dict(self.eval_Q_{}.state_dict())'.format(i, i))
        # parameter transfer for target_Q network
        self.step_counter += 1

        for i in range(self.reward_dim):
            exec('r_{} = torch.FloatTensor(data[:,{}-self.reward_dim][ :,np.newaxis])'.format(i, i))
        a = torch.FloatTensor(data[:, self.state_dim:self.state_dim + self.action_dim])
        step = torch.FloatTensor(data[:, -(self.reward_dim+1):-self.reward_dim])
        # format in memory：s,a,step,r

        if self.double_q:
            for i in range(self.reward_dim):
                exec('q_eval_{} = self.eval_Q_{}(torch.cat([a,step],dim=1))'.format(i, i))
                exec('q_next_eval_{} = []'.format(i))
                exec('q_next_target_{} = []'.format(i))

            for batch_index in range(len(data_smi)):
                if data[batch_index, -(self.reward_dim+1)]+1 == max_step:
                    for i in range(self.reward_dim):
                        exec('q_next_eval_{}.append(torch.zeros(1,1))'.format(i))
                        exec('q_next_target_{}.append(torch.zeros(1,1))'.format(i))
                else:
                    s_ = data_smi[batch_index]
                    a_ = get_valid_actions(s_, atom_types=['C', 'O', 'N'],
                                           allow_removal=True,
                                           allow_no_modification=True,
                                           allowed_ring_sizes=(5, 6),
                                           allow_bonds_between_rings=False)
                    a__fps = torch.FloatTensor(np.array(get_fps_list(a_, length=1024)))
                    step_ = torch.full((a__fps.size(0), 1), data[batch_index, -(self.reward_dim + 1)] + 1)
                    for i in range(self.reward_dim):
                        exec('q_next_eval_{}.append(self.eval_Q_{}(torch.cat([a__fps,step_],dim=1)).detach())'.format(i, i))
                        exec('q_next_target_{}.append(self.target_Q_{}(torch.cat([a__fps,step_],dim=1)).detach())'.format(i, i))

            for i in range(self.reward_dim):
                exec('q_next_max_index_{} = [i.argmax() for i in q_next_eval_{}]'.format(i, i))
                exec('q_next_max_{}=[]'.format(i))
                exec('for j in range(len(q_next_max_index_{})): q_next_max_{}.append(q_next_target_{}[j][q_next_max_index_{}[j]])'.format(i, i, i, i))
                exec('q_target_{} = r_{} + torch.FloatTensor([{}*k for k in q_next_max_{}]).unsqueeze(1)'.format(i, i,
                                                                                                                 self.gamma,
                                                                                                                 i))
                exec('loss_{} = self.loss_function(q_eval_{}, q_target_{})'.format(i, i, i))
                exec('self.optimizer_{}.zero_grad()'.format(i))
                exec('loss_{}.backward()'.format(i))
                exec('self.optimizer_{}.step()'.format(i))
        else:
            for i in range(self.reward_dim):
                exec('q_eval_{} = self.eval_Q_{}(torch.cat([a,step],dim=1))'.format(i, i))
                exec('q_next_{} = []'.format(i))

            for batch_index in range(len(data_smi)):
                if data[batch_index, -(self.reward_dim+1)]+1 == max_step:
                    for i in range(self.reward_dim):
                        exec('q_next_{}.append(torch.zeros(1,1))'.format(i))
                else:
                    s_ = data_smi[batch_index]
                    a_ = get_valid_actions(s_, atom_types=['C', 'O', 'N'],
                                           allow_removal=True,
                                           allow_no_modification=True,
                                           allowed_ring_sizes=(5, 6),
                                           allow_bonds_between_rings=False)
                    a__fps = torch.FloatTensor(np.array(get_fps_list(a_, length=1024)))
                    step_ = torch.full((a__fps.size(0), 1), data[batch_index, -(self.reward_dim+1)]+1)
                    for i in range(self.reward_dim):
                        exec('q_next_{}.append(self.target_Q_{}(torch.cat([a__fps,step_],dim=1)).detach())'.format(i, i))

            for i in range(self.reward_dim):
                exec('q_next_max_{} = [i.max() for i in q_next_{}]'.format(i, i))
                # align q_eval and q_target
                exec('q_target_{} = r_{} + torch.FloatTensor([{}*j for j in q_next_max_{}]).unsqueeze(1)'.format(i, i,
                                                                                                                 self.gamma,
                                                                                                                 i))
                exec('loss_{} = self.loss_function(q_eval_{}, q_target_{})'.format(i, i, i))
                exec('self.optimizer_{}.zero_grad()'.format(i))
                exec('loss_{}.backward()'.format(i))
                exec('self.optimizer_{}.step()'.format(i))

    def epsilon_up(self):
        self.epsilon = self.epsilon + self.epsilon_increase if self.epsilon < self.epsilon_max else self.epsilon_max

    def lr_down(self):
        self.lr = self.lr * self.lr_decrease if self.lr > 0.0001 else 0.0001
