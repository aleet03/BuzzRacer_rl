import torch
import torch.nn as nn
import random
from collections import namedtuple, deque
import torch.nn.functional as F

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done') )

class Actor(nn.Module):

    def __init__(self, n_observations, n_actions, action_limit):
        super(Actor, self).__init__()
        self.action_limit = action_limit
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # called via Module(x)
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))

        # output is scaled to [-action_limit, action_limit] using tanh
        return torch.tanh(self.layer3(x)) * self.action_limit
    
class Critic(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(n_observations + n_actions, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, 1)

    # called via Modele(x)
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))

        return self.layer3(x)
        
    
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
