import os
import torch
import torch.nn as nn
import numpy as np 
import random
from collections import deque
import pickle

# Neural Network for DQN

class DQN(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.net(x)
    

# Replay Buffer for Experience Replay

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones)
        )
    
    def __len__(self):
        return len(self.buffer)
    

# DQN Agent

class DQNAgent:
    def __init__(
            self,
            state_dim=6,
            action_dim=2,
            lr=5e-4,
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.05,
            epsilon_decay=0.999,       # SLOWER decay — explore longer
            batch_size=64,
            buffer_capacity=100000,    # 10x LARGER buffer — retains good experiences
            target_sync=10,
            warmup_steps=2000          # more warmup for diverse initial data
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_sync = target_sync
        self.warmup_steps = warmup_steps
        self.steps_done = 0
        self.tau = 0.005               # SOFT target update instead of hard copy

        # 2 networks: policy and target
        self.policy_net = DQN(state_dim, 128, action_dim)
        self.target_net = DQN(state_dim, 128, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss — robust to outlier death penalties
        self.buffer = ReplayBuffer(buffer_capacity)

    # action selection with epsilon-greedy strategy

    def choose_action(self, state, greedy=False):
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state_t)
            return q_values.argmax().item()
        
    # store transition in replay buffer
    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)
        self.steps_done += 1

    # training step — now uses Double DQN to reduce overestimation

    def train_step(self):
        if len(self.buffer) < self.warmup_steps:
            return None
        
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # DOUBLE DQN: policy net picks the best action, target net evaluates it
        # This reduces Q-value overestimation which was causing instability
        with torch.no_grad():
            best_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            max_next_q = self.target_net(next_states).gather(1, best_actions).squeeze(1)
            target_q = rewards + self.gamma * max_next_q * (1 - dones)

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()
    
    # SOFT target update — blend weights smoothly instead of hard copy
    def sync_target(self):
        for tp, pp in zip(self.target_net.parameters(), self.policy_net.parameters()):
            tp.data.copy_(self.tau * pp.data + (1 - self.tau) * tp.data)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # save and load model weights

    def save(self, path = "dqn_flappy.pth"):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'optimizer': self.optimizer.state_dict()
        }, path)
        # save buffer separately using pickle
        with open(path.replace('.pth', '_buffer.pkl'), 'wb') as f:
            pickle.dump(self.buffer.buffer, f)

        print(f"Model saved to {path}")

    def load(self, path = "dqn_flappy.pth"):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        # load buffer if it exists
        buf_path = path.replace('.pth', '_buffer.pkl')
        if os.path.exists(buf_path):
            with open(buf_path, 'rb') as f:
                self.buffer.buffer = pickle.load(f)
            print(f"Loaded buffer ← {buf_path}  ({len(self.buffer)} transitions)")
        print(f"Loaded ← {path}  (ε={self.epsilon:.3f}  steps={self.steps_done})")
