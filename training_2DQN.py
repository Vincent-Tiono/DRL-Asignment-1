import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm
from collections import deque

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "dueling_ddqn.pt"
HIDDEN_DIM = 128
BUFFER_SIZE = 100000
BATCH_SIZE = 128
GAMMA = 0.99
LR = 0.0005
TAU = 0.005
NUM_EPISODES = 2000

class NoisyLinear(nn.Module):
    """Noisy linear layer for exploration"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.w_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.w_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.b_mu = nn.Parameter(torch.empty(out_features))
        self.b_sigma = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.w_mu, nonlinearity='relu')
        nn.init.constant_(self.w_sigma, 0.5 / np.sqrt(self.w_mu.size(1)))
        nn.init.uniform_(self.b_mu, -0.1, 0.1)
        nn.init.constant_(self.b_sigma, 0.5 / np.sqrt(self.w_mu.size(1)))
        
    def forward(self, x):
        if self.training:
            w = self.w_mu + self.w_sigma * torch.randn_like(self.w_sigma)
            b = self.b_mu + self.b_sigma * torch.randn_like(self.b_sigma)
        else:
            w = self.w_mu
            b = self.b_mu
        return torch.mm(x, w.t()) + b

class DuelingQNetwork(nn.Module):
    """Dueling Double DQN with Noisy Networks"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.feature = nn.Sequential(
            NoisyLinear(input_dim, HIDDEN_DIM),
            nn.ReLU()
        )
        self.value = nn.Sequential(
            NoisyLinear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            NoisyLinear(HIDDEN_DIM, 1)
        )
        self.advantage = nn.Sequential(
            NoisyLinear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            NoisyLinear(HIDDEN_DIM, output_dim)
        )
        
    def forward(self, x):
        x = self.feature(x)
        value = self.value(x)
        advantage = self.advantage(x)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

class ReplayBuffer:
    """Standard replay buffer with uniform sampling"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states), np.array(actions), np.array(rewards),
            np.array(next_states), np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)

class TaxiAgent:
    def __init__(self, state_dim, action_dim):
        self.policy_net = DuelingQNetwork(state_dim, action_dim).to(DEVICE)
        self.target_net = DuelingQNetwork(state_dim, action_dim).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.buffer = ReplayBuffer(BUFFER_SIZE)
        self.action_dim = action_dim
        
    def act(self, state):
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            q_values = self.policy_net(state_t)
        return torch.argmax(q_values).item()
    
    def process_state(self, obs):
        """Feature engineering for the Taxi environment"""
        taxi_row, taxi_col, *stations, obs_n, obs_s, obs_e, obs_w, pass_stat, dest_stat = obs
        stations = np.array(stations).reshape(4, 2)
        
        features = [
            taxi_row / 20.0,
            taxi_col / 20.0,
            *((taxi_row - stations[:,0]) + (taxi_col - stations[:,1])) / 40.0,
            obs_n, obs_s, obs_e, obs_w,
            pass_stat, dest_stat,
            (pass_stat == 0) * np.min(np.abs(stations - np.array([[taxi_row, taxi_col]])).sum(axis=1)) / 40.0,
            (pass_stat == 1) * np.min(np.abs(stations - np.array([[taxi_row, taxi_col]])).sum(axis=1)) / 40.0
        ]
        
        # Ensure the feature vector has the correct length
        features = features[:16]  # Truncate if too long
        features.extend([0.0] * (16 - len(features)))  # Pad with zeros if too short
        
        return np.array(features, dtype=np.float32)

    def optimize(self):
        if len(self.buffer) < BATCH_SIZE:
            return 0.0
        
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)
        
        # Convert to tensors
        states_t = torch.FloatTensor(states).to(DEVICE)
        actions_t = torch.LongTensor(actions).to(DEVICE)
        rewards_t = torch.FloatTensor(rewards).to(DEVICE)
        next_states_t = torch.FloatTensor(next_states).to(DEVICE)
        dones_t = torch.FloatTensor(dones).to(DEVICE)
        
        # Double DQN with target network
        with torch.no_grad():
            next_actions = self.policy_net(next_states_t).argmax(1)
            next_q = self.target_net(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q = rewards_t + (1 - dones_t) * GAMMA * next_q
        
        current_q = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze()
        
        # Huber loss
        loss = nn.HuberLoss()(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()
        
        # Soft target update
        for t_param, p_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            t_param.data.copy_(TAU * p_param.data + (1.0 - TAU) * t_param.data)
        
        return loss.item()

def train():
    from simple_custom_taxi_env import SimpleTaxiEnv
    
    env = SimpleTaxiEnv()
    agent = TaxiAgent(state_dim=16, action_dim=6)
    best_reward = -np.inf
    
    for episode in tqdm(range(NUM_EPISODES)):
        obs, _ = env.reset()
        state = agent.process_state(obs)
        episode_reward = 0
        loss_total = 0
        steps = 0
        
        while True:
            action = agent.act(state)
            next_obs, reward, done, _ = env.step(action)
            next_state = agent.process_state(next_obs)
            
            # Store transition
            agent.buffer.push(state, action, reward, next_state, done)
            episode_reward += reward
            state = next_state
            obs = next_obs
            
            # Train every 4 steps
            if steps % 4 == 0:
                loss = agent.optimize()
                loss_total += loss
            
            if done:
                break
            steps += 1
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save(agent.policy_net.state_dict(), MODEL_PATH)
        
        # Progress logging
        if (episode + 1) % 50 == 0:
            avg_loss = loss_total / steps if steps > 0 else 0
            print(f"Ep {episode+1}: Reward {episode_reward:.1f} | Loss {avg_loss:.4f}")

if __name__ == "__main__":
    train()