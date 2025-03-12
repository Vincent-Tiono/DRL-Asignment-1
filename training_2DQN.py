import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "ddqn.pt"

# ================== Enhanced Network Architecture ==================
class DuelingDQN(nn.Module):
    """Dueling Double DQN with noisy layers"""
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 128),  # input_dim should be 18
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        features = self.feature(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values

# ================== Prioritized Experience Replay ==================
class PrioritizedReplayBuffer:
    def __init__(self, capacity=100000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.beta = 0.4
        self.beta_increment = 0.001

    def add(self, experience):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return []
            
        priorities = self.priorities[:len(self.buffer)]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        experiences = [self.buffer[idx] for idx in indices]
        
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = np.min([1.0, self.beta + self.beta_increment])
        
        return experiences, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)

# ================== Enhanced Feature Engineering ==================
class StateProcessor:
    @staticmethod
    def process(obs):
        taxi_row, taxi_col, s0r, s0c, s1r, s1c, s2r, s2c, s3r, s3c, \
        obs_n, obs_s, obs_e, obs_w, pass_stat, dest_stat = obs
        
        # Relative positions
        stations = [(s0r, s0c), (s1r, s1c), (s2r, s2c), (s3r, s3c)]
        dists = [abs(taxi_row-r) + abs(taxi_col-c) for r,c in stations]
        min_dist = min(dists)
        
        # Exactly 17 features to match evaluation
        features = [
            # Normalized position (2)
            taxi_row/4.0, taxi_col/4.0,
            
            # Obstacles (4)
            obs_n, obs_s, obs_e, obs_w,
            
            # Status (2)
            pass_stat, dest_stat,
            
            # Normalized distances to stations (4)
            *[d/8.0 for d in dists],
            
            # Closest station (1)
            min_dist/8.0,
            
            # Relative positions to first two stations (4)
            (taxi_row - s0r)/4.0,
            (taxi_col - s0c)/4.0,
            (taxi_row - s1r)/4.0,
            (taxi_col - s1c)/4.0
        ]
        
        # Ensure exactly 17 features
        assert len(features) == 17, f"Expected 17 features, got {len(features)}"
        
        return torch.FloatTensor(features).to(DEVICE).unsqueeze(0)  # Add batch dimension

    @staticmethod
    def get_min_distance(obs):
        """Calculate minimum distance to any station"""
        taxi_row, taxi_col = obs[0], obs[1]
        stations = [
            (obs[2], obs[3]),  # Station 0
            (obs[4], obs[5]),  # Station 1
            (obs[6], obs[7]),  # Station 2
            (obs[8], obs[9])   # Station 3
        ]
        return min(abs(taxi_row-r) + abs(taxi_col-c) for r,c in stations)

# ================== Reward Shaping ==================
class RewardShaper:
    @staticmethod
    def shape(obs, action, next_obs, reward):
        # Basic penalties/rewards
        shaped = reward * 2.0  # Amplify environment rewards
        
        # Illegal move penalty
        if action < 4 and obs[10+action] == 1:
            shaped -= 15.0
            
        # Progress reward
        curr_dist = StateProcessor.get_min_distance(obs)
        next_dist = StateProcessor.get_min_distance(next_obs)
        shaped += (curr_dist - next_dist) * 20.0
        
        # Fuel conservation
        shaped -= 0.2  # Per-step penalty
        
        return shaped

# ================== Training Agent ==================
class TaxiAgent:
    def __init__(self, state_dim, action_dim):
        self.policy_net = DuelingDQN(state_dim, action_dim).to(DEVICE)
        self.target_net = DuelingDQN(state_dim, action_dim).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=0.00025, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10000)
        self.buffer = PrioritizedReplayBuffer(capacity=100000)
        
        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = 256
        self.update_freq = 100
        self.steps = 0
        
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995

    def act(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randint(0, 5)
        
        with torch.no_grad():
            q_values = self.policy_net(state)
            return torch.argmax(q_values).item()

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return 0.0
            
        # Sample from buffer
        experiences, indices, weights = self.buffer.sample(self.batch_size)
        weights = torch.FloatTensor(weights).to(DEVICE)
        
        # Unpack batch and convert to tensors
        batch = list(zip(*experiences))
        states = torch.FloatTensor(np.stack(batch[0])).to(DEVICE)  # Convert numpy to tensor
        actions = torch.LongTensor(batch[1]).to(DEVICE)
        rewards = torch.FloatTensor(batch[2]).to(DEVICE)
        next_states = torch.FloatTensor(np.stack(batch[3])).to(DEVICE)  # Convert numpy to tensor
        dones = torch.FloatTensor(batch[4]).to(DEVICE)
        
        # Current Q values for chosen actions
        q_values = self.policy_net(states)  # Shape: [batch_size, num_actions]
        current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # Shape: [batch_size]
        
        # Double DQN: Use policy net to select actions, target net to evaluate
        with torch.no_grad():
            # Select best actions from policy net
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)  # Shape: [batch_size, 1]
            # Evaluate Q values using target net
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)  # Shape: [batch_size]
            # Compute target Q values
            target_q = rewards + (1 - dones) * self.gamma * next_q  # Shape: [batch_size]
        
        # Compute loss
        loss = torch.mean(weights * (current_q - target_q)**2)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()
        self.scheduler.step()
        
        # Update priorities
        errors = torch.abs(current_q - target_q).detach().cpu().numpy()
        self.buffer.update_priorities(indices, errors + 1e-5)
        
        # Soft target update
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau*policy_param.data + (1-self.tau)*target_param.data)
        
        return loss.item()

    def train_episode(self, env):
        obs, _ = env.reset()
        state = StateProcessor.process(obs)  # Shape: [1, state_dim]
        total_reward = 0
        done = False
        episode_loss = 0.0
        step_count = 0
        
        while not done:
            # Select action
            action = self.act(state)
            
            # Take action
            next_obs, reward, done, _ = env.step(action)
            next_state = StateProcessor.process(next_obs)  # Shape: [1, state_dim]
            
            # Shape reward
            shaped_reward = RewardShaper.shape(obs, action, next_obs, reward)
            
            # Store transition
            self.buffer.add((
                state.squeeze(0).cpu().numpy(),  # Remove batch dim for storage
                action,
                shaped_reward,
                next_state.squeeze(0).cpu().numpy(),  # Remove batch dim for storage
                done
            ))
            
            # Learn from experience
            if len(self.buffer) >= self.batch_size:
                loss = self.learn()
                episode_loss += loss
            
            # Update tracking variables
            total_reward += reward
            step_count += 1
            self.steps += 1
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Move to next state
            state = next_state
            obs = next_obs
        
        # Calculate average loss
        avg_loss = episode_loss / step_count if step_count > 0 else 0.0
        
        return total_reward, avg_loss

def main():
    from simple_custom_taxi_env import SimpleTaxiEnv
    
    env = SimpleTaxiEnv()
    state_dim = 17  # Based on StateProcessor output
    agent = TaxiAgent(state_dim, 6)
    
    best_score = -np.inf
    rewards = []
    
    for episode in tqdm(range(2000)):
        score, loss = agent.train_episode(env)
        rewards.append(score)
        
        if score > best_score:
            best_score = score
            torch.save(agent.policy_net.state_dict(), MODEL_PATH)
            
        if (episode+1) % 50 == 0:
            avg_score = np.mean(rewards[-50:])
            print(f"Ep {episode+1} | Avg: {avg_score:.1f} | Best: {best_score} | ε: {agent.epsilon:.3f} | Loss: {loss:.3f}")

if __name__ == "__main__":
    main()