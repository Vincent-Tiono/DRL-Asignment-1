import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm

# Constants
MODEL_PATH = "dqn.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNet(nn.Module):
    """Q-Network for DQN algorithm"""
    def __init__(self, in_dim, out_dim, hidden_dim=128):
        super(QNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.apply(self._init_weights)
    
    def _init_weights(self, mod):
        if isinstance(mod, nn.Linear):
            nn.init.kaiming_normal_(mod.weight, nonlinearity='relu')
            if mod.bias is not None:
                nn.init.constant_(mod.bias, 0.0)
    
    def forward(self, x):
        return self.network(x)

class Memory:
    """Experience replay buffer"""
    def __init__(self, size=100000):
        self.size = size
        self.buffer = []
        self.pos = 0
        
    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.size:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.pos = (self.pos + 1) % self.size
        
    def batch(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*samples))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class FeatureExtractor:
    """Extracts features from raw observations"""
    @staticmethod
    def process(obs):
        # Unpack observation
        t_row, t_col, s0_row, s0_col, s1_row, s1_col, s2_row, s2_col, s3_row, s3_col, \
        obs_n, obs_s, obs_e, obs_w, pass_stat, dest_stat = obs
        
        # Calculate distances to stations
        stations = [
            (s0_row, s0_col),
            (s1_row, s1_col),
            (s2_row, s2_col),
            (s3_row, s3_col)
        ]
        
        dists = []
        for s_row, s_col in stations:
            manhattan = abs(t_row - s_row) + abs(t_col - s_col)
            norm_dist = manhattan / 20.0  # Normalize by max possible distance
            dists.append(norm_dist)
        
        # Combine features
        features = [
            obs_n, obs_s, obs_e, obs_w,
            pass_stat, dest_stat,
            *dists,
            min(dists)
        ]
        
        return torch.FloatTensor(features).to(DEVICE), dists

class RewardShaper:
    """Shapes rewards to provide better learning signals"""
    @staticmethod
    def shape(obs, next_obs, action, reward):
        t_row, t_col, _, _, _, _, _, _, _, _, \
        obs_n, obs_s, obs_e, obs_w, pass_stat, dest_stat = obs
        
        next_t_row, next_t_col, _, _, _, _, _, _, _, _, \
        next_obs_n, next_obs_s, next_obs_e, next_obs_w, next_pass_stat, next_dest_stat = next_obs
        
        _, curr_dists = FeatureExtractor.process(obs)
        _, next_dists = FeatureExtractor.process(next_obs)
        
        new_reward = reward
        
        # Penalize hitting obstacles
        if (action == 0 and obs_s == 1) or \
           (action == 1 and obs_n == 1) or \
           (action == 2 and obs_e == 1) or \
           (action == 3 and obs_w == 1):
            new_reward -= 20.0
        
        # Reward being in open space
        if obs_n == 0 and obs_s == 0 and obs_e == 0 and obs_w == 0:
            new_reward += 0.5
        
        min_curr_dist = min(curr_dists)
        min_next_dist = min(next_dists)
        
        # Reward progress toward pickup/destination
        if pass_stat == 0:  # Passenger not picked up
            dist_change = min_next_dist - min_curr_dist
            new_reward -= dist_change * 3.0  # Reward for getting closer
        
        if pass_stat == 1:  # Passenger picked up
            dist_change = min_next_dist - min_curr_dist
            new_reward -= dist_change * 5.0  # Stronger reward when passenger is aboard
        
        # Reward successful pickup
        if action == 4 and pass_stat == 1 and next_pass_stat == 1:
            new_reward += 5.0
        
        # Penalize wrong pickup/dropoff
        if action == 4 and pass_stat == 0:
            new_reward -= 2.0
        
        if action == 5 and dest_stat == 0:
            new_reward -= 2.0
        
        # Penalize no movement
        if action < 4 and t_row == next_t_row and t_col == next_t_col:
            new_reward -= 0.5
        
        return new_reward

class Agent:
    """DQN Agent implementation"""
    def __init__(self, state_dim=11, action_dim=6, lr=0.001, gamma=0.99, tau=0.005):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        
        # Networks
        self.policy_net = QNet(state_dim, action_dim).to(DEVICE)
        self.target_net = QNet(state_dim, action_dim).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=100, verbose=True
        )
        self.criterion = nn.HuberLoss(delta=1.0)
        
        # Exploration
        self.eps = 1.0
        self.eps_min = 0.01
        self.eps_decay = 0.9995
        
    def act(self, state, explore=True):
        if explore and random.random() < self.eps:
            return random.randint(0, self.action_dim-1)
        else:
            with torch.no_grad():
                q_values = self.policy_net(state)
            return torch.argmax(q_values).item()
    
    def learn(self, states, actions, rewards, next_states, dones):
        # Convert to tensors
        states = torch.FloatTensor(states).to(DEVICE)
        actions = torch.LongTensor(actions).to(DEVICE)
        rewards = torch.FloatTensor(rewards).to(DEVICE)
        next_states = torch.FloatTensor(next_states).to(DEVICE)
        dones = torch.FloatTensor(dones).to(DEVICE)
        
        # Current Q-values
        curr_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN: Use policy net to select action, target net to evaluate
        with torch.no_grad():
            next_actions = self.policy_net(next_states).max(1)[1]
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Compute loss and update
        loss = self.criterion(curr_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.scheduler.step(loss)
        
        return loss.item()
    
    def update_target(self):
        """Soft update of target network"""
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def decay_epsilon(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)
    
    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)
    
    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())

class Trainer:
    """Handles the training process"""
    def __init__(self, agent, env, memory, batch_size=128):
        self.agent = agent
        self.env = env
        self.memory = memory
        self.batch_size = batch_size
        self.best_reward = -float('inf')
        self.target_update_freq = 10
    
    def train_episode(self):
        obs, _ = self.env.reset()
        state, _ = FeatureExtractor.process(obs)
        done = False
        total_reward = 0
        episode_loss = 0
        steps = 0
        
        while not done:
            # Select action
            action = self.agent.act(state)
            
            # Take action
            next_obs, reward, done, _ = self.env.step(action)
            next_state, _ = FeatureExtractor.process(next_obs)
            
            # Shape reward
            shaped_reward = RewardShaper.shape(obs, next_obs, action, reward)
            
            # Store transition
            self.memory.add(
                state.cpu().numpy(),
                action,
                shaped_reward,
                next_state.cpu().numpy(),
                done
            )
            
            total_reward += reward
            state = next_state
            obs = next_obs
            steps += 1
            
            # Learn
            if len(self.memory) >= self.batch_size:
                batch = self.memory.batch(self.batch_size)
                loss = self.agent.learn(*batch)
                episode_loss += loss
        
        return total_reward, episode_loss / steps if steps > 0 else 0
    
    def train(self, episodes=5000):
        """Main training loop"""
        for episode in tqdm(range(episodes)):
            reward, loss = self.train_episode()
            
            # Update target network
            if (episode + 1) % self.target_update_freq == 0:
                self.agent.update_target()
            
            # Decay exploration
            self.agent.decay_epsilon()
            
            # Save best model
            if reward > self.best_reward:
                self.best_reward = reward
                self.agent.save(MODEL_PATH)
            
            # Logging
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode+1}, Reward: {reward:.2f}, Loss: {loss:.4f}, Epsilon: {self.agent.eps:.4f}")

def main():
    """Main function to run training"""
    from simple_custom_taxi_env import SimpleTaxiEnv
    
    # Initialize components
    env = SimpleTaxiEnv()
    agent = Agent(state_dim=11, action_dim=6, lr=0.0005, gamma=0.99, tau=0.005)
    memory = Memory(size=100000)
    trainer = Trainer(agent, env, memory, batch_size=128)
    
    # Train agent
    trainer.train(episodes=5000)

if __name__ == "__main__":
    main()