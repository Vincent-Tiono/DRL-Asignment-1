import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm

# Config
CFG = {
    "model_path": "dqn.pt",
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "buffer_size": 50000,
    "batch_size": 64,
    "gamma": 0.99,
    "lr": 0.001,
    "episodes": 15000,
    "eps_start": 1.0,
    "eps_min": 0.01,
    "eps_decay": 0.9999,
    "target_update_freq": 5,
    "tau": 0.001
}

class Network(nn.Module):
    """Neural network for Q-learning"""
    def __init__(self, in_dim, out_dim):
        super(Network, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.uniform_(m.bias, -0.05, 0.05)
    
    def forward(self, x):
        return self.layers(x)

class Memory:
    """Experience replay buffer"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        
    def store(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.pos = (self.pos + 1) % self.capacity
        
    def get_batch(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class StateProcessor:
    """Process raw state to usable features"""
    def __init__(self, device):
        self.device = device
    
    def process(self, obs):
        # Unpack observation
        t_row, t_col, s0_row, s0_col, s1_row, s1_col, \
        s2_row, s2_col, s3_row, s3_col, \
        obs_n, obs_s, obs_e, obs_w, \
        pass_loc, dest_loc = obs
        
        # Get station positions
        stations = [
            (s0_row, s0_col),
            (s1_row, s1_col),
            (s2_row, s2_col),
            (s3_row, s3_col)
        ]
        
        # Calculate distances to stations
        dists = []
        for s_row, s_col in stations:
            dist = abs(t_row - s_row) + abs(t_col - s_col)
            norm_dist = dist / 20.0  # Normalize
            dists.append(norm_dist)
        
        # Create feature vector
        features = [
            obs_n, obs_s, obs_e, obs_w,
            pass_loc, dest_loc,
            dists[0], dists[1], dists[2], dists[3],
            min(dists)
        ]
        
        return torch.FloatTensor(features).to(self.device), dists

class RewardShaper:
    """Shape rewards to help learning"""
    def __init__(self, processor):
        self.processor = processor
    
    def shape(self, obs, next_obs, action, reward):
        # Unpack observations
        t_row, t_col, _, _, _, _, _, _, _, _, \
        obs_n, obs_s, obs_e, obs_w, \
        pass_loc, dest_loc = obs
        
        next_t_row, next_t_col, _, _, _, _, _, _, _, _, \
        next_obs_n, next_obs_s, next_obs_e, next_obs_w, \
        next_pass_loc, next_dest_loc = next_obs
        
        # Get distances
        _, curr_dists = self.processor.process(obs)
        _, next_dists = self.processor.process(next_obs)
        
        new_reward = reward
        
        # Penalty for hitting obstacles
        if (action == 0 and obs_s == 1) or \
           (action == 1 and obs_n == 1) or \
           (action == 2 and obs_e == 1) or \
           (action == 3 and obs_w == 1):
            new_reward -= 15.0
        
        # Reward for clear path
        if obs_n == 0 and obs_s == 0 and obs_e == 0 and obs_w == 0:
            new_reward += 15
        
        # Progress toward/away from nearest station
        min_curr = min(curr_dists)
        min_next = min(next_dists)
        
        # Reward for getting closer to a station when no passenger
        if pass_loc == 0:
            if min_next < min_curr:
                new_reward += 1.0
            elif min_next > min_curr:
                new_reward -= 1.0
        
        # Stronger reward when carrying passenger
        if pass_loc == 1:
            if min_next < min_curr:
                new_reward += 2.0
            elif min_next > min_curr:
                new_reward -= 2.0
        
        # Reward for successful pickup
        if action == 4 and pass_loc == 1 and next_pass_loc == 1:
            new_reward += 5.0
        
        # Penalties for invalid actions
        if action == 4 and pass_loc == 0:
            new_reward -= 1.0
        
        if action == 5 and dest_loc == 0:
            new_reward -= 1.0
        
        # Penalty for hitting walls or obstacles
        if action < 4 and t_row == next_t_row and t_col == next_t_col:
            new_reward -= 0.2
        
        return new_reward

class Agent:
    """RL agent with DQN"""
    def __init__(self, config):
        self.cfg = config
        self.device = config["device"]
        self.processor = StateProcessor(self.device)
        self.reward_shaper = RewardShaper(self.processor)
        
        # Networks
        self.policy_net = Network(11, 6).to(self.device)
        self.target_net = Network(11, 6).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config["lr"])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=200, verbose=True
        )
        self.criterion = nn.HuberLoss(delta=1.0)
        
        # Memory
        self.memory = Memory(config["buffer_size"])
        
        # Training params
        self.epsilon = config["eps_start"]
        self.eps_min = config["eps_min"]
        self.eps_decay = config["eps_decay"]
        self.gamma = config["gamma"]
        self.batch_size = config["batch_size"]
        self.tau = config["tau"]
        self.best_reward = -float('inf')
    
    def select_action(self, state):
        """Select action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            return random.choice([0, 1, 2, 3, 4, 5])
        else:
            with torch.no_grad():
                q_values = self.policy_net(state)
            return torch.argmax(q_values).item()
    
    def update_target(self):
        """Soft update target network"""
        for t_param, p_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            t_param.data.copy_(self.tau * p_param.data + (1.0 - self.tau) * t_param.data)
    
    def learn(self):
        """Learn from experience replay"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.get_batch(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q values
        curr_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute next Q values (Double DQN)
        with torch.no_grad():
            next_actions = self.policy_net(next_states).max(1)[1]
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Compute loss and update
        loss = self.criterion(curr_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step(loss)
    
    def train(self, episodes):
        """Train the agent"""
        from simple_custom_taxi_env import SimpleTaxiEnv
        env = SimpleTaxiEnv()
        
        for episode in tqdm(range(episodes)):
            obs, _ = env.reset()
            state_tensor, _ = self.processor.process(obs)
            done = False
            total_reward = 0
            
            # Episode loop
            while not done:
                # Select action
                action = self.select_action(state_tensor)
                
                # Take action
                next_obs, reward, done, _ = env.step(action)
                next_state_tensor, _ = self.processor.process(next_obs)
                
                # Shape reward
                shaped_reward = self.reward_shaper.shape(obs, next_obs, action, reward)
                
                # Store experience
                self.memory.store(
                    state_tensor.cpu().numpy(),
                    action,
                    shaped_reward,
                    next_state_tensor.cpu().numpy(),
                    done
                )
                
                # Update state
                total_reward += reward
                obs = next_obs
                state_tensor = next_state_tensor
                
                # Learn
                self.learn()
            
            # Decay epsilon
            self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)
            
            # Update target network
            if (episode + 1) % self.cfg["target_update_freq"] == 0:
                self.update_target()
            
            # Save best model
            if total_reward > self.best_reward:
                self.best_reward = total_reward
                torch.save(self.policy_net.state_dict(), self.cfg["model_path"])
    
    def load_model(self, path=None):
        """Load a trained model"""
        if path is None:
            path = self.cfg["model_path"]
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
    
    def get_action(self, obs):
        """Get action from observation (for testing)"""
        state_tensor, _ = self.processor.process(obs)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return torch.argmax(q_values).item()

def main():
    """Main function to train agent"""
    agent = Agent(CFG)
    agent.train(CFG["episodes"])

# For student_agent.py compatibility
def get_action(obs):
    """Function to get action from observation (for submission)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = StateProcessor(device)
    
    # Create and load model
    model = Network(11, 6).to(device)
    model.load_state_dict(torch.load("dqn.pt", map_location=device))
    model.eval()
    
    # Process state and get action
    state_tensor, _ = processor.process(obs)
    with torch.no_grad():
        q_values = model(state_tensor)
    return torch.argmax(q_values).item()

if __name__ == "__main__":
    main()