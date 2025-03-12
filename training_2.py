import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm

# Global variables
MODEL_PATH = "dqn.pt"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )
        self.apply(self._init_weights)
    
    def _init_weights(self, mod):
        if isinstance(mod, nn.Linear):
            nn.init.kaiming_uniform_(mod.weight, nonlinearity='relu')
            if mod.bias is not None:
                nn.init.uniform_(mod.bias, -0.05, 0.05)
    
    def forward(self, x):
        return self.net(x)

class Memory:
    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.buffer = []
        self.pos = 0
        
    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.max_size:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.pos = (self.pos + 1) % self.max_size
        
    def get_batch(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

def extract_features(obs):
    # Unpack observation
    t_row, t_col, s0_row, s0_col, s1_row, s1_col, \
    s2_row, s2_col, s3_row, s3_col, \
    obs_n, obs_s, obs_e, obs_w, \
    pass_stat, dest_stat = obs
    
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
        manhattan = abs(t_row - s_row) + abs(t_col - s_col)
        norm_dist = manhattan / 20.0  
        dists.append(norm_dist)
    
    # Create feature vector
    feats = [
        obs_n,
        obs_s, 
        obs_e,
        obs_w,
        pass_stat,
        dest_stat,
        dists[0],
        dists[1],
        dists[2],
        dists[3],
        min(dists)
    ]
    
    return torch.FloatTensor(feats).to(DEV), dists

def calculate_reward(obs, next_obs, action, base_reward):
    # Unpack current observation
    t_row, t_col, s0_row, s0_col, s1_row, s1_col, \
    s2_row, s2_col, s3_row, s3_col, \
    obs_n, obs_s, obs_e, obs_w, \
    pass_stat, dest_stat = obs
    
    # Unpack next observation
    n_t_row, n_t_col, _, _, _, _, _, _, _, _, \
    n_obs_n, n_obs_s, n_obs_e, n_obs_w, \
    n_pass_stat, n_dest_stat = next_obs
    
    # Get distances
    _, curr_dists = extract_features(obs)
    _, next_dists = extract_features(next_obs)
    
    # Start with base reward
    mod_reward = base_reward
    
    # Penalize hitting obstacles
    if (action == 0 and obs_s == 1) or \
       (action == 1 and obs_n == 1) or \
       (action == 2 and obs_e == 1) or \
       (action == 3 and obs_w == 1):
        mod_reward -= 15.0
    
    # Reward open spaces
    if obs_n == 0 and obs_s == 0 and obs_e == 0 and obs_w == 0:
        mod_reward += 15
    
    # Get minimum distances
    min_curr = min(curr_dists)
    min_next = min(next_dists)
    
    # Reward/penalize for distance changes
    if pass_stat == 0:
        if min_next < min_curr:
            mod_reward += 1.0
        elif min_next > min_curr:
            mod_reward -= 1.0
    
    if pass_stat == 1:
        if min_next < min_curr:
            mod_reward += 2.0
        elif min_next > min_curr:
            mod_reward -= 2.0
    
    # Reward/penalize pickup/dropoff actions
    if action == 4 and pass_stat == 1 and n_pass_stat == 1:
        mod_reward += 5.0
    
    if action == 4 and pass_stat == 0:
        mod_reward -= 1.0
    
    if action == 5 and dest_stat == 0:
        mod_reward -= 1.0
    
    # Penalize no movement
    if action < 4 and t_row == n_t_row and t_col == n_t_col:
        mod_reward -= 0.2
    
    return mod_reward

def update_target(target, policy, tau=0.001):
    for t_param, p_param in zip(target.parameters(), policy.parameters()):
        t_param.data.copy_(tau * p_param.data + (1.0 - tau) * t_param.data)

def train_step(policy, target, opt, mem, batch_size, gamma, criterion):
    states, actions, rewards, next_states, dones = mem.get_batch(batch_size)
    
    states = torch.FloatTensor(states).to(DEV)
    actions = torch.LongTensor(actions).to(DEV)
    rewards = torch.FloatTensor(rewards).to(DEV)
    next_states = torch.FloatTensor(next_states).to(DEV)
    dones = torch.FloatTensor(dones).to(DEV)
    
    # Get current Q values
    curr_q = policy(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    
    # Get target Q values
    with torch.no_grad():
        next_actions = policy(next_states).max(1)[1]
        next_q = target(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target_q = rewards + gamma * next_q * (1 - dones)
    
    # Calculate loss and update
    loss = criterion(curr_q, target_q)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
    opt.step()
    
    return loss

def select_action(policy, state, epsilon):
    if random.random() < epsilon:
        return random.choice([0, 1, 2, 3, 4, 5])
    else:
        with torch.no_grad():
            q_values = policy(state)
        return torch.argmax(q_values).item()

def train_agent(episodes=10000, gamma=0.99, batch_size=64):
    from simple_custom_taxi_env import SimpleTaxiEnv
    
    # Initialize environment and networks
    env = SimpleTaxiEnv()
    policy = DQN(11, 6).to(DEV)
    target = DQN(11, 6).to(DEV)
    target.load_state_dict(policy.state_dict())
    target.eval()
    
    # Initialize training components
    opt = optim.Adam(policy.parameters(), lr=0.001)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=200, verbose=True)
    crit = nn.HuberLoss(delta=1.0)
    mem = Memory(max_size=50000)
    
    # Training parameters
    eps = 1.0
    eps_min = 0.01
    eps_decay = 0.9999
    update_freq = 5
    
    best_reward = -float('inf')
    
    # Main training loop
    for ep in tqdm(range(episodes)):
        obs, _ = env.reset()
        state, _ = extract_features(obs)
        done = False
        total_reward = 0
        
        # Episode loop
        while not done:
            # Select and execute action
            action = select_action(policy, state, eps)
            next_obs, reward, done, _ = env.step(action)
            next_state, _ = extract_features(next_obs)
            
            # Process rewards and store experience
            mod_reward = calculate_reward(obs, next_obs, action, reward)
            mem.add(
                state.cpu().numpy(),
                action,
                mod_reward,
                next_state.cpu().numpy(),
                done
            )
            
            # Update tracking variables
            total_reward += reward
            obs = next_obs
            state = next_state
            
            # Perform training if enough samples
            if len(mem) >= batch_size:
                loss = train_step(policy, target, opt, mem, batch_size, gamma, crit)
                sched.step(loss)

        # Update epsilon
        eps = max(eps_min, eps * eps_decay)
        
        # Update target network
        if (ep + 1) % update_freq == 0:
            update_target(target, policy)
        
        # Save best model
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(policy.state_dict(), MODEL_PATH)

def get_action(obs):
    # Load model
    model = DQN(11, 6).to(DEV)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEV))
    model.eval()
    
    # Process observation
    state_tensor, _ = extract_features(obs)
    
    # Get action
    with torch.no_grad():
        q_values = model(state_tensor)
    
    return torch.argmax(q_values).item()

if __name__ == "__main__":
    train_agent(episodes=1000)