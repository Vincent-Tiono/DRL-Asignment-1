import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from collections import deque
import random

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "dqn.pt"

class DQN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )
        self._init_weights()
    
    def _init_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                nn.init.constant_(layer.bias, 0.1)
    
    def forward(self, x):
        return self.layers(x)

class Memory:
    def __init__(self, cap=100000):
        self.buffer = deque(maxlen=cap)
    
    def add(self, state, act, rew, next_state, done):
        self.buffer.append((state, act, rew, next_state, done))
    
    def sample(self, size):
        batch = random.sample(self.buffer, min(size, len(self.buffer)))
        s, a, r, ns, d = zip(*batch)
        return np.stack(s), np.stack(a), np.stack(r), np.stack(ns), np.stack(d)
    
    def __len__(self):
        return len(self.buffer)

class Agent:
    def __init__(self, in_dim, out_dim):
        self.policy = DQN(in_dim, out_dim).to(DEVICE)
        self.target = DQN(in_dim, out_dim).to(DEVICE)
        self.target.load_state_dict(self.policy.state_dict())
        self.optim = optim.AdamW(self.policy.parameters(), lr=3e-4, weight_decay=1e-5)
        self.loss_fn = nn.SmoothL1Loss()
        self.memory = Memory()
        self.eps = 1.0
        self.eps_min = 0.01
        self.eps_decay = 0.9995
        self.gamma = 0.995
        self.batch = 128
        self.tau = 0.005
    
    def get_action(self, state):
        if random.random() < self.eps:
            return random.randint(0, 5)
        with torch.no_grad():
            q = self.policy(state)
            return q.argmax().item()
    
    def decay_eps(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)
    
    def update(self):
        if len(self.memory) < self.batch:
            return
        
        s, a, r, ns, d = self.memory.sample(self.batch)
        s = torch.FloatTensor(s).to(DEVICE)
        a = torch.LongTensor(a).to(DEVICE)
        r = torch.FloatTensor(r).to(DEVICE)
        ns = torch.FloatTensor(ns).to(DEVICE)
        d = torch.FloatTensor(d).to(DEVICE)
        
        curr_q = self.policy(s).gather(1, a.unsqueeze(1)).squeeze()
        
        with torch.no_grad():
            next_acts = self.policy(ns).argmax(1)
            next_q = self.target(ns).gather(1, next_acts.unsqueeze(1)).squeeze()
            target_q = r + self.gamma * next_q * (1 - d)
        
        loss = self.loss_fn(curr_q, target_q)
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 2.0)
        self.optim.step()
        
        for t, p in zip(self.target.parameters(), self.policy.parameters()):
            t.data.copy_(self.tau*p.data + (1-self.tau)*t.data)
    
    def save(self):
        torch.save(self.policy.state_dict(), MODEL_PATH)

def process_state(obs):
    t_row, t_col, *stations, obst_n, obst_s, obst_e, obst_w, pass_loc, dest_loc = obs
    
    station_coords = [(stations[i], stations[i+1]) for i in range(0, 8, 2)]
    distances = [(abs(t_row - r) + abs(t_col - c)) / 20.0 for r, c in station_coords]
    
    features = [
        obst_n, obst_s, obst_e, obst_w,
        pass_loc, dest_loc,
        *distances,
        min(distances)
    ]
    
    return torch.FloatTensor(features).to(DEVICE), distances

def shape_reward(curr_obs, next_obs, act, rew):
    *_, obst_n, obst_s, obst_e, obst_w, pass_loc, _ = curr_obs
    next_t_row, next_t_col, *_ = next_obs
    
    # Obstacle penalty
    if (act == 0 and obst_s) or (act == 1 and obst_n) or \
       (act == 2 and obst_e) or (act == 3 and obst_w):
        rew -= 15
    
    # Open space bonus
    if not any([obst_n, obst_s, obst_e, obst_w]):
        rew += 15
    
    # Distance-based shaping
    _, curr_dists = process_state(curr_obs)
    _, next_dists = process_state(next_obs)
    curr_min = min(curr_dists)
    next_min = min(next_dists)
    
    if pass_loc == 0:
        rew += 1.0 if next_min < curr_min else -1.0
    else:
        rew += 2.0 if next_min < curr_min else -2.0
    
    # Action penalties
    if act == 4:  # Pickup
        rew += 5 if pass_loc == 1 else -1
    elif act == 5:  # Dropoff
        rew -= 1 if pass_loc != 1 else 0
    
    # Stationary penalty
    if act < 4 and (curr_obs[0], curr_obs[1]) == (next_t_row, next_t_col):
        rew -= 0.2
    
    return rew

class Trainer:
    def __init__(self, episodes=15000):
        from simple_custom_taxi_env import SimpleTaxiEnv
        self.env = SimpleTaxiEnv()
        self.agent = Agent(11, 6)
        self.episodes = episodes
        self.best = -float('inf')
    
    def run(self):
        for _ in tqdm(range(self.episodes)):
            obs, _ = self.env.reset()
            state, _ = process_state(obs)
            done = False
            total = 0
            
            while not done:
                act = self.agent.get_action(state)
                next_obs, rew, done, _ = self.env.step(act)
                next_state, _ = process_state(next_obs)
                shaped_rew = shape_reward(obs, next_obs, act, rew)
                
                self.agent.memory.add(
                    state.cpu().numpy(),
                    act,
                    shaped_rew,
                    next_state.cpu().numpy(),
                    done
                )
                
                state = next_state
                obs = next_obs
                total += rew
                self.agent.update()
            
            self.agent.decay_eps()
            if total > self.best:
                self.best = total
                self.agent.save()

if __name__ == "__main__":
    trainer = Trainer()
    trainer.run()