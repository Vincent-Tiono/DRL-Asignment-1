'''
import torch
import torch.nn as nn
import torch.optim as optim
import os
import random

def preprocess_state(obs):
    taxi_row, taxi_col, station0_row, station0_col, station1_row, station1_col, \
    station2_row, station2_col, station3_row, station3_col, \
    obstacle_north, obstacle_south, obstacle_east, obstacle_west, \
    passenger_look, destination_look = obs
    
    station_positions = [
        (station0_row, station0_col),
        (station1_row, station1_col),
        (station2_row, station2_col),
        (station3_row, station3_col)
    ]
    
    distances_to_stations = [
        (abs(taxi_row - row) + abs(taxi_col - col)) / 20.0
        for row, col in station_positions
    ]
    
    features = [
        obstacle_north, obstacle_south, obstacle_east, obstacle_west,
        passenger_look, destination_look,
        *distances_to_stations,
        min(distances_to_stations)
    ]
    
    return torch.FloatTensor(features).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

class DQNwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

EPSILON = 0.1  # Exploration probability
def get_action(obs):
    if random.random() < EPSILON:
        return random.choice([0, 1, 2, 3, 4, 5])
    
    state_tensor = preprocess_state(obs)
    with torch.no_grad():
        q_values = get_action.model(state_tensor)
    
    return torch.argmax(q_values).item()

# Load model only once
if not hasattr(get_action, "model"):
    with open("dqn.pt", "rb") as f:
        get_action.model = DQNwork(11, 6).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        get_action.model.load_state_dict(torch.load(f, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
        get_action.model.eval()
'''
'''
# working = -1607.94, torch
import torch
import torch.nn as nn
import os
import random

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "dqn.pt"
EPS = 0.05  # Reduced exploration for evaluation

class DQN(nn.Module):
    """Q-Network for DQN algorithm"""
    def __init__(self, in_dim, out_dim, hidden_dim=128):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x):
        return self.network(x)

def extract_features(obs):
    """Extract features from raw observation"""
    t_row, t_col, s0_row, s0_col, s1_row, s1_col, s2_row, s2_col, s3_row, s3_col, \
    obs_n, obs_s, obs_e, obs_w, pass_stat, dest_stat = obs
    
    # Calculate distances to stations
    stations = [
        (s0_row, s0_col),
        (s1_row, s1_col),
        (s2_row, s2_col),
        (s3_row, s3_col)
    ]
    
    dists = [
        (abs(t_row - row) + abs(t_col - col)) / 20.0
        for row, col in stations
    ]
    
    # Combine features
    features = [
        obs_n, obs_s, obs_e, obs_w,
        pass_stat, dest_stat,
        *dists,
        min(dists)
    ]
    
    return torch.FloatTensor(features).to(DEVICE)

# Main evaluation function
def get_action(obs):
    """Get action based on observation"""
    # Epsilon-greedy exploration
    if random.random() < EPS:
        return random.choice([0, 1, 2, 3, 4, 5])
    
    # Extract features
    state = extract_features(obs)
    
    # Get action from model
    with torch.no_grad():
        q_values = get_action.model(state)
    
    return torch.argmax(q_values).item()

# Load model only once
if not hasattr(get_action, "model"):
    with open("dqn.pt", "rb") as f:
        get_action.model = DQN(11, 6).to(DEVICE)
        get_action.model.load_state_dict(torch.load(f, map_location=DEVICE))
        get_action.model.eval()
'''

import numpy as np
import pickle
import random
import gym

with open("q_table.pkl", "rb") as f:
    Q_table = pickle.load(f)

def get_action(obs):
    # Extract relevant features to form a key.
    taxi_row, taxi_col, _, _, _, _, _, _, _, _, obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs
    state = (obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look)
    
    # Look up Q-values for this state if available; otherwise, use fallback.
    if state in Q_table:
        q_values = Q_table[state]
    else:
        # Fallback: return neutral Q-values if state not found.
        q_values = [0.0] * 6
        
    return int(np.argmax(q_values))


''' 
Working = -8005

import torch
import torch.nn as nn
import numpy as np
import random

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "ddqn.pt"

class DQN(nn.Module):
    """Dueling Double DQN with noisy layers"""
    def __init__(self, in_dim, out_dim, hidden_dim=128):
        super(DQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.apply(self._initialize_weights)
    
    def _initialize_weights(self, mod):
        if isinstance(mod, nn.Linear):
            nn.init.kaiming_normal_(mod.weight, nonlinearity='relu')
            if mod.bias is not None:
                nn.init.constant_(mod.bias, 0.0)
    
    def forward(self, x):
        return self.feature(x)
    

def process_observation(obs):
    taxi_row, taxi_col, s0r, s0c, s1r, s1c, s2r, s2c, s3r, s3c, \
    obs_n, obs_s, obs_e, obs_w, pass_stat, dest_stat = obs
    
    stations = [(s0r, s0c), (s1r, s1c), (s2r, s2c), (s3r, s3c)]
    dists = [abs(taxi_row-r) + abs(taxi_col-c) for r,c in stations]
    
    # Exactly 17 features to match training
    features = [
        taxi_row/4.0, taxi_col/4.0,
        obs_n, obs_s, obs_e, obs_w,
        pass_stat, dest_stat,
        *[d/8.0 for d in dists],
        min(dists)/8.0,
        (taxi_row - s0r)/4.0, (taxi_col - s0c)/4.0,
        (taxi_row - s1r)/4.0, (taxi_col - s1c)/4.0
    ]
    
    return torch.FloatTensor(features).to(DEVICE).unsqueeze(0)  # Add batch dimension

def get_action(obs):
    state = process_observation(obs)
    with torch.no_grad():
        q_values = get_action.model(state)
    return torch.argmax(q_values).item()

# Load model once
if not hasattr(get_action, "model"):
    model = DQN(17, 6).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    get_action.model = model
'''
