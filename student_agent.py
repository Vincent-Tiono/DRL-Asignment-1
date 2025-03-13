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
    
    # Calculate Manhattan distances to each station
    stations = [
        (s0_row, s0_col),
        (s1_row, s1_col),
        (s2_row, s2_col),
        (s3_row, s3_col)
    ]
    dists = [(abs(t_row - row) + abs(t_col - col)) / 20.0 for row, col in stations]
    
    # Combine features: obstacles, flags, 4 distances, and the minimum of these distances
    features = [
        obs_n, obs_s, obs_e, obs_w,
        pass_stat, dest_stat,
        *dists,
        min(dists)
    ]
    
    return torch.FloatTensor(features).to(DEVICE)

def get_action(obs):
    """Get action based on observation using epsilon-greedy selection."""
    # With probability EPS take a random action
    if random.random() < EPS:
        return random.choice([0, 1, 2, 3, 4, 5])
    
    # Otherwise, use the network's output
    state = extract_features(obs)
    with torch.no_grad():
        q_values = get_action.model(state)
    return torch.argmax(q_values).item()

# Load model only once
if not hasattr(get_action, "model"):
    with open(MODEL_PATH, "rb") as f:
        get_action.model = DQN(11, 6).to(DEVICE)
        get_action.model.load_state_dict(torch.load(f, map_location=DEVICE))
        get_action.model.eval()