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

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
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
        get_action.model = QNetwork(11, 6).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        get_action.model.load_state_dict(torch.load(f, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
        get_action.model.eval()
'''
''''
working = -1607.94
import torch
import torch.nn as nn
import os
import random

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "dqn.pt"
EPS = 0.05  # Reduced exploration for evaluation

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
        get_action.model = QNet(11, 6).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        get_action.model.load_state_dict(torch.load(f, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
        get_action.model.eval()''
'''

import torch
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "dueling_ddqn.pt"

class DuelingQNetwork(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.feature = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU()
        )
        self.value = torch.nn.Sequential(
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )
        self.advantage = torch.nn.Sequential(
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        x = self.feature(x)
        value = self.value(x)
        advantage = self.advantage(x)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

def get_action(obs):
    if not hasattr(get_action, 'model'):
        get_action.model = DuelingQNetwork(16, 6).to(DEVICE)
        get_action.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        get_action.model.eval()
    
    # Process observation
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
    
    state_t = torch.FloatTensor(features).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        q_values = get_action.model(state_t)
    return torch.argmax(q_values).item()