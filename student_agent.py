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
# working = -1607.94, train.py
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
'''Before 3/20'''
'''
import torch
import torch.nn as nn
import os
import random

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "dqn_0320.pt"  # This model must be trained with state_dim=12
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

def extract_features(obs, passenger_on=0):
    """Extract features from raw observation and append passenger flag.
    
    Expected obs (16 elements):
        taxi_row, taxi_col,
        s0_row, s0_col,
        s1_row, s1_col,
        s2_row, s2_col,
        s3_row, s3_col,
        obs_n, obs_s, obs_e, obs_w,
        pass_stat, dest_stat
    
    We compute:
      - Four obstacle indicators: obs_n, obs_s, obs_e, obs_w
      - Two additional env flags: pass_stat, dest_stat
      - Four normalized Manhattan distances (each divided by 20.0)
      - The minimum of these distances
      - The passenger_on flag (0 or 1)
      
    Total state dimension = 4 + 2 + 4 + 1 + 1 = 12.
    """
    t_row, t_col, s0_row, s0_col, s1_row, s1_col, s2_row, s2_col, s3_row, s3_col, \
    obs_n, obs_s, obs_e, obs_w, pass_stat, dest_stat = obs
    
    # Calculate Manhattan distances to the four stations
    stations = [
        (s0_row, s0_col),
        (s1_row, s1_col),
        (s2_row, s2_col),
        (s3_row, s3_col)
    ]
    dists = [(abs(t_row - row) + abs(t_col - col)) / 20.0 for row, col in stations]
    
    features = [
        obs_n, obs_s, obs_e, obs_w,    # obstacles
        pass_stat, dest_stat,          # env flags for passenger & destination
        *dists,                       # 4 distances
        min(dists),                   # minimum distance
        passenger_on                # extra flag indicating if taxi is carrying a passenger
    ]
    
    return torch.FloatTensor(features).to(DEVICE)

def get_action(obs, passenger_on=0):
    """Get action based on observation.
    
    Includes an optional passenger_on parameter (default 0) to match the 12-dimensional state.
    Uses epsilon-greedy exploration.
    """
    if random.random() < EPS:
        return random.choice([0, 1, 2, 3, 4, 5])
    
    state = extract_features(obs, passenger_on)
    with torch.no_grad():
        q_values = get_action.model(state)
    return torch.argmax(q_values).item()

# Load model only once
if not hasattr(get_action, "model"):
    with open(MODEL_PATH, "rb") as f:
        # Notice state dimension is now 12 instead of 11!
        get_action.model = DQN(12, 6).to(DEVICE)
        get_action.model.load_state_dict(torch.load(f, map_location=DEVICE))
        get_action.model.eval()
'''


import torch
import torch.nn as nn
import os
import random

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "dqn_0320.pt"  # This model must be trained with state_dim=12
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

def extract_features(obs):  # No passenger_on
    t_row, t_col, s0_row, s0_col, s1_row, s1_col, s2_row, s2_col, s3_row, s3_col, \
    obs_n, obs_s, obs_e, obs_w, pass_stat, dest_stat = obs
    
    stations = [(s0_row, s0_col), (s1_row, s1_col), (s2_row, s2_col), (s3_row, s3_col)]
    dists = [(abs(t_row - row) + abs(t_col - col)) / 20.0 for row, col in stations]
    
    features = [
        obs_n, obs_s, obs_e, obs_w,    # 4
        pass_stat, dest_stat,          # 2
        *dists,                       # 4
        min(dists)                    # 1
    ]  # Total = 11
    
    return torch.FloatTensor(features).to(DEVICE)

def get_action(obs):
    if random.random() < EPS:
        return random.choice([0, 1, 2, 3, 4, 5])
    state = extract_features(obs)
    with torch.no_grad():
        q_values = get_action.model(state)
    return torch.argmax(q_values).item()

# Load model
if not hasattr(get_action, "model"):
    with open(MODEL_PATH, "rb") as f:
        get_action.model = DQN(11, 6).to(DEVICE)  # Match training: in_dim=11
        get_action.model.load_state_dict(torch.load(f, map_location=DEVICE))
        get_action.model.eval()