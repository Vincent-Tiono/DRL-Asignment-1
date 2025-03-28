import torch
import torch.nn as nn
import os
import random

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "dqn.pt"  # This model must be trained with state_dim=12
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