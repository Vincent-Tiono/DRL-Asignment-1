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
    with open("dqnetwork.pt", "rb") as f:
        get_action.model = QNetwork(11, 6).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        get_action.model.load_state_dict(torch.load(f, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
        get_action.model.eval()