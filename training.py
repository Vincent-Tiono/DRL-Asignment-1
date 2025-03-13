import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm

# Global variables
MODEL_FILE = "dqn.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.uniform_(module.bias, -0.05, 0.05)
    
    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

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
    
    distances_to_stations = []
    for station_row, station_col in station_positions:
        manhattan_dist = abs(taxi_row - station_row) + abs(taxi_col - station_col)
        normalized_dist = manhattan_dist / 20.0  
        distances_to_stations.append(normalized_dist)
    
    features = [
        obstacle_north,
        obstacle_south, 
        obstacle_east,
        obstacle_west,
        passenger_look,
        destination_look,
        distances_to_stations[0],
        distances_to_stations[1],
        distances_to_stations[2],
        distances_to_stations[3],
        min(distances_to_stations)
    ]
    
    return torch.FloatTensor(features).to(DEVICE), distances_to_stations

def shape_reward(obs, next_obs, action, reward):
    taxi_row, taxi_col, station0_row, station0_col, station1_row, station1_col, \
    station2_row, station2_col, station3_row, station3_col, \
    obstacle_north, obstacle_south, obstacle_east, obstacle_west, \
    passenger_look, destination_look = obs
    
    next_taxi_row, next_taxi_col, _, _, _, _, _, _, _, _, \
    next_obstacle_north, next_obstacle_south, next_obstacle_east, next_obstacle_west, \
    next_passenger_look, next_destination_look = next_obs
    
    _, current_distances = preprocess_state(obs)
    _, next_distances = preprocess_state(next_obs)
    
    shaped_reward = reward
    
    if (action == 0 and obstacle_south == 1) or \
       (action == 1 and obstacle_north == 1) or \
       (action == 2 and obstacle_east == 1) or \
       (action == 3 and obstacle_west == 1):
        shaped_reward -= 15.0
    
    if obstacle_north == 0 and obstacle_south == 0 and obstacle_east == 0 and obstacle_west == 0:
        shaped_reward += 15
    
    min_current_distance = min(current_distances)
    min_next_distance = min(next_distances)
    
    if passenger_look == 0:
        if min_next_distance < min_current_distance:
            shaped_reward += 1.0
        elif min_next_distance > min_current_distance:
            shaped_reward -= 1.0
    
    if passenger_look == 1:
        if min_next_distance < min_current_distance:
            shaped_reward += 2.0
        elif min_next_distance > min_current_distance:
            shaped_reward -= 2.0
    
    if action == 4 and passenger_look == 1 and next_passenger_look == 1:
        shaped_reward += 5.0
    
    if action == 4 and passenger_look == 0:
        shaped_reward -= 1.0
    
    if action == 5 and destination_look == 0:
        shaped_reward -= 1.0
    
    if action < 4 and taxi_row == next_taxi_row and taxi_col == next_taxi_col:
        shaped_reward -= 0.2
    
    return shaped_reward

def soft_update(target_net, policy_net, tau=0.001):
    for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

def train_agent(num_episodes=10000, gamma=0.99, batch_size=64):
    from simple_custom_taxi_env import SimpleTaxiEnv
    
    env = SimpleTaxiEnv()
    policy_net = DQN(11, 6).to(DEVICE)
    target_net = DQN(11, 6).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200, verbose=True)
    criterion = nn.HuberLoss(delta=1.0)
    replay_buffer = ReplayBuffer(capacity=50000)
    
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.9999
    target_update_frequency = 5
    
    best_reward = -float('inf')
    
    for episode in tqdm(range(num_episodes)):
        obs, _ = env.reset()
        state_tensor, _ = preprocess_state(obs)
        done = False
        total_reward = 0
        
        while not done:
            if random.random() < epsilon:
                action = random.choice([0, 1, 2, 3, 4, 5])
            else:
                with torch.no_grad():
                    q_values = policy_net(state_tensor)
                action = torch.argmax(q_values).item()
            
            next_obs, reward, done, _ = env.step(action)
            next_state_tensor, _ = preprocess_state(next_obs)
            shaped_reward = shape_reward(obs, next_obs, action, reward)
            
            replay_buffer.push(
                state_tensor.cpu().numpy(),
                action,
                shaped_reward,
                next_state_tensor.cpu().numpy(),
                done
            )
            
            total_reward += reward
            obs = next_obs
            state_tensor = next_state_tensor
            
            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                
                states = torch.FloatTensor(states).to(DEVICE)
                actions = torch.LongTensor(actions).to(DEVICE)
                rewards = torch.FloatTensor(rewards).to(DEVICE)
                next_states = torch.FloatTensor(next_states).to(DEVICE)
                dones = torch.FloatTensor(dones).to(DEVICE)
                
                current_q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                
                with torch.no_grad():
                    next_action_indices = policy_net(next_states).max(1)[1]
                    next_q_values = target_net(next_states).gather(1, next_action_indices.unsqueeze(1)).squeeze(1)
                    target_q_values = rewards + gamma * next_q_values * (1 - dones)
                
                loss = criterion(current_q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step(loss)

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        if (episode + 1) % target_update_frequency == 0:
            soft_update(target_net, policy_net)
        
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(policy_net.state_dict(), MODEL_FILE)

if __name__ == "__main__":
    train_agent(num_episodes=1000)