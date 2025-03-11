# Remember to adjust your student ID in meta.xml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from tqdm import tqdm

# Global variables
MODEL_FILE = "q_network.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
EPSILON = 0.1  # For exploration during testing

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

        # Initialize weights with custom initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # He initialization for weights (good for ReLU activations)
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            # Initialize bias to small random values
            if module.bias is not None:
                nn.init.uniform_(module.bias, -0.05, 0.05)
    
    def forward(self, x):
        return self.network(x)

# Add a ReplayBuffer class for experience replay
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
    """
    Convert observation to a tensor for the neural network.
    This function uses relative positions and distances between key elements.
    """
    # Extract key components from the observation
    taxi_row, taxi_col, station0_row, station0_col, station1_row, station1_col, \
    station2_row, station2_col, station3_row, station3_col, \
    obstacle_north, obstacle_south, obstacle_east, obstacle_west, \
    passenger_look, destination_look = obs
    
    # Calculate distances to stations (these will be useful regardless of which one has the passenger/destination)
    station_positions = [
        (station0_row, station0_col),
        (station1_row, station1_col),
        (station2_row, station2_col),
        (station3_row, station3_col)
    ]
    
    # Calculate Manhattan distances from taxi to each station
    distances_to_stations = []
    for station_row, station_col in station_positions:
        manhattan_dist = abs(taxi_row - station_row) + abs(taxi_col - station_col)
        # Normalize by dividing by a reasonable maximum distance (e.g., grid size)
        # Assuming grid is no larger than 10x10
        normalized_dist = manhattan_dist / 20.0  
        distances_to_stations.append(normalized_dist)
    
    # Create feature vector with more meaningful relative information
    features = [
        # Obstacle information (binary)
        obstacle_north,
        obstacle_south, 
        obstacle_east,
        obstacle_west,
        
        # Passenger and destination information (binary)
        passenger_look,
        destination_look,
        
        # Distances to all stations (continuous, normalized)
        distances_to_stations[0],
        distances_to_stations[1],
        distances_to_stations[2],
        distances_to_stations[3],
        
        # Minimum distance to any station (helps with general navigation)
        min(distances_to_stations)
    ]
    
    return torch.FloatTensor(features).to(DEVICE), distances_to_stations

def get_action(obs):
    """
    Takes an observation as input and returns an action (0-5).
    Uses the trained Q-network to select the best action.
    """
    # Load model if it exists and hasn't been loaded yet
    if not hasattr(get_action, "model"):
        if os.path.exists(MODEL_FILE):
            get_action.model = QNetwork(11, 6).to(DEVICE)
            get_action.model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
            get_action.model.eval()
        else:
            # If model doesn't exist, return random actions
            return random.choice([0, 1, 2, 3, 4, 5])
    
    # Epsilon-greedy policy for exploration during testing
    if random.random() < EPSILON:
        return random.choice([0, 1, 2, 3, 4, 5])
    
    # Preprocess state and get Q-values
    state_tensor, _ = preprocess_state(obs)
    with torch.no_grad():
        q_values = get_action.model(state_tensor)
    
    # Return action with highest Q-value
    return torch.argmax(q_values).item()

    # return random.choice([0, 1, 2, 3, 4, 5])

def shape_reward(obs, next_obs, action, reward):
    """
    Apply reward shaping to encourage more efficient behavior based on distance metrics.
    """
    # Extract information from observations
    taxi_row, taxi_col, station0_row, station0_col, station1_row, station1_col, \
    station2_row, station2_col, station3_row, station3_col, \
    obstacle_north, obstacle_south, obstacle_east, obstacle_west, \
    passenger_look, destination_look = obs
    
    next_taxi_row, next_taxi_col, _, _, _, _, _, _, _, _, \
    next_obstacle_north, next_obstacle_south, next_obstacle_east, next_obstacle_west, \
    next_passenger_look, next_destination_look = next_obs
    
    # Get distances to stations for current and next state
    _, current_distances = preprocess_state(obs)
    _, next_distances = preprocess_state(next_obs)
    
    shaped_reward = reward
    
    # Penalty for trying to move into obstacles
    if (action == 0 and obstacle_south == 1) or \
       (action == 1 and obstacle_north == 1) or \
       (action == 2 and obstacle_east == 1) or \
       (action == 3 and obstacle_west == 1):
        shaped_reward -= 15.0
    
    # Reward for being in open space (no obstacles)
    if obstacle_north == 0 and obstacle_south == 0 and obstacle_east == 0 and obstacle_west == 0:
        shaped_reward += 15  # Small bonus for being in open space
    
    # Find the minimum distance to any station in current and next state
    min_current_distance = min(current_distances)
    min_next_distance = min(next_distances)
    
    # If not carrying a passenger, reward for getting closer to any station
    if passenger_look == 0:
        # If we're getting closer to the nearest station
        if min_next_distance < min_current_distance:
            shaped_reward += 1.0
        # If we're moving away from all stations
        elif min_next_distance > min_current_distance:
            shaped_reward -= 1.0
    
    # If carrying a passenger, reward for getting closer to destination
    if passenger_look == 1:
        # If we're getting closer to any station (potential destination)
        if min_next_distance < min_current_distance:
            shaped_reward += 2.0
        # If we're moving away from all stations
        elif min_next_distance > min_current_distance:
            shaped_reward -= 2.0
    
    # Reward for successful pickup
    if action == 4 and passenger_look == 1 and next_passenger_look == 1:
        shaped_reward += 5.0
    
    # Penalty for trying pickup when no passenger is present
    if action == 4 and passenger_look == 0:
        shaped_reward -= 1.0
    
    # Penalty for trying dropoff when not at destination
    if action == 5 and destination_look == 0:
        shaped_reward -= 1.0
    
    # Penalty for not moving (staying in the same position)
    if action < 4 and taxi_row == next_taxi_row and taxi_col == next_taxi_col:
        shaped_reward -= 0.2
    
    return shaped_reward

def soft_update(target_net, policy_net, tau=0.001):
    """Soft update of target network: θ′ ← τθ + (1 − τ)θ′"""
    for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

def train_agent(num_episodes=10000, gamma=0.99, batch_size=64):
    """
    Train the agent using DQN with experience replay and target network.
    """
    from simple_custom_taxi_env import SimpleTaxiEnv
    
    # Initialize environment
    env = SimpleTaxiEnv()
    
    # Initialize Q-networks (policy and target)
    policy_net = QNetwork(11, 6).to(DEVICE)
    target_net = QNetwork(11, 6).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()  # Target network is only used for inference
    
    # Initialize optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200, verbose=True)
    criterion = nn.HuberLoss(delta=1.0)
    
    # Initialize replay buffer
    replay_buffer = ReplayBuffer(capacity=50000)
    
    # Training parameters
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.9999
    target_update_frequency = 5  # Update target network every N episodes
    
    # Training loop
    best_reward = -float('inf')
    losses = []
    episode_rewards = []
    
    for episode in tqdm(range(num_episodes)):
        obs, _ = env.reset()
            
        state_tensor, _ = preprocess_state(obs)  # Only use the tensor, ignore distances
        done = False
        total_reward = 0
        episode_losses = []
        steps_rewards = []
        
        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.choice([0, 1, 2, 3, 4, 5])
            else:
                with torch.no_grad():
                    q_values = policy_net(state_tensor)
                action = torch.argmax(q_values).item()
            
            # Take action and observe next state
            next_obs, reward, done, _, _ = env.step(action)
                
            next_state_tensor, _ = preprocess_state(next_obs)  # Only use the tensor
            
            # Apply reward shaping
            shaped_reward = shape_reward(obs, next_obs, action, reward)
            
            # Store transition in replay buffer
            replay_buffer.push(
                state_tensor.cpu().numpy(),
                action,
                shaped_reward,
                next_state_tensor.cpu().numpy(),
                done
            )
            
            total_reward += reward  # Track original reward for evaluation
            
            # Move to next state
            obs = next_obs
            state_tensor = next_state_tensor
            
            # Train on a batch of transitions if buffer has enough samples
            if len(replay_buffer) >= batch_size:
                # Sample a batch from replay buffer
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                
                # Convert to tensors
                states = torch.FloatTensor(states).to(DEVICE)
                actions = torch.LongTensor(actions).to(DEVICE)
                rewards = torch.FloatTensor(rewards).to(DEVICE)
                next_states = torch.FloatTensor(next_states).to(DEVICE)
                dones = torch.FloatTensor(dones).to(DEVICE)
                
                # Compute current Q values
                current_q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                
                # Double DQN: Use policy network to select actions and target network to evaluate them
                with torch.no_grad():
                    # Select actions using policy network
                    next_action_indices = policy_net(next_states).max(1)[1]
                    # Evaluate Q-values for those actions using target network
                    next_q_values = target_net(next_states).gather(1, next_action_indices.unsqueeze(1)).squeeze(1)
                    target_q_values = rewards + gamma * next_q_values * (1 - dones)
                
                # Compute loss and update
                loss = criterion(current_q_values, target_q_values)
                
                # Gradient clipping to prevent exploding gradients
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Update learning rate scheduler
                scheduler.step(loss)
                
                # Track loss
                episode_losses.append(loss.item())
                steps_rewards.append(total_reward)

        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        # Update target network periodically
        if (episode + 1) % target_update_frequency == 0:
            soft_update(target_net, policy_net)
        
        # Track metrics
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        avg_reward = np.mean(steps_rewards) if steps_rewards else 0
        losses.append(avg_loss)
        episode_rewards.append(total_reward)
        
        # Track best reward
        if total_reward > best_reward:
            best_reward = total_reward
            # Save best model
            torch.save(policy_net.state_dict(), MODEL_FILE)
        
        # Print progress
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Average Reward: {avg_reward:.2f}, Best: {best_reward:.2f}, Average Loss: {avg_loss:.6f}, Epsilon: {epsilon:.4f}")
    
    print("Training completed and model saved.")

if __name__ == "__main__":
    # This will only run when you execute this file directly
    train_agent(num_episodes=10000)

# else:
#     model_path = "q_network.pt"
#     # TODO: load the model
#     model = QNetwork(8, 6).to(DEVICE)
#     model.load_state_dict(torch.load(model_path, map_location=DEVICE))
#     model.eval()