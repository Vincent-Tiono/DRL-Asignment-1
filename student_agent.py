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

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        # Increased network capacity to handle more complex state representation
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
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
    This function uses relative positions and features that work regardless of grid size.
    """
    # Extract key components from the observation
    taxi_row, taxi_col, station0_row, station0_col, station1_row, station1_col, \
    station2_row, station2_col, station3_row, station3_col, \
    obstacle_north, obstacle_south, obstacle_east, obstacle_west, \
    passenger_look, destination_look = obs
    
    # Estimate grid size based on maximum coordinate values
    # This approach adapts to any grid size
    max_coord = max(taxi_row, taxi_col, station0_row, station0_col, 
                    station1_row, station1_col, station2_row, station2_col,
                    station3_row, station3_col)
    estimated_grid_size = max_coord + 1  # +1 because coordinates are 0-indexed
    
    # Create feature vector with normalized positions and relative distances
    features = [
        # Normalized taxi position
        taxi_row / max(estimated_grid_size, 1),
        taxi_col / max(estimated_grid_size, 1),
        
        # Relative distances to stations (normalized)
        # These work regardless of grid size
        abs(taxi_row - station0_row) / max(estimated_grid_size, 1),
        abs(taxi_col - station0_col) / max(estimated_grid_size, 1),
        abs(taxi_row - station1_row) / max(estimated_grid_size, 1),
        abs(taxi_col - station1_col) / max(estimated_grid_size, 1),
        abs(taxi_row - station2_row) / max(estimated_grid_size, 1),
        abs(taxi_col - station2_col) / max(estimated_grid_size, 1),
        abs(taxi_row - station3_row) / max(estimated_grid_size, 1),
        abs(taxi_col - station3_col) / max(estimated_grid_size, 1),
        
        # Directional indicators to stations
        # These help the agent learn directional movement
        np.sign(taxi_row - station0_row),
        np.sign(taxi_col - station0_col),
        np.sign(taxi_row - station1_row),
        np.sign(taxi_col - station1_col),
        np.sign(taxi_row - station2_row),
        np.sign(taxi_col - station2_col),
        np.sign(taxi_row - station3_row),
        np.sign(taxi_col - station3_col),
        
        # Obstacles and status indicators
        float(obstacle_north),
        float(obstacle_south), 
        float(obstacle_east),
        float(obstacle_west),
        float(passenger_look),
        float(destination_look)
    ]
    
    return torch.FloatTensor(features).to(DEVICE)

def get_action(obs):
    """
    Takes an observation as input and returns an action (0-5).
    Uses the trained Q-network to select the best action.
    """
    # Load model if it exists and hasn't been loaded yet
    if not hasattr(get_action, "model"):
        if os.path.exists(MODEL_FILE):
            get_action.model = DQN(24, 6).to(DEVICE)  # Updated input dimension
            get_action.model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
            get_action.model.eval()
        else:
            # If model doesn't exist, return random actions
            return random.choice([0, 1, 2, 3, 4, 5])
    
    # Epsilon-greedy policy for exploration during testing
    if random.random() < EPSILON:
        return random.choice([0, 1, 2, 3, 4, 5])
    
    # Preprocess state and get Q-values
    state_tensor = preprocess_state(obs)
    with torch.no_grad():
        q_values = get_action.model(state_tensor)
    
    # Return action with highest Q-value
    return torch.argmax(q_values).item()

def shape_reward(obs, next_obs, action, reward):
    """
    Apply reward shaping to guide the agent toward optimal behavior.
    Works with any grid size.
    """
    # Extract information from observations
    taxi_row, taxi_col, s0_row, s0_col, s1_row, s1_col, s2_row, s2_col, s3_row, s3_col, \
    obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs
    
    next_taxi_row, next_taxi_col, _, _, _, _, _, _, _, _, \
    _, _, _, _, next_passenger_look, next_destination_look = next_obs
    
    shaped_reward = reward
    stations = [(s0_row, s0_col), (s1_row, s1_col), (s2_row, s2_col), (s3_row, s3_col)]
    
    # Initialize state tracking if not already done
    if not hasattr(shape_reward, "previous_positions"):
        shape_reward.previous_positions = []
        shape_reward.last_action = None
        shape_reward.action_count = 0
        shape_reward.passenger_picked = False
        shape_reward.last_passenger_distance = float('inf')
        shape_reward.last_destination_distance = float('inf')
        shape_reward.visited_stations = set()
    
    # Track if passenger has been picked up
    if passenger_look == 1 and action == 4 and reward > 0:
        shape_reward.passenger_picked = True
    
    # Reward for exploring stations (helps with any grid size)
    current_position = (taxi_row, taxi_col)
    for station in stations:
        if abs(taxi_row - station[0]) + abs(taxi_col - station[1]) <= 1:
            if station not in shape_reward.visited_stations:
                shape_reward.visited_stations.add(station)
                shaped_reward += 2.0  # Reward for visiting a new station
    
    # PHASE 1: FINDING AND PICKING UP PASSENGER
    if not shape_reward.passenger_picked:
        # Calculate Manhattan distance to all stations
        min_distance = float('inf')
        for station in stations:
            distance = abs(taxi_row - station[0]) + abs(taxi_col - station[1])
            min_distance = min(min_distance, distance)
        
        # Reward for getting closer to any station (helps exploration)
        if min_distance < shape_reward.last_passenger_distance:
            shaped_reward += 0.5
        shape_reward.last_passenger_distance = min_distance
        
        # Reward for discovering passenger
        if passenger_look == 0 and next_passenger_look == 1:
            shaped_reward += 10.0
        
        # Reward for picking up passenger when visible
        if passenger_look == 1 and action == 4 and reward > 0:
            shaped_reward += 20.0
        
        # Penalty for attempting pickup when no passenger is visible
        if passenger_look == 0 and action == 4:
            shaped_reward -= 3.0
    
    # PHASE 2: DELIVERING PASSENGER
    else:
        # Reward for discovering destination
        if destination_look == 0 and next_destination_look == 1:
            shaped_reward += 10.0
        
        # Reward for dropping off at destination
        if destination_look == 1 and action == 5 and reward > 0:
            shaped_reward += 20.0
        
        # Penalty for attempting dropoff when not at destination
        if destination_look == 0 and action == 5:
            shaped_reward -= 3.0
        
        # Calculate Manhattan distance to all stations
        min_distance = float('inf')
        for station in stations:
            distance = abs(taxi_row - station[0]) + abs(taxi_col - station[1])
            min_distance = min(min_distance, distance)
        
        # Reward for getting closer to any station (helps find destination)
        if min_distance < shape_reward.last_destination_distance:
            shaped_reward += 0.5
        shape_reward.last_destination_distance = min_distance
    
    # GENERAL MOVEMENT REWARDS/PENALTIES
    
    # Penalty for hitting walls (no movement despite action)
    if action < 4 and taxi_row == next_taxi_row and taxi_col == next_taxi_col:
        shaped_reward -= 1.0
    
    # Penalty for revisiting the same position multiple times (anti-looping)
    shape_reward.previous_positions.append(current_position)
    if len(shape_reward.previous_positions) > 10:
        shape_reward.previous_positions.pop(0)
    
    # Count position occurrences in recent history
    position_count = shape_reward.previous_positions.count(current_position)
    if position_count > 2:
        shaped_reward -= 0.3 * position_count
    
    # Penalty for action oscillation
    if shape_reward.last_action is not None:
        if (action == 0 and shape_reward.last_action == 1) or \
           (action == 1 and shape_reward.last_action == 0) or \
           (action == 2 and shape_reward.last_action == 3) or \
           (action == 3 and shape_reward.last_action == 2):
            shape_reward.action_count += 1
            if shape_reward.action_count > 1:
                shaped_reward -= 0.5 * shape_reward.action_count
        else:
            shape_reward.action_count = 0
    
    # Update last action
    shape_reward.last_action = action
    
    return shaped_reward

def train_agent(num_episodes=10000, gamma=0.99, batch_size=64):
    """
    Train the agent using DQN with improved parameters and reward shaping.
    """
    from simple_custom_taxi_env import SimpleTaxiEnv
    
    # Initialize environment
    env = SimpleTaxiEnv()
    
    # Initialize Q-networks (policy and target)
    policy_net = DQN(24, 6).to(DEVICE)  # Updated input dimension
    target_net = DQN(24, 6).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    # Initialize optimizer with appropriate learning rate
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    criterion = nn.HuberLoss()
    
    # Initialize replay buffer
    replay_buffer = ReplayBuffer(capacity=100000)
    
    # Training parameters
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.9999
    target_update_frequency = 5
    
    # Training loop
    best_reward = -float('inf')
    best_success_rate = 0.0
    losses = []
    episode_rewards = []
    
    # For tracking progress
    success_rate = []
    window_size = 100
    
    for episode in tqdm(range(num_episodes)):
        obs, _ = env.reset()
        state_tensor = preprocess_state(obs)
        done = False
        total_reward = 0
        episode_losses = []
        steps = 0
        
        while not done and steps < 200:
            steps += 1
            
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.choice([0, 1, 2, 3, 4, 5])
            else:
                with torch.no_grad():
                    q_values = policy_net(state_tensor)
                action = torch.argmax(q_values).item()
            
            # Take action and observe next state
            next_obs, reward, done, _, _ = env.step(action)
            next_state_tensor = preprocess_state(next_obs)
            
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
                states, actions, batch_rewards, next_states, dones = replay_buffer.sample(batch_size)
                
                # Convert to tensors
                states = torch.FloatTensor(states).to(DEVICE)
                actions = torch.LongTensor(actions).to(DEVICE)
                batch_rewards = torch.FloatTensor(batch_rewards).to(DEVICE)
                next_states = torch.FloatTensor(next_states).to(DEVICE)
                dones = torch.FloatTensor(dones).to(DEVICE)
                
                # Compute current Q values
                current_q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                
                # Compute target Q values with target network (Double DQN approach)
                with torch.no_grad():
                    # Select actions using policy network
                    policy_actions = policy_net(next_states).max(1)[1].unsqueeze(1)
                    # Evaluate actions using target network
                    next_q_values = target_net(next_states).gather(1, policy_actions).squeeze(1)
                    target_q_values = batch_rewards + gamma * next_q_values * (1 - dones)
                
                # Compute loss and update
                loss = criterion(current_q_values, target_q_values)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
                optimizer.step()
                
                episode_losses.append(loss.item())
        
        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        # Update target network periodically
        if (episode + 1) % target_update_frequency == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # Track metrics
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        losses.append(avg_loss)
        episode_rewards.append(total_reward)
        
        # Track success rate (reward > 20 indicates successful delivery)
        success = 1 if total_reward > 20 else 0
        success_rate.append(success)
        if len(success_rate) > window_size:
            success_rate.pop(0)
        
        # Calculate current success rate
        current_success_rate = np.mean(success_rate)
        
        # Save model based on both reward and success rate
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(policy_net.state_dict(), MODEL_FILE)
        
        # Also save if we have a better success rate
        if current_success_rate > best_success_rate and len(success_rate) >= window_size/2:
            best_success_rate = current_success_rate
            torch.save(policy_net.state_dict(), "best_success_" + MODEL_FILE)
        
        # Print progress
        if (episode + 1) % 100 == 0:
            avg_success = current_success_rate * 100
            print(f"Episode {episode + 1}/{num_episodes}, Reward: {total_reward:.2f}, Best: {best_reward:.2f}, Loss: {avg_loss:.6f}, Epsilon: {epsilon:.4f}, Success Rate: {avg_success:.1f}%")
    
    # At the end of training, use the model with the best success rate
    if os.path.exists("best_success_" + MODEL_FILE):
        os.replace("best_success_" + MODEL_FILE, MODEL_FILE)
    
    print("Training completed and model saved.")
    return episode_rewards, losses

if __name__ == "__main__":
    # This will only run when you execute this file directly
    rewards_history, losses_history = train_agent(num_episodes=10000)