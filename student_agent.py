import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from tqdm import tqdm

# Global variables
MODEL_PATH = "q_network.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
EPSILON = 0.08  # Slightly reduced from 0.1 for testing to be more greedy

class DQN(nn.Module):
    def __init__(self, state_size, action_count):
        super(DQN, self).__init__()
        # Enhanced network architecture with better layer sizes
        self.layers = nn.Sequential(
            nn.Linear(state_size, 160),  # Increased from 128
            nn.ReLU(),
            nn.Linear(160, 80),  # Increased from 64
            nn.ReLU(),
            nn.Linear(80, 40),  # Increased from 32
            nn.ReLU(),
            nn.Linear(40, action_count)
        )
    
    def forward(self, x):
        return self.layers(x)

# Memory buffer for experience replay
class ExperienceMemory:
    def __init__(self, max_size=12000):  # Increased from 10000
        self.max_size = max_size
        self.experiences = []
        self.insert_index = 0
        
    def store(self, state, action, reward, next_state, terminal):
        if len(self.experiences) < self.max_size:
            self.experiences.append(None)
        self.experiences[self.insert_index] = (state, action, reward, next_state, terminal)
        self.insert_index = (self.insert_index + 1) % self.max_size
        
    def batch_retrieve(self, batch_size):
        batch = random.sample(self.experiences, batch_size)
        state, action, reward, next_state, terminal = map(np.stack, zip(*batch))
        return state, action, reward, next_state, terminal
    
    def __len__(self):
        return len(self.experiences)

def process_observation(observation):
    """
    Transform raw observation into feature vector for neural network.
    Uses normalized and relative positions for grid-size independence.
    """
    # Extract observation components
    taxi_r, taxi_c, loc0_r, loc0_c, loc1_r, loc1_c, \
    loc2_r, loc2_c, loc3_r, loc3_c, \
    wall_up, wall_down, wall_right, wall_left, \
    is_passenger_visible, is_destination_visible = observation
    
    # Calculate max coordinate to estimate grid dimensions
    max_pos = max(taxi_r, taxi_c, loc0_r, loc0_c, 
                loc1_r, loc1_c, loc2_r, loc2_c,
                loc3_r, loc3_c)
    grid_size = max_pos + 1  # Adding 1 because coordinates start at 0
    
    # Create normalized feature vector
    features = [
        # Normalized taxi coordinates
        taxi_r / max(grid_size, 1),
        taxi_c / max(grid_size, 1),
        
        # Normalized distances to all locations
        abs(taxi_r - loc0_r) / max(grid_size, 1),
        abs(taxi_c - loc0_c) / max(grid_size, 1),
        abs(taxi_r - loc1_r) / max(grid_size, 1),
        abs(taxi_c - loc1_c) / max(grid_size, 1),
        abs(taxi_r - loc2_r) / max(grid_size, 1),
        abs(taxi_c - loc2_c) / max(grid_size, 1),
        abs(taxi_r - loc3_r) / max(grid_size, 1),
        abs(taxi_c - loc3_c) / max(grid_size, 1),
        
        # Direction indicators to stations (relative positioning)
        np.sign(taxi_r - loc0_r),
        np.sign(taxi_c - loc0_c),
        np.sign(taxi_r - loc1_r),
        np.sign(taxi_c - loc1_c),
        np.sign(taxi_r - loc2_r),
        np.sign(taxi_c - loc2_c),
        np.sign(taxi_r - loc3_r),
        np.sign(taxi_c - loc3_c),
        
        # Wall/obstacle indicators and status flags
        float(wall_up),
        float(wall_down), 
        float(wall_right),
        float(wall_left),
        float(is_passenger_visible),
        float(is_destination_visible)
    ]
    
    return torch.FloatTensor(features).to(DEVICE)

def get_action(obs):
    """
    Determines the best action for a given observation.
    Uses the trained neural network with epsilon-greedy exploration.
    """
    # Lazily load model on first call
    if not hasattr(get_action, "model"):
        if os.path.exists(MODEL_PATH):
            get_action.model = DQN(24, 6).to(DEVICE)
            get_action.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            get_action.model.eval()
        else:
            # Return random action if model doesn't exist
            return random.choice([0, 1, 2, 3, 4, 5])
    
    # Epsilon-greedy policy for exploration during testing
    if random.random() < EXPLORATION_RATE:
        return random.choice([0, 1, 2, 3, 4, 5])
    
    # Get network prediction for best action
    state_tensor = process_observation(obs)
    with torch.no_grad():
        action_values = get_action.model(state_tensor)
    
    # Return action with highest predicted value
    return torch.argmax(action_values).item()

def enhance_reward(obs, next_obs, action, reward):
    """
    Implements reward shaping to improve learning efficiency.
    Works with variable grid sizes by using relative metrics.
    """
    # Extract observation details
    taxi_r, taxi_c, s0_r, s0_c, s1_r, s1_c, s2_r, s2_c, s3_r, s3_c, \
    wall_up, wall_down, wall_right, wall_left, has_passenger, has_destination = obs
    
    next_taxi_r, next_taxi_c, _, _, _, _, _, _, _, _, \
    _, _, _, _, next_has_passenger, next_has_destination = next_obs
    
    modified_reward = reward
    stations = [(s0_r, s0_c), (s1_r, s1_c), (s2_r, s2_c), (s3_r, s3_c)]
    
    # Initialize state tracking variables
    if not hasattr(enhance_reward, "position_history"):
        enhance_reward.position_history = []
        enhance_reward.prev_action = None
        enhance_reward.repeat_action_count = 0
        enhance_reward.passenger_onboard = False
        enhance_reward.prev_passenger_dist = float('inf')
        enhance_reward.prev_destination_dist = float('inf')
        enhance_reward.discovered_stations = set()
    
    # Update passenger status when picked up
    if has_passenger == 1 and action == 4 and reward > 0:
        enhance_reward.passenger_onboard = True
    
    # Add exploration bonus for discovering new stations
    current_pos = (taxi_r, taxi_c)
    for station in stations:
        if abs(taxi_r - station[0]) + abs(taxi_c - station[1]) <= 1:
            if station not in enhance_reward.discovered_stations:
                enhance_reward.discovered_stations.add(station)
                modified_reward += 2.5  # Increased from 2.0 to encourage exploration
    
    # PHASE 1: SEEKING AND PICKING UP PASSENGER
    if not enhance_reward.passenger_onboard:
        # Find closest station
        min_dist = float('inf')
        for station in stations:
            dist = abs(taxi_r - station[0]) + abs(taxi_c - station[1])
            min_dist = min(min_dist, dist)
        
        # Progress reward for approaching stations
        if min_dist < enhance_reward.prev_passenger_dist:
            modified_reward += 0.7  # Increased from 0.5
        enhance_reward.prev_passenger_dist = min_dist
        
        # Reward for discovering passenger
        if has_passenger == 0 and next_has_passenger == 1:
            modified_reward += 12.0  # Increased from 10.0
        
        # Reward for successful pickup
        if has_passenger == 1 and action == 4 and reward > 0:
            modified_reward += 25.0  # Increased from 20.0
        
        # Penalty for invalid pickup attempts
        if has_passenger == 0 and action == 4:
            modified_reward -= 3.5  # Increased from 3.0
    
    # PHASE 2: DELIVERING PASSENGER
    else:
        # Reward for discovering destination
        if has_destination == 0 and next_has_destination == 1:
            modified_reward += 12.0  # Increased from 10.0
        
        # Reward for successful dropoff
        if has_destination == 1 and action == 5 and reward > 0:
            modified_reward += 25.0  # Increased from 20.0
        
        # Penalty for invalid dropoff attempts
        if has_destination == 0 and action == 5:
            modified_reward -= 3.5  # Increased from 3.0
        
        # Find closest station
        min_dist = float('inf')
        for station in stations:
            dist = abs(taxi_r - station[0]) + abs(taxi_c - station[1])
            min_dist = min(min_dist, dist)
        
        # Progress reward for approaching stations
        if min_dist < enhance_reward.prev_destination_dist:
            modified_reward += 0.7  # Increased from 0.5
        enhance_reward.prev_destination_dist = min_dist
    
    # MOVEMENT EFFICIENCY REWARDS/PENALTIES
    
    # Penalty for hitting walls
    if action < 4 and taxi_r == next_taxi_r and taxi_c == next_taxi_c:
        modified_reward -= 1.2  # Increased from 1.0
    
    # Anti-looping penalty based on position history
    enhance_reward.position_history.append(current_pos)
    if len(enhance_reward.position_history) > 12:  # Increased from 10
        enhance_reward.position_history.pop(0)
    
    # Count position repetitions
    pos_repetitions = enhance_reward.position_history.count(current_pos)
    if pos_repetitions > 2:
        modified_reward -= 0.4 * pos_repetitions  # Increased from 0.3
    
    # Penalize oscillating behaviors
    if enhance_reward.prev_action is not None:
        if (action == 0 and enhance_reward.prev_action == 1) or \
           (action == 1 and enhance_reward.prev_action == 0) or \
           (action == 2 and enhance_reward.prev_action == 3) or \
           (action == 3 and enhance_reward.prev_action == 2):
            enhance_reward.repeat_action_count += 1
            if enhance_reward.repeat_action_count > 1:
                modified_reward -= 0.7 * enhance_reward.repeat_action_count  # Increased from 0.5
        else:
            enhance_reward.repeat_action_count = 0
    
    # Store current action for next comparison
    enhance_reward.prev_action = action
    
    return modified_reward

def train_taxi_agent(num_episodes=12000, discount_factor=0.99, batch_size=96):
    """
    Train the agent using enhanced DQN approach with optimized parameters.
    """
    from simple_custom_taxi_env import SimpleTaxiEnv
    
    # Create environment
    env = SimpleTaxiEnv()
    
    # Initialize networks
    policy_network = DQN(24, 6).to(DEVICE)
    target_network = DQN(24, 6).to(DEVICE)
    target_network.load_state_dict(policy_network.state_dict())
    target_network.eval()
    
    # Enhanced optimizer with tuned learning rate
    optimizer = optim.Adam(policy_network.parameters(), lr=0.0008)  # Reduced from 0.001
    loss_function = nn.SmoothL1Loss()  # Same as HuberLoss but explicitly named
    
    # Initialize experience buffer
    memory = ExperienceMemory(capacity=120000)  # Increased from 100000
    
    # Training hyperparameters
    start_epsilon = 1.0
    min_epsilon = 0.008  # Reduced from 0.01
    epsilon_decay = 0.9998  # Slower decay from 0.9999
    target_update_freq = 4  # More frequent updates from 5
    
    # Tracking variables
    highest_reward = -float('inf')
    best_success_percentage = 0.0
    loss_history = []
    reward_history = []
    
    # Success tracking
    success_window = []
    window_size = 100
    
    for episode in tqdm(range(num_episodes)):
        observation, _ = env.reset()
        state = process_observation(observation)
        done = False
        episode_reward = 0
        episode_loss_values = []
        step_count = 0
        
        while not done and step_count < 200:
            step_count += 1
            
            # Select action with epsilon-greedy policy
            epsilon = max(min_epsilon, start_epsilon * (epsilon_decay ** episode))
            if random.random() < epsilon:
                action = random.choice([0, 1, 2, 3, 4, 5])
            else:
                with torch.no_grad():
                    q_values = policy_network(state)
                action = torch.argmax(q_values).item()
            
            # Execute action
            next_observation, reward, done, _, _ = env.step(action)
            next_state = process_observation(next_observation)
            
            # Apply reward shaping
            shaped_reward = enhance_reward(observation, next_observation, action, reward)
            
            # Store experience
            memory.store(
                state.cpu().numpy(),
                action,
                shaped_reward,
                next_state.cpu().numpy(),
                done
            )
            
            # Track original reward for evaluation
            episode_reward += reward
            
            # Update state
            observation = next_observation
            state = next_state
            
            # Learn from experiences if enough samples available
            if len(memory) >= batch_size:
                # Sample batch
                states, actions, rewards, next_states, dones = memory.batch_retrieve(batch_size)
                
                # Convert to tensors
                states = torch.FloatTensor(states).to(DEVICE)
                actions = torch.LongTensor(actions).to(DEVICE)
                rewards = torch.FloatTensor(rewards).to(DEVICE)
                next_states = torch.FloatTensor(next_states).to(DEVICE)
                dones = torch.FloatTensor(dones).to(DEVICE)
                
                # Get current Q values
                current_q = policy_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                
                # Compute target Q values using Double DQN approach
                with torch.no_grad():
                    # Select best actions according to policy network
                    best_actions = policy_network(next_states).max(1)[1].unsqueeze(1)
                    # Evaluate those actions using target network
                    max_next_q = target_network(next_states).gather(1, best_actions).squeeze(1)
                    target_q = rewards + discount_factor * max_next_q * (1 - dones)
                
                # Compute loss and update weights
                loss = loss_function(current_q, target_q)
                
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(policy_network.parameters(), max_norm=1.0)
                optimizer.step()
                
                episode_loss_values.append(loss.item())
        
        # Update target network periodically
        if (episode + 1) % target_update_freq == 0:
            target_network.load_state_dict(policy_network.state_dict())
        
        # Track metrics
        avg_loss = np.mean(episode_loss_values) if episode_loss_values else 0
        loss_history.append(avg_loss)
        reward_history.append(episode_reward)
        
        # Track success rate (reward > 20 indicates successful delivery)
        is_success = 1 if episode_reward > 20 else 0
        success_window.append(is_success)
        if len(success_window) > window_size:
            success_window.pop(0)
        
        # Calculate current success rate
        current_success_rate = np.mean(success_window)
        
        # Save model based on reward improvement
        if episode_reward > highest_reward:
            highest_reward = episode_reward
            torch.save(policy_network.state_dict(), MODEL_PATH)
        
        # Also save on success rate improvement
        if current_success_rate > best_success_percentage and len(success_window) >= window_size/2:
            best_success_percentage = current_success_rate
            torch.save(policy_network.state_dict(), "best_success_" + MODEL_PATH)
        
        # Print training progress
        if (episode + 1) % 100 == 0:
            success_percent = current_success_rate * 100
            print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}, Best: {highest_reward:.2f}, Loss: {avg_loss:.6f}, Epsilon: {epsilon:.4f}, Success Rate: {success_percent:.1f}%")
    
    # Use the model with best success rate
    if os.path.exists("best_success_" + MODEL_PATH):
        os.replace("best_success_" + MODEL_PATH, MODEL_PATH)
    
    print("Training completed and model saved successfully.")
    return reward_history, loss_history

if __name__ == "__main__":
    # Run training when script is executed directly
    reward_history, loss_history = train_taxi_agent(num_episodes=12000)