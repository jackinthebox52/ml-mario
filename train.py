import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os
import math
import argparse
from datetime import datetime
import os
from gym_super_mario_bros import SuperMarioBrosEnv
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym import Wrapper
import time
from typing import Tuple
from functools import reduce
from operator import mul


class GymAPIWrapper(Wrapper):
    """Bridge between old and new gym API"""
    def reset(self, **kwargs):
        obs = self.env.reset()
        return obs, {}
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, False, info

def create_mario_env():
    env = SuperMarioBrosEnv()
    env = GymAPIWrapper(env)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, 84)
    env = FrameStack(env, 4)
    return env

class DQN(nn.Module):
    def __init__(self, input_shape: Tuple[int, int, int], n_actions: int):
        """
        Deep Q-Network
        input_shape: (C, H, W) tuple of input dimensions
        n_actions: number of possible actions
        """
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        conv_out_size = self._get_conv_output(input_shape)
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    
    def _get_conv_output(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.shape))
    
    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

class ReplayBuffer:
    def __init__(self, capacity):
        # Pre-allocate memory for the entire buffer
        self.capacity = capacity
        self.states = np.zeros((capacity, 4, 84, 84), dtype=np.float32)
        self.next_states = np.zeros((capacity, 4, 84, 84), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0
    
    def push(self, state, action, reward, next_state, done):
        # Convert and squeeze the extra dimension
        state_arr = np.array(state, dtype=np.float32)
        next_state_arr = np.array(next_state, dtype=np.float32)
        
        if state_arr.shape[-1] == 1:
            state_arr = state_arr.squeeze(-1)
            next_state_arr = next_state_arr.squeeze(-1)
            
        # Normalize and store
        self.states[self.pos] = state_arr / 255.0
        self.next_states[self.pos] = next_state_arr / 255.0
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.states[indices]),
            torch.from_numpy(self.actions[indices]),
            torch.from_numpy(self.rewards[indices]),
            torch.from_numpy(self.next_states[indices]),
            torch.from_numpy(self.dones[indices])
        )
    
    def __len__(self):
        return self.size
    
class MarioAgent:
    def __init__(self, state_shape, n_actions, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.state_shape = state_shape
        self.n_actions = n_actions
        
        # Networks
        self.policy_net = DQN(state_shape, n_actions).to(device)
        self.target_net = DQN(state_shape, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Training parameters
        self.batch_size = 128
        self.gamma = 0.99
        self.epsilon_start = 1.0
        self.epsilon_final = 0.01
        self.epsilon_decay = 100000
        self.target_update = 1000
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.00025)
        
        # Pre-allocate tensors on device
        self.batch_indices = torch.arange(self.batch_size, device=device)
        
        # Optimized replay buffer
        self.memory = ReplayBuffer(100000)
        
        # Initialize steps counter
        self.steps = 0
    
    def _preprocess_observation(self, obs):
        """Convert LazyFrames or numpy array to proper tensor format"""
        if isinstance(obs, np.ndarray):
            array = obs
        else:
            # Handle LazyFrames from FrameStack wrapper
            array = np.array(obs)
        
        # Ensure correct shape and normalization
        if array.shape[-1] == 1:
            array = array.squeeze(-1)
        return array
    
    def select_action(self, state):
        if random.random() > self.get_epsilon():
            with torch.no_grad():
                # Process the state properly
                state_array = self._preprocess_observation(state)
                state_tensor = torch.from_numpy(state_array).unsqueeze(0).to(self.device, non_blocking=True) / 255.0
                return self.policy_net(state_tensor).max(1)[1].item()
        return random.randrange(self.n_actions)
    
    def preprocess_state(self, state):
        """Convert state to correct tensor format efficiently"""
        # Convert to numpy array first, then to tensor
        state = np.array(state, dtype=np.float32) / 255.0
        if state.shape[-1] == 1:
            state = state.squeeze(-1)
        return torch.from_numpy(state).to(self.device)
    
    def get_epsilon(self):
        return self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
            math.exp(-1. * self.steps / self.epsilon_decay)
    
    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample and transfer to GPU in one go
        states, actions, rewards, next_states, dones = [
            t.to(self.device, non_blocking=True) 
            for t in self.memory.sample(self.batch_size)
        ]
        
        # Compute current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute next Q values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss and optimize
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad(set_to_none=True)  # Slightly faster than zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.steps += 1
        return loss.item()
     
    def save(self, path):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps': self.steps
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps = checkpoint['steps']
        
class StuckDetector:
    def __init__(self, timeout=3, position_threshold=30):
        self.timeout = timeout
        self.position_threshold = position_threshold
        self.last_significant_move_time = time.time()
        self.last_x_pos = 0
        
    def check(self, info):
        current_time = time.time()
        current_x_pos = info.get('x_pos', 0)
        
        # Check if Mario has moved significantly
        if abs(current_x_pos - self.last_x_pos) > self.position_threshold:
            self.last_significant_move_time = current_time
            
        self.last_x_pos = current_x_pos
        
        # Check if we've been stuck
        time_since_move = current_time - self.last_significant_move_time
        is_stuck = time_since_move > self.timeout
        
        return is_stuck
        
    def reset(self):
        self.last_significant_move_time = time.time()
        self.last_x_pos = 0


class RewardManager:
    def __init__(self):
        self.status_multipliers = {
            0: 1.0,    # Small Mario
            1: 1.5,    # Super Mario
            2: 2.0     # Fire Mario
        }
        self.stomps = 0
        
    def get_reward(self, progress: float, mario_status: int) -> float:
        """
        Calculate reward based on:
        - Forward progress (scaled down to reasonable numbers)
        - Current power-up state
        - Number of successful stomps
        """
        # Base reward is scaled progress
        if progress > 0:
            base_reward = progress / 50.0  # Scale down distance rewards
            
            # Apply status multiplier
            status_mult = self.status_multipliers.get(mario_status, 1.0)
            
            # Small bonus for each stomp we've done (encourages keeping enemies dead)
            stomp_bonus = self.stomps * 0.1
            
            return base_reward * (status_mult + stomp_bonus)
        return 0
    
    def add_stomp(self):
        """Record a successful stomp"""
        self.stomps += 1
        return 5.0  # Immediate fixed bonus for stomping
    
    def get_death_penalty(self):
        """Fixed penalty for dying"""
        return -10.0
    
    def get_stuck_penalty(self):
        """Fixed penalty for getting stuck"""
        return -5.0
    
    def reset(self):
        """Reset stomp counter for new episode"""
        self.stomps = 0
    
    def get_info(self) -> dict:
        """Get current state for logging"""
        return {
            'stomps': self.stomps,
            'stomp_bonus': self.stomps * 0.1
        }

def train_mario(args):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_dir = os.path.join('checkpoints', timestamp)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    env = create_mario_env()
    state_shape = (4, 84, 84)
    n_actions = env.action_space.n
    agent = MarioAgent(state_shape, n_actions)
    stuck_detector = StuckDetector()
    reward_manager = RewardManager()
    
    episodes = 10000
    save_frequency = 10
    update_frequency = 8
    step_counter = 0
    status_dict = {'small': 0, 'big': 1, 'fire': 2}
    
    # Frame limiting for GUI mode
    frame_limit = 1/15
    last_render_time = time.time()
    
    # Save training metadata
    with open(os.path.join(checkpoint_dir, 'training_info.txt'), 'w') as f:
        f.write(f'Training started: {timestamp}\n')
        f.write(f'Update frequency: {update_frequency}\n')
        f.write(f'Save frequency: {save_frequency}\n')
        f.write(f'GUI enabled: {args.gui}\n')
        f.write(f'Verbose mode: {args.verbose}\n')
        f.write(f'Stuck detection: enabled (timeout=3s)\n')
    
    for episode in range(episodes):
        state, _ = env.reset()
        stuck_detector.reset()
        reward_manager.reset()
        total_reward = 0
        done = False
        episode_start_time = time.time()
        terminated_by_stuck = False
        died = False
        
        # Get initial position
        _, _, _, _, info = env.step(0)
        start_x_pos = info.get('x_pos', 0)
        state, _ = env.reset()
        
        # Track position
        last_x_pos = start_x_pos
        
        while not done:
            current_time = time.time()
            
            if args.gui and (current_time - last_render_time) >= frame_limit:
                env.render()
                last_render_time = current_time
            
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Get Mario's status and position
            mario_status = status_dict.get(info.get('status', 'small'), 0)
            current_x_pos = info.get('x_pos', 0)
            
            # Check if Mario died
            if info.get('life', 2) < 2:
                died = True
                reward = reward_manager.get_death_penalty()
                terminated = True
            else:
                # Calculate progress
                progress = current_x_pos - last_x_pos
                
                # Check for stomps
                if info.get('flag_get', False):  # Mario stomped an enemy
                    reward = reward_manager.add_stomp()
                else:
                    # Normal progress reward
                    reward = reward_manager.get_reward(progress, mario_status)
                
                # Check if Mario is stuck
                if stuck_detector.check(info):
                    terminated = True
                    terminated_by_stuck = True
                    reward = reward_manager.get_stuck_penalty()
            
            done = terminated or truncated
            
            agent.memory.push(state, action, reward, next_state, done)
            
            if step_counter % update_frequency == 0:
                loss = agent.train_step()
            
            state = next_state
            total_reward += reward
            step_counter += 1
            last_x_pos = current_x_pos
        
        episode_duration = time.time() - episode_start_time
        distance_traveled = current_x_pos - start_x_pos
        reward_info = reward_manager.get_info()
        
        # Save checkpoint every save_frequency episodes
        if episode % save_frequency == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'episode_{episode:05d}.pth')
            agent.save(checkpoint_path)
            
            meta_path = os.path.join(checkpoint_dir, f'episode_{episode:05d}_meta.txt')
            with open(meta_path, 'w') as f:
                f.write(f'Episode: {episode}\n')
                f.write(f'Total Reward: {total_reward}\n')
                f.write(f'Duration: {episode_duration:.2f}s\n')
                f.write(f'Total Steps: {step_counter}\n')
                f.write(f'Current Epsilon: {agent.get_epsilon():.4f}\n')
                f.write(f'Died: {died}\n')
                f.write(f'Terminated by stuck: {terminated_by_stuck}\n')
                f.write(f'Distance traveled: {distance_traveled}\n')
                f.write(f'Stomps: {reward_info["stomps"]}\n')
        
        outcome = "DIED" if died else "STUCK" if terminated_by_stuck else "DONE"
        mario_state = "SMALL" if mario_status == 0 else "SUPER" if mario_status == 1 else "FIRE"
        if args.verbose:
            print(f'Episode {episode} [{outcome}] [{mario_state}]: Reward = {total_reward:.1f}, '\
                  f'Duration = {episode_duration:.2f}s, Distance = {distance_traveled}, '\
                  f'Stomps = {reward_info["stomps"]}')
        elif episode % 2 == 0:
            print(f'Episode {episode} [{outcome}] [{mario_state}]: Reward = {total_reward:.1f}, '\
                  f'Duration = {episode_duration:.2f}s, Distance = {distance_traveled}, '\
                  f'Stomps = {reward_info["stomps"]}')
        
def parse_args():
    parser = argparse.ArgumentParser(description='Train Mario RL agent')
    parser.add_argument('--gui', action='store_true', default=False,
                      help='Enable GUI display (default: disabled)')
    parser.add_argument('--verbose', action='store_true', default=False,
                      help='Enable verbose output (default: disabled)')
    parser.add_argument('--fps', type=int, default=15,
                      help='Target FPS when GUI is enabled (default: 15)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_mario(args)
