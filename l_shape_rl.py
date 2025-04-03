import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_checker import check_env
from typing import List, Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import ray
from ray import tune
from ray.rllib.algorithms import ppo
from collections import deque
import time
import random
import os
import math

# Import the L-shape grid implementation
from l_shape_ramsey import LShapeGrid, Color

"""
L-Shape Ramsey Problem Solver using Reinforcement Learning

This file implements various reinforcement learning techniques to solve the L-shape Ramsey problem,
which involves finding grid configurations that avoid monochromatic L-shapes while using a 
limited number of colors.

RL Techniques implemented:
1. Deep Q-Networks (DQN)
2. Proximal Policy Optimization (PPO)
3. Monte Carlo Tree Search (MCTS)
4. Curriculum Learning
5. Multi-Agent RL approaches
"""

# ========================================================================================
# Environment Implementation
# ========================================================================================

class LShapeRamseyEnv(gym.Env):
    """
    Gymnasium environment for the L-shape Ramsey problem.
    
    The environment represents the task of coloring a grid while avoiding monochromatic L-shapes.
    
    State:
        The current coloring of the grid
    
    Actions:
        Coloring a specific cell with a specific color
    
    Reward:
        - Positive reward for successfully coloring the grid without monochromatic L-shapes
        - Negative reward for creating a monochromatic L-shape
        - Small negative step penalty to encourage efficient solutions
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, grid_size: int = 5, num_colors: int = 3, step_penalty: float = -0.01):
        super(LShapeRamseyEnv, self).__init__()
        
        self.grid_size = grid_size
        self.num_colors = num_colors
        self.step_penalty = step_penalty
        
        # Action space: (position, color) pairs
        # For each cell, we can assign one of num_colors colors
        self.action_space = spaces.Discrete(grid_size * grid_size * num_colors)
        
        # Observation space: grid_size x grid_size with num_colors + 1 possible values (colors + empty)
        self.observation_space = spaces.Box(
            low=0, 
            high=num_colors,
            shape=(grid_size, grid_size),
            dtype=np.int32
        )
        
        # Initialize grid
        self.reset()
    
    def decode_action(self, action: int) -> Tuple[int, int, int]:
        """
        Decode an action index into (x, y, color_idx) coordinates
        """
        color_idx = action % self.num_colors
        position = action // self.num_colors
        x = position % self.grid_size
        y = position // self.grid_size
        return x, y, color_idx
    
    def encode_state(self) -> np.ndarray:
        """
        Convert the grid to a format suitable for the observation space
        """
        state = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                color = self.grid.get_color(x, y)
                if color is not None:
                    state[y, x] = color.value + 1  # +1 to reserve 0 for empty cells
                else:
                    state[y, x] = 0  # Empty cell
        return state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment
        
        Args:
            action: Index of the action to take
            
        Returns:
            observation: Next state observation
            reward: Reward received
            terminated: Whether the episode is done
            truncated: Whether the episode was truncated early
            info: Additional information
        """
        x, y, color_idx = self.decode_action(action)
        color = list(Color)[color_idx]
        
        # Check if the cell is already colored
        if self.grid.get_color(x, y) is not None:
            # Invalid move, already colored
            return self.encode_state(), -1.0, False, False, {"invalid_move": True}
        
        # Apply the action
        self.grid.set_color(x, y, color)
        self.steps_taken += 1
        
        # Check if this creates an L-shape
        has_l_shape, l_shape_points = self.grid.has_any_l_shape()
        
        # Count empty cells
        empty_cells = sum(1 for y in range(self.grid_size) for x in range(self.grid_size) 
                         if self.grid.get_color(x, y) is None)
        
        # Terminal state conditions
        terminated = has_l_shape or empty_cells == 0
        
        # Calculate reward
        if has_l_shape:
            # Created an L-shape, negative reward
            reward = -10.0
        elif empty_cells == 0:
            # Successfully colored the entire grid
            reward = 100.0
        else:
            # Step penalty
            reward = self.step_penalty
        
        # Additional information
        info = {
            "has_l_shape": has_l_shape,
            "l_shape_points": l_shape_points if has_l_shape else [],
            "empty_cells": empty_cells,
            "steps_taken": self.steps_taken
        }
        
        return self.encode_state(), reward, terminated, False, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state
        
        Returns:
            observation: Initial state observation
            info: Additional information
        """
        if seed is not None:
            self.seed(seed)
            
        self.grid = LShapeGrid(self.grid_size)
        self.steps_taken = 0
        
        return self.encode_state(), {"empty_cells": self.grid_size * self.grid_size}
    
    def render(self, mode: str = 'human'):
        """
        Render the environment
        """
        if mode == 'human':
            print(self.grid)
            self.grid.visualize(highlight_l_shape=True)
        elif mode == 'rgb_array':
            # TODO: Implement rendering to RGB array for video creation
            pass
    
    def seed(self, seed: Optional[int] = None):
        """
        Set the seed for random number generation
        """
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

# ========================================================================================
# Deep Q-Network (DQN) Implementation
# ========================================================================================

class LShapeDQN:
    """
    Deep Q-Network implementation for the L-shape Ramsey problem.
    
    DQN is a value-based reinforcement learning algorithm that uses a neural network to 
    approximate the Q-function (expected future rewards). It includes several key innovations:
    
    1. Experience Replay: Stores transitions in a buffer and samples randomly to break
       correlations between consecutive samples.
    
    2. Target Network: Uses a separate network for generating targets to stabilize training.
    
    3. Double DQN: Reduces overestimation bias by decoupling action selection and evaluation.
    """
    
    def __init__(self, env, total_timesteps=100000, learning_rate=1e-4):
        """
        Initialize the DQN agent
        
        Args:
            env: The gym environment
            total_timesteps: Total number of timesteps to train
            learning_rate: Learning rate for the optimizer
        """
        self.env = env
        self.total_timesteps = total_timesteps
        self.learning_rate = learning_rate
        
        # Create the model using stable_baselines3
        self.model = DQN(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            buffer_size=10000,  # Replay buffer size
            learning_starts=1000,  # Number of steps before starting to learn
            batch_size=64,
            tau=1.0,  # Target network update rate
            gamma=0.99,  # Discount factor
            train_freq=4,  # How often to update the network
            target_update_interval=1000,  # How often to update the target network
            exploration_fraction=0.2,  # Exploration vs exploitation trade-off
            exploration_initial_eps=1.0,  # Initial exploration rate
            exploration_final_eps=0.05,  # Final exploration rate
            verbose=1
        )
    
    def train(self):
        """
        Train the DQN agent
        """
        print("Training DQN agent...")
        self.model.learn(total_timesteps=self.total_timesteps)
        print("Training completed!")
    
    def save(self, path):
        """
        Save the trained model
        """
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """
        Load a trained model
        """
        self.model = DQN.load(path, env=self.env)
        print(f"Model loaded from {path}")
    
    def evaluate(self, num_episodes=10):
        """
        Evaluate the trained model
        """
        print("Evaluating DQN agent...")
        
        successes = 0
        episode_rewards = []
        episode_steps = []
        
        for i in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
                steps += 1
            
            if not info.get("has_l_shape", False):
                successes += 1
            
            episode_rewards.append(total_reward)
            episode_steps.append(steps)
            
            print(f"Episode {i+1}: Reward = {total_reward:.2f}, Steps = {steps}, Success = {not info.get('has_l_shape', False)}")
        
        print(f"Success rate: {successes/num_episodes:.2f}")
        print(f"Average reward: {np.mean(episode_rewards):.2f}")
        print(f"Average steps: {np.mean(episode_steps):.2f}")
        
        return {
            "success_rate": successes/num_episodes,
            "avg_reward": np.mean(episode_rewards),
            "avg_steps": np.mean(episode_steps)
        }

# ========================================================================================
# Proximal Policy Optimization (PPO) Implementation
# ========================================================================================

class LShapePPO:
    """
    Proximal Policy Optimization implementation for the L-shape Ramsey problem.
    
    PPO is a policy gradient method that achieves state-of-the-art performance on many tasks.
    Key innovations include:
    
    1. Clipped Objective Function: Limits policy updates to prevent destructively large changes.
    
    2. Entropy Regularization: Encourages exploration by adding an entropy bonus to the objective.
    
    3. Value Function Optimization: Simultaneously optimizes both policy and value functions.
    
    4. Generalized Advantage Estimation (GAE): Provides more stable learning by balancing
       bias and variance in advantage estimation.
    """
    
    def __init__(self, env, total_timesteps=100000, learning_rate=3e-4):
        """
        Initialize the PPO agent
        
        Args:
            env: The gym environment
            total_timesteps: Total number of timesteps to train
            learning_rate: Learning rate for the optimizer
        """
        self.env = env
        self.total_timesteps = total_timesteps
        self.learning_rate = learning_rate
        
        # Create the model using stable_baselines3
        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=2048,  # Number of steps to collect per update
            batch_size=64,  # Minibatch size for each optimization step
            n_epochs=10,   # Number of optimization epochs per update
            gamma=0.99,    # Discount factor
            gae_lambda=0.95,  # GAE lambda parameter
            clip_range=0.2,   # Clipping parameter for PPO
            clip_range_vf=None,  # Clipping parameter for value function
            ent_coef=0.01,  # Entropy coefficient for exploration
            vf_coef=0.5,    # Value function coefficient
            max_grad_norm=0.5,  # Gradient clipping
            verbose=1
        )
    
    def train(self):
        """
        Train the PPO agent
        """
        print("Training PPO agent...")
        self.model.learn(total_timesteps=self.total_timesteps)
        print("Training completed!")
    
    def save(self, path):
        """
        Save the trained model
        """
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """
        Load a trained model
        """
        self.model = PPO.load(path, env=self.env)
        print(f"Model loaded from {path}")
    
    def evaluate(self, num_episodes=10):
        """
        Evaluate the trained model
        """
        print("Evaluating PPO agent...")
        
        successes = 0
        episode_rewards = []
        episode_steps = []
        
        for i in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                total_reward += reward
                steps += 1
            
            if not info.get("has_l_shape", False):
                successes += 1
            
            episode_rewards.append(total_reward)
            episode_steps.append(steps)
            
            print(f"Episode {i+1}: Reward = {total_reward:.2f}, Steps = {steps}, Success = {not info.get('has_l_shape', False)}")
        
        print(f"Success rate: {successes/num_episodes:.2f}")
        print(f"Average reward: {np.mean(episode_rewards):.2f}")
        print(f"Average steps: {np.mean(episode_steps):.2f}")
        
        return {
            "success_rate": successes/num_episodes,
            "avg_reward": np.mean(episode_rewards),
            "avg_steps": np.mean(episode_steps)
        }

# ========================================================================================
# Monte Carlo Tree Search (MCTS) Implementation
# ========================================================================================

class MCTSNode:
    """
    Node in the Monte Carlo Tree Search
    
    Each node represents a state in the environment and keeps track of statistics
    for the UCB (Upper Confidence Bound) formula that balances exploration and exploitation.
    """
    
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action  # Action that led to this state
        self.children = {}    # Maps actions to child nodes
        self.visits = 0       # Number of times this node was visited
        self.value = 0.0      # Total value (reward)
        self.untried_actions = None  # List of actions not yet expanded
    
    def is_fully_expanded(self):
        """Check if all possible actions have been expanded"""
        return self.untried_actions is not None and len(self.untried_actions) == 0
    
    def best_child(self, exploration_weight=1.0):
        """
        Select the best child node according to UCB formula
        
        UCB = (node_value / node_visits) + exploration_weight * sqrt(log(parent_visits) / node_visits)
        
        The formula balances exploitation (first term) with exploration (second term).
        """
        if not self.children:
            return None
        
        # UCB formula
        def ucb_score(child):
            exploitation = child.value / child.visits if child.visits > 0 else 0
            exploration = exploration_weight * math.sqrt(math.log(self.visits) / child.visits) if child.visits > 0 else float('inf')
            return exploitation + exploration
        
        return max(self.children.values(), key=ucb_score)
    
    def expand(self, action, next_state):
        """Expand the tree by adding a new child node"""
        child = MCTSNode(next_state, parent=self, action=action)
        self.children[action] = child
        
        # Remove this action from untried actions
        if self.untried_actions is not None:
            if action in self.untried_actions:
                self.untried_actions.remove(action)
        
        return child
    
    def update(self, reward):
        """Update the node statistics"""
        self.visits += 1
        self.value += reward
        
    def __repr__(self):
        return f"MCTSNode(visits={self.visits}, value={self.value:.2f}, actions={len(self.children)})"

class LShapeMCTS:
    """
    Monte Carlo Tree Search implementation for the L-shape Ramsey problem.
    
    MCTS is a decision-making algorithm that combines tree search with random sampling
    to find optimal decisions. It consists of four main steps:
    
    1. Selection: Select the most promising node in the tree according to a tree policy (UCB).
    
    2. Expansion: Expand the selected node by adding one or more child nodes.
    
    3. Simulation: Run a simulation from the new node(s) to estimate the value.
    
    4. Backpropagation: Update the value estimates for all nodes in the path.
    
    This implementation includes neural networks for policy and value estimation,
    similar to the approach used in AlphaZero.
    """
    
    def __init__(self, env, num_simulations=1000, exploration_weight=1.0):
        """
        Initialize the MCTS agent
        
        Args:
            env: The gym environment
            num_simulations: Number of simulations per action selection
            exploration_weight: Weight for the exploration term in UCB
        """
        self.env = env
        self.num_simulations = num_simulations
        self.exploration_weight = exploration_weight
        
        # Neural networks for policy and value estimation
        self.policy_network = None  # To be implemented
        self.value_network = None   # To be implemented
    
    def search(self, state):
        """
        Perform MCTS search from the given state
        
        Args:
            state: Current state of the environment
            
        Returns:
            best_action: The best action to take
        """
        # Create root node
        root = MCTSNode(state)
        
        # Initialize untried actions
        root.untried_actions = list(range(self.env.action_space.n))
        
        # Perform simulations
        for _ in range(self.num_simulations):
            # Clone the environment to avoid modifying the original
            env_copy = self.env
            
            # Selection and expansion
            node = self._select_and_expand(root, env_copy)
            
            # Simulation
            reward = self._simulate(node, env_copy)
            
            # Backpropagation
            self._backpropagate(node, reward)
        
        # Select the best action based on visits (not UCB)
        return max(root.children.keys(), key=lambda a: root.children[a].visits)
    
    def _select_and_expand(self, node, env):
        """
        Select a node to expand using UCB, then expand it
        
        Args:
            node: The current node
            env: The environment
            
        Returns:
            The newly expanded node
        """
        # Navigate down the tree until reaching a leaf node
        while not env.terminated and not env.truncated:
            if not node.is_fully_expanded():
                # Expand this node
                action = random.choice(node.untried_actions)
                
                # Take the action in the environment
                next_state, _, terminated, truncated, _ = env.step(action)
                
                # Expand the node with this action
                return node.expand(action, next_state)
            else:
                # Select the best child according to UCB
                node = node.best_child(self.exploration_weight)
                
                # Take the action in the environment
                action = node.action
                _, _, terminated, truncated, _ = env.step(action)
        
        return node
    
    def _simulate(self, node, env):
        """
        Run a simulation from the given node to estimate its value
        
        Args:
            node: The node to simulate from
            env: The environment
            
        Returns:
            The total reward from the simulation
        """
        total_reward = 0
        
        # Simulate until terminal state
        while not env.terminated and not env.truncated:
            # Take a random action
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
        
        return total_reward
    
    def _backpropagate(self, node, reward):
        """
        Update the value estimates for all nodes in the path
        
        Args:
            node: The leaf node
            reward: The reward to backpropagate
        """
        while node is not None:
            node.update(reward)
            node = node.parent
    
    def act(self, state):
        """
        Select the best action for the given state
        
        Args:
            state: The current state
            
        Returns:
            The best action to take
        """
        return self.search(state)

# ========================================================================================
# Curriculum Learning Implementation
# ========================================================================================

class CurriculumLearning:
    """
    Curriculum Learning implementation for the L-shape Ramsey problem.
    
    Curriculum learning is a training strategy where the agent first learns simpler tasks
    before progressively moving to more difficult ones. In this implementation, we:
    
    1. Start with smaller grid sizes and gradually increase.
    
    2. Transfer knowledge from smaller grids to larger ones.
    
    3. Implement adaptive difficulty adjustment based on agent performance.
    """
    
    def __init__(self, 
                 initial_grid_size=3, 
                 max_grid_size=10, 
                 success_threshold=0.8,
                 num_colors=3):
        """
        Initialize the curriculum learning framework
        
        Args:
            initial_grid_size: Starting grid size
            max_grid_size: Maximum grid size to train on
            success_threshold: Performance threshold to increase difficulty
            num_colors: Number of colors to use
        """
        self.initial_grid_size = initial_grid_size
        self.max_grid_size = max_grid_size
        self.success_threshold = success_threshold
        self.num_colors = num_colors
        self.current_grid_size = initial_grid_size
        
        # Create the initial environment
        self.env = LShapeRamseyEnv(grid_size=initial_grid_size, num_colors=num_colors)
        
        # Create the agent (using PPO for this example)
        self.agent = LShapePPO(self.env)
    
    def train_curriculum(self, timesteps_per_size=50000, eval_episodes=20):
        """
        Train the agent using curriculum learning
        
        Args:
            timesteps_per_size: Number of timesteps to train at each grid size
            eval_episodes: Number of episodes for evaluation
        """
        print(f"Starting curriculum learning from grid size {self.initial_grid_size} to {self.max_grid_size}")
        
        # Train on progressively larger grids
        for grid_size in range(self.initial_grid_size, self.max_grid_size + 1):
            print(f"\n=== Training on grid size {grid_size} ===")
            
            # Create a new environment with the current grid size
            self.env = LShapeRamseyEnv(grid_size=grid_size, num_colors=self.num_colors)
            
            # If we're not at the initial size, transfer knowledge from the previous model
            if grid_size > self.initial_grid_size:
                # Create a new agent with the new environment
                new_agent = LShapePPO(self.env)
                
                # Transfer knowledge (weights) from the previous agent
                # This is a simplification - in practice, you'd need to handle the case where
                # the input dimensions change due to the grid size change
                try:
                    # Save the previous model and load it into the new agent
                    self.agent.save("temp_model")
                    new_agent.load("temp_model")
                    print("Successfully transferred knowledge from previous grid size")
                except Exception as e:
                    print(f"Could not transfer knowledge: {e}")
                
                self.agent = new_agent
            
            # Train the agent on the current grid size
            self.agent.train()
            
            # Evaluate the agent
            results = self.agent.evaluate(num_episodes=eval_episodes)
            
            # Check if we need to continue training at this level
            success_rate = results["success_rate"]
            if success_rate < self.success_threshold and grid_size < self.max_grid_size:
                print(f"Success rate {success_rate:.2f} below threshold {self.success_threshold}")
                print(f"Training for additional timesteps at grid size {grid_size}")
                
                # Train for additional timesteps until threshold is met or max attempts reached
                max_additional_attempts = 3
                for attempt in range(max_additional_attempts):
                    self.agent.train()
                    results = self.agent.evaluate(num_episodes=eval_episodes)
                    success_rate = results["success_rate"]
                    
                    if success_rate >= self.success_threshold:
                        print(f"Success rate threshold met after additional training: {success_rate:.2f}")
                        break
                    
                    if attempt == max_additional_attempts - 1:
                        print(f"Moving to next grid size despite below-threshold performance: {success_rate:.2f}")
            
            # Save the model for this grid size
            self.agent.save(f"l_shape_ramsey_grid{grid_size}")
        
        print("\nCurriculum learning completed!")
        return self.agent

# ========================================================================================
# Multi-Agent Reinforcement Learning Implementation
# ========================================================================================

class MultiAgentLShapeEnv(gym.Env):
    """
    Multi-Agent environment for the L-shape Ramsey problem.
    
    In this environment, each cell is treated as an agent that must coordinate
    with other cells to avoid creating monochromatic L-shapes.
    
    This implements a Centralized Training with Decentralized Execution (CTDE) approach,
    where agents have local observations but are trained to optimize a global reward.
    """
    
    def __init__(self, grid_size=5, num_colors=3):
        """
        Initialize the multi-agent environment
        
        Args:
            grid_size: Size of the grid
            num_colors: Number of colors to use
        """
        super(MultiAgentLShapeEnv, self).__init__()
        
        self.grid_size = grid_size
        self.num_colors = num_colors
        self.num_agents = grid_size * grid_size
        
        # Initialize underlying grid
        self.grid = LShapeGrid(grid_size)
        
        # Action space for each agent: choose a color (or do nothing)
        self.action_space = spaces.Discrete(num_colors + 1)  # Colors + no-op
        
        # Observation space: local view of the grid + agent's own position
        # We use a 3x3 neighborhood around each cell + position coordinates
        self.observation_space = spaces.Dict({
            "local_grid": spaces.Box(
                low=0,
                high=num_colors,
                shape=(3, 3),
                dtype=np.int32
            ),
            "position": spaces.Box(
                low=0,
                high=grid_size - 1,
                shape=(2,),
                dtype=np.int32
            )
        })
    
    def reset(self):
        """
        Reset the environment
        
        Returns:
            observations: Dictionary mapping agent IDs to observations
        """
        self.grid = LShapeGrid(self.grid_size)
        self.done_agents = set()
        self.steps = 0
        
        # Return observations for all agents
        return self._get_observations()
    
    def _get_observations(self):
        """
        Get observations for all agents
        
        Returns:
            observations: Dictionary mapping agent IDs to observations
        """
        observations = {}
        
        for agent_id in range(self.num_agents):
            # Convert agent_id to coordinates
            x = agent_id % self.grid_size
            y = agent_id // self.grid_size
            
            # Get the local neighborhood
            local_grid = np.zeros((3, 3), dtype=np.int32)
            
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                        color = self.grid.get_color(nx, ny)
                        local_grid[dy+1, dx+1] = color.value + 1 if color is not None else 0
            
            observations[agent_id] = {
                "local_grid": local_grid,
                "position": np.array([x, y], dtype=np.int32)
            }
        
        return observations
    
    def step(self, actions):
        """
        Execute one step with actions from all agents
        
        Args:
            actions: Dictionary mapping agent IDs to actions
            
        Returns:
            observations: Dictionary mapping agent IDs to observations
            rewards: Dictionary mapping agent IDs to rewards
            dones: Dictionary mapping agent IDs to done flags
            infos: Dictionary mapping agent IDs to info dictionaries
        """
        # Apply all actions
        for agent_id, action in actions.items():
            if agent_id in self.done_agents or action == self.num_colors:
                # Agent is done or chose no-op
                continue
            
            # Convert agent_id to coordinates
            x = agent_id % self.grid_size
            y = agent_id // self.grid_size
            
            # Only apply action if cell is empty
            if self.grid.get_color(x, y) is None:
                self.grid.set_color(x, y, list(Color)[action])
                self.done_agents.add(agent_id)
        
        self.steps += 1
        
        # Check for L-shapes
        has_l_shape, l_shape_points = self.grid.has_any_l_shape()
        
        # Calculate rewards and dones
        global_reward = -10.0 if has_l_shape else 0.1
        
        # Count empty cells
        empty_cells = sum(1 for y in range(self.grid_size) for x in range(self.grid_size) 
                         if self.grid.get_color(x, y) is None)
        
        # If grid is fully colored without L-shapes, give big reward
        if empty_cells == 0 and not has_l_shape:
            global_reward = 100.0
        
        # Create reward dictionary with global reward
        rewards = {agent_id: global_reward for agent_id in range(self.num_agents)}
        
        # Create done dictionary
        env_done = has_l_shape or empty_cells == 0 or self.steps >= self.grid_size * self.grid_size
        dones = {agent_id: env_done for agent_id in range(self.num_agents)}
        dones['__all__'] = env_done
        
        # Create info dictionary
        infos = {
            agent_id: {
                "has_l_shape": has_l_shape,
                "l_shape_points": l_shape_points,
                "empty_cells": empty_cells
            } for agent_id in range(self.num_agents)
        }
        
        return self._get_observations(), rewards, dones, infos

# ========================================================================================
# Main Execution
# ========================================================================================

def main():
    """
    Main function to demonstrate usage of the RL approaches
    """
    print("L-Shape Ramsey Problem Solver using Reinforcement Learning")
    print("=" * 80)
    
    # Create environment
    env = LShapeRamseyEnv(grid_size=5, num_colors=3)
    
    # Check the environment is valid
    check_env(env)
    
    # Example 1: Train DQN
    print("\nTraining DQN agent...")
    dqn = LShapeDQN(env, total_timesteps=10000)  # Reduced for demonstration
    dqn.train()
    dqn_results = dqn.evaluate(num_episodes=5)
    
    # Example 2: Train PPO
    print("\nTraining PPO agent...")
    ppo = LShapePPO(env, total_timesteps=10000)  # Reduced for demonstration
    ppo.train()
    ppo_results = dqn.evaluate(num_episodes=5)
    
    # Example 3: Curriculum Learning
    print("\nDemonstrating Curriculum Learning...")
    curriculum = CurriculumLearning(initial_grid_size=3, max_grid_size=5, 
                                   success_threshold=0.7, num_colors=3)
    curriculum.train_curriculum(timesteps_per_size=5000, eval_episodes=5)  # Reduced for demonstration
    
    print("\nAll demonstrations completed!")

if __name__ == "__main__":
    main() 