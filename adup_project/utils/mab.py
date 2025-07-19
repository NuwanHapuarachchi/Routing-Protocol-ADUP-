"""
Multi-Armed Bandit implementation for ADUP intelligent routing.
Uses Epsilon-Greedy algorithm for exploration vs exploitation in path selection.
"""

import random
import math
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import time


@dataclass
class PathMetrics:
    """Metrics for a network path."""
    delay: float = 0.0          # Average delay in milliseconds
    jitter: float = 0.0         # Average jitter in microseconds
    packet_loss: float = 0.0    # Packet loss percentage (0-100)
    congestion: float = 0.0     # Congestion level (0-100)
    bandwidth: float = 0.0      # Available bandwidth (optional)
    
    def __post_init__(self):
        """Ensure all metrics are within valid ranges."""
        self.delay = max(0.0, self.delay)
        self.jitter = max(0.0, self.jitter)
        self.packet_loss = max(0.0, min(100.0, self.packet_loss))
        self.congestion = max(0.0, min(100.0, self.congestion))
        self.bandwidth = max(0.0, self.bandwidth)


@dataclass
class PathPerformance:
    """Performance tracking for a path."""
    total_reward: float = 0.0
    num_selections: int = 0
    last_used: float = 0.0
    success_rate: float = 0.0
    average_latency: float = 0.0
    
    @property
    def average_reward(self) -> float:
        """Calculate average reward for this path."""
        if self.num_selections == 0:
            return 0.0
        return self.total_reward / self.num_selections
    
    def update(self, reward: float, latency: float = 0.0, success: bool = True):
        """Update performance metrics with new observation."""
        self.total_reward += reward
        self.num_selections += 1
        self.last_used = time.time()
        
        # Update success rate using exponential moving average
        alpha = 0.1  # Learning rate
        if self.num_selections == 1:
            self.success_rate = 1.0 if success else 0.0
        else:
            self.success_rate = (1 - alpha) * self.success_rate + alpha * (1.0 if success else 0.0)
        
        # Update average latency
        if self.num_selections == 1:
            self.average_latency = latency
        else:
            self.average_latency = (1 - alpha) * self.average_latency + alpha * latency


class MultiArmedBandit:
    """
    Multi-Armed Bandit for intelligent path selection in ADUP.
    
    Uses Epsilon-Greedy algorithm to balance exploration vs exploitation
    when multiple paths are available with similar costs.
    """
    
    def __init__(self, 
                 epsilon: float = 0.1,
                 decay_rate: float = 0.99,
                 min_epsilon: float = 0.01,
                 confidence_threshold: int = 10):
        """
        Initialize the Multi-Armed Bandit.
        
        Args:
            epsilon: Initial exploration rate (0.0 = pure exploitation, 1.0 = pure exploration)
            decay_rate: Rate at which epsilon decays over time
            min_epsilon: Minimum epsilon value to maintain some exploration
            confidence_threshold: Minimum samples needed before trusting average reward
        """
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.decay_rate = decay_rate
        self.min_epsilon = min_epsilon
        self.confidence_threshold = confidence_threshold
        
        # Path performance tracking
        self.path_performance: Dict[str, PathPerformance] = defaultdict(PathPerformance)
        
        # Reward weights for different metrics (configurable)
        self.metric_weights = {
            'delay': 0.3,
            'jitter': 0.2,
            'packet_loss': 0.3,
            'congestion': 0.2
        }
        
        # Statistics
        self.total_selections = 0
        self.exploration_count = 0
        self.exploitation_count = 0
    
    def select_arm(self, available_arms: List[str]) -> int:
        """Select an arm using epsilon-greedy for fast convergence."""
        if not available_arms:
            return 0
        
        # Ensure all arms are tracked
        for arm in available_arms:
            if arm not in self.path_performance:
                self.path_performance[arm] = PathPerformance()
        
        self.total_selections += 1
        
        # Epsilon-greedy selection
        if random.random() < self.epsilon:
            # Exploration: select random arm
            selected_idx = random.randint(0, len(available_arms) - 1)
            self.exploration_count += 1
        else:
            # Exploitation: select arm with highest average reward
            best_idx = 0
            best_reward = float('-inf')
            
            for i, arm in enumerate(available_arms):
                performance = self.path_performance[arm]
                if performance.num_selections < self.confidence_threshold:
                    # UCB for insufficient data
                    if performance.num_selections == 0:
                        ucb_score = float('inf')
                    else:
                        confidence = (2 * math.log(self.total_selections) / performance.num_selections) ** 0.5
                        ucb_score = performance.average_reward + confidence
                else:
                    ucb_score = performance.average_reward
                
                if ucb_score > best_reward:
                    best_reward = ucb_score
                    best_idx = i
            
            selected_idx = best_idx
            self.exploitation_count += 1
        
        # Decay epsilon for faster convergence
        self._decay_epsilon()
        
        return selected_idx

    def update_arm(self, arm_id: str, reward: float):
        """Update arm performance for fast learning."""
        if arm_id not in self.path_performance:
            self.path_performance[arm_id] = PathPerformance()
        
        self.path_performance[arm_id].update(reward, success=reward > 0.1)

    def select_path(self, available_paths: List[str], 
                   path_metrics: Optional[Dict[str, PathMetrics]] = None) -> str:
        """
        Select the best path using Epsilon-Greedy algorithm.
        
        Args:
            available_paths: List of available path IDs
            path_metrics: Current metrics for each path (optional)
        
        Returns:
            Selected path ID
        """
        if not available_paths:
            raise ValueError("No available paths provided")
        
        # Ensure all paths are added to the bandit
        for path_id in available_paths:
            if path_id not in self.path_performance:
                metrics = path_metrics.get(path_id) if path_metrics else None
                self.add_path(path_id, metrics)
        
        self.total_selections += 1
        
        # Epsilon-greedy selection
        if random.random() < self.epsilon:
            # Exploration: select random path
            selected_path = random.choice(available_paths)
            self.exploration_count += 1
        else:
            # Exploitation: select path with highest average reward
            selected_path = self._select_best_path(available_paths)
            self.exploitation_count += 1
        
        # Decay epsilon
        self._decay_epsilon()
        
        return selected_path
    
    def update_reward(self, path_id: str, metrics: PathMetrics, 
                     success: bool = True, latency: float = 0.0):
        """
        Update the reward for a path based on observed performance.
        
        Args:
            path_id: Path identifier
            metrics: Observed metrics for the path
            success: Whether the transmission was successful
            latency: Observed latency
        """
        if path_id not in self.path_performance:
            self.add_path(path_id)
        
        # Calculate reward based on metrics
        reward = self._calculate_reward(metrics)
        
        # Apply penalty for failure
        if not success:
            reward *= 0.1  # Severe penalty for failed transmissions
        
        # Update path performance
        self.path_performance[path_id].update(reward, latency, success)
    
    def add_path(self, path_id: str, initial_metrics: Optional[PathMetrics] = None):
        """Add a new path to the bandit."""
        if path_id not in self.path_performance:
            self.path_performance[path_id] = PathPerformance()
            
            # If initial metrics provided, calculate initial reward
            if initial_metrics:
                initial_reward = self._calculate_reward(initial_metrics)
                self.path_performance[path_id].total_reward = initial_reward
                self.path_performance[path_id].num_selections = 1
    
    def _select_best_path(self, available_paths: List[str]) -> str:
        """Select the path with the highest average reward."""
        best_path = None
        best_score = float('-inf')
        
        for path_id in available_paths:
            performance = self.path_performance[path_id]
            
            if performance.num_selections < self.confidence_threshold:
                # Use UCB for paths with insufficient data
                if performance.num_selections == 0:
                    ucb_score = float('inf')  # Unvisited paths get highest priority
                else:
                    confidence = np.sqrt(2 * np.log(self.total_selections) / performance.num_selections)
                    ucb_score = performance.average_reward + confidence
            else:
                # Use average reward for well-explored paths
                ucb_score = performance.average_reward
            
            if ucb_score > best_score:
                best_score = ucb_score
                best_path = path_id
        
        return best_path if best_path else available_paths[0]
    
    def _calculate_reward(self, metrics: PathMetrics) -> float:
        """Calculate reward based on path metrics."""
        # Normalize metrics to 0-1 range (inverted for delay, jitter, loss, congestion)
        normalized_delay = max(0, 1.0 - min(1.0, metrics.delay / 1000.0))  # Assume max 1000ms
        normalized_jitter = max(0, 1.0 - min(1.0, metrics.jitter / 10000.0))  # Assume max 10000μs
        normalized_loss = max(0, 1.0 - metrics.packet_loss / 100.0)
        normalized_congestion = max(0, 1.0 - metrics.congestion / 100.0)
        
        # Calculate weighted reward
        reward = (
            self.metric_weights['delay'] * normalized_delay +
            self.metric_weights['jitter'] * normalized_jitter +
            self.metric_weights['packet_loss'] * normalized_loss +
            self.metric_weights['congestion'] * normalized_congestion
        )
        
        return reward
    
    def _decay_epsilon(self):
        """Decay epsilon to reduce exploration over time."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)
