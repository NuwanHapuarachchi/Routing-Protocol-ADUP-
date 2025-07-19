"""
Base router class with common functionality for all routing protocols.
Provides shared attributes and methods used by ADUP, RIP, and OSPF implementations.
"""

import simpy
import networkx as nx
import pandas as pd
import random
import time
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class RoutingTableEntry:
    """Generic routing table entry."""
    destination: str            # Destination network/host
    next_hop: str              # Next hop router ID
    metric: float              # Path cost/metric
    interface: str             # Outgoing interface
    timestamp: float = 0.0     # Last update time
    source: str = ""           # Source of route information
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class LinkMetrics:
    """Link metrics for dynamic cost calculation."""
    base_delay: float = 10.0       # Base delay in milliseconds
    base_jitter: float = 1.0       # Base jitter in microseconds
    base_packet_loss: float = 0.0  # Base packet loss percentage
    base_congestion: float = 0.0   # Base congestion level
    bandwidth: float = 100.0       # Link bandwidth in Mbps
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current metrics with random variation to simulate real conditions."""
        return {
            'delay': max(0.1, self.base_delay + random.uniform(-2.0, 5.0)),
            'jitter': max(0.0, self.base_jitter + random.uniform(-0.5, 2.0)),
            'packet_loss': max(0.0, min(100.0, self.base_packet_loss + random.uniform(-0.1, 1.0))),
            'congestion': max(0.0, min(100.0, self.base_congestion + random.uniform(-5.0, 10.0))),
            'bandwidth': max(1.0, self.bandwidth + random.uniform(-10.0, 10.0))
        }


class BaseRouter(ABC):
    """Base router class providing common functionality for all routing protocols."""
    
    def __init__(self, 
                 env: simpy.Environment,
                 router_id: str,
                 network_graph: nx.Graph,
                 metric_weights: Optional[Dict[str, float]] = None):
        """Initialize base router."""
        self.env = env
        self.router_id = router_id
        self.network_graph = network_graph
        
        # Routing table: destination -> RoutingTableEntry
        self.routing_table: Dict[str, RoutingTableEntry] = {}
        
        # Neighbor information
        self.neighbors: Set[str] = set()
        self.neighbor_interfaces: Dict[str, str] = {}  # neighbor_id -> interface_name
        self.neighbor_metrics: Dict[str, LinkMetrics] = {}
        
        # Protocol configuration
        self.metric_weights = metric_weights or {
            'delay': 0.3,
            'jitter': 0.2,
            'packet_loss': 0.3,
            'congestion': 0.2
        }
        
        # Security
        self.pre_shared_key = "adup_secure_key_2024"
        
        # Statistics
        self.packets_sent = 0
        self.packets_received = 0
        self.packets_dropped = 0
        self.convergence_time = 0.0
        self.last_convergence_start = 0.0
        
        # Initialize neighbors and interfaces
        self._initialize_neighbors()
    
    def _initialize_neighbors(self):
        """Initialize neighbor information from network graph."""
        if self.router_id in self.network_graph:
            for neighbor in self.network_graph.neighbors(self.router_id):
                self.neighbors.add(neighbor)
                
                # Create interface name
                interface_name = f"eth_{neighbor}"
                self.neighbor_interfaces[neighbor] = interface_name
                
                # Initialize link metrics from graph edge data
                edge_data = self.network_graph[self.router_id][neighbor]
                self.neighbor_metrics[neighbor] = LinkMetrics(
                    base_delay=edge_data.get('delay', 10.0),
                    base_jitter=edge_data.get('jitter', 1.0),
                    base_packet_loss=edge_data.get('packet_loss', 0.0),
                    base_congestion=edge_data.get('congestion', 0.0),
                    bandwidth=edge_data.get('bandwidth', 100.0)
                )
    
    @abstractmethod
    def run(self):
        """Main protocol process - must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def handle_neighbor_down(self, neighbor_id: str):
        """Handle neighbor failure - must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def handle_neighbor_up(self, neighbor_id: str):
        """Handle neighbor recovery - must be implemented by subclasses."""
        pass
    
    def calculate_composite_metric(self, metrics: Dict[str, float]) -> float:
        """Calculate composite path metric using weighted sum."""
        # Normalize metrics to same scale
        normalized_delay = metrics.get('delay', 0.0) / 1000.0  # Convert to seconds
        normalized_jitter = metrics.get('jitter', 0.0) / 10000.0  # Normalize jitter
        normalized_loss = metrics.get('packet_loss', 0.0) / 100.0  # Percentage to ratio
        normalized_congestion = metrics.get('congestion', 0.0) / 100.0  # Percentage to ratio
        
        # Calculate composite metric (lower is better)
        composite = (
            self.metric_weights['delay'] * normalized_delay +
            self.metric_weights['jitter'] * normalized_jitter +
            self.metric_weights['packet_loss'] * normalized_loss +
            self.metric_weights['congestion'] * normalized_congestion
        )
        
        return composite * 1000  # Scale for easier handling
    
    def log_event(self, event_type: str, event_data: Dict[str, Any]):
        """Log simulation events for analysis."""
        # This will be handled by the SimulationManager
        pass
    
    def start_convergence_timer(self):
        """Start convergence timing."""
        self.last_convergence_start = self.env.now
    
    def stop_convergence_timer(self):
        """Stop convergence timing and update convergence time."""
        if self.last_convergence_start > 0:
            self.convergence_time = self.env.now - self.last_convergence_start
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get router statistics for analysis."""
        return {
            'router_id': self.router_id,
            'packets_sent': self.packets_sent,
            'packets_received': self.packets_received,
            'packets_dropped': self.packets_dropped,
            'routing_table_size': len(self.routing_table),
            'neighbor_count': len(self.neighbors),
            'convergence_time': self.convergence_time
        }
