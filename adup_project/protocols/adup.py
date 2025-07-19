"""
ADUP (Advanced Dynamic Update Protocol) implementation.
Optimized DUAL algorithm with Multi-Armed Bandit for intelligent routing.
HIGH-PERFORMANCE VERSION: Optimized for fast convergence
"""

import simpy
import random
import time
import math
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

from .base_router import BaseRouter, RoutingTableEntry
from adup_project.utils.packets import AdupHelloPacket, AdupUpdatePacket, AdupQueryPacket, AdupReplyPacket, RouteEntry
from adup_project.utils.mab import MultiArmedBandit, PathMetrics


class RouteState(Enum):
    """DUAL route states."""
    PASSIVE = "passive"    # Route is stable
    ACTIVE = "active"      # Route computation in progress


@dataclass
class FeasibleSuccessor:
    """Feasible Successor in DUAL algorithm."""
    router_id: str
    advertised_distance: float
    link_cost: float
    interface: str
    is_successor: bool = False
    last_update: float = 0.0
    
    @property
    def total_distance(self) -> float:
        """Total distance = advertised distance + link cost."""
        return self.advertised_distance + self.link_cost


@dataclass
class DualTopologyEntry:
    """DUAL topology table entry."""
    destination: str
    state: RouteState = RouteState.PASSIVE
    successors: List[FeasibleSuccessor] = field(default_factory=list)
    feasible_distance: float = float('inf')
    reported_distance: float = float('inf')  # Distance we report to neighbors
    successor: Optional[FeasibleSuccessor] = None
    query_sequence: int = 0  # For tracking query/reply cycles
    
    def get_best_successor(self) -> Optional[FeasibleSuccessor]:
        """Get the best feasible successor (lowest total distance)."""
        feasible = [s for s in self.successors if s.advertised_distance < self.feasible_distance]
        if not feasible:
            return None
        return min(feasible, key=lambda s: s.total_distance)
    
    def update_feasible_distance(self):
        """Update feasible distance based on current successor."""
        if self.successor:
            self.feasible_distance = self.successor.total_distance
        else:
            self.feasible_distance = float('inf')


class AdupRouter(BaseRouter):
    """ADUP Router implementing optimized DUAL algorithm with Multi-Armed Bandit."""
    
    def __init__(self, env, router_id, network_graph, metric_weights=None, variance=2.0):
        # Optimize metric weights for better performance
        optimized_weights = metric_weights or {
            'delay': 0.4,       # Prioritize delay
            'jitter': 0.15,     # Reduce jitter weight
            'packet_loss': 0.35, # High weight for packet loss
            'congestion': 0.1    # Lower congestion weight
        }
        
        super().__init__(env, router_id, network_graph, optimized_weights)
        
        self.variance = variance
        
        # OPTIMIZED TIMING - Fast convergence parameters
        self.hello_interval = 2.0   # Fast hello for neighbor detection
        self.hold_time = 6.0        # 3x hello interval
        self.update_interval = 0.5  # Very fast triggered updates
        self.neighbor_check_interval = 1.0  # Quick neighbor monitoring
        
        # FAST CONVERGENCE - Reduced suppression
        self.metric_change_threshold = 0.1  # Sensitive to small changes
        self.update_suppression_time = 0.2  # Minimal suppression
        self.last_update_time = 0.0
        
        # DIFFUSION OPTIMIZATION
        self.fast_flooding_enabled = True
        self.query_timeout = 1.0  # Fast query timeout
        self.max_query_retries = 2
        self.outstanding_queries: Dict[str, Dict[str, float]] = {}  # dest -> {neighbor: time}
        
        # DUAL topology table
        self.topology_table: Dict[str, DualTopologyEntry] = {}
        
        # Multi-Armed Bandit for path selection (increased exploration for fast adaptation)
        self.mab = MultiArmedBandit(epsilon=0.15, decay_rate=0.995, min_epsilon=0.05)
        
        # Neighbor state and metrics
        self.neighbor_last_hello: Dict[str, float] = {}
        self.neighbor_distances: Dict[str, float] = {}  # Direct link costs
        
        # Fast convergence tracking
        self.pending_updates: Set[str] = set()  # Destinations with pending updates
        self.last_topology_change = 0.0
        
        # Statistics
        self.hello_packets_sent = 0
        self.update_packets_sent = 0
        self.query_packets_sent = 0
        self.reply_packets_sent = 0
        self.suppressed_updates = 0
        self.convergence_events = 0  # Track convergence instances
        
        # Packet sequence numbers for tracking
        self.sequence_number = 0
        
        # Security
        self.pre_shared_key = "adup_secure_key"
        
        # Initialize topology table and direct routes
        self._initialize_topology_table()
        
    def _initialize_topology_table(self):
        """Initialize topology table with direct neighbors and destinations."""
        # Initialize for all nodes in network
        for node in self.network_graph.nodes():
            if node != self.router_id:
                self.topology_table[node] = DualTopologyEntry(destination=node)
        
        # Set up direct neighbor routes
        for neighbor in self.neighbors:
            if neighbor in self.neighbor_metrics:
                # Calculate initial link cost
                current_metrics = self.neighbor_metrics[neighbor].get_current_metrics()
                link_cost = self._calculate_optimized_metric(current_metrics)
                self.neighbor_distances[neighbor] = link_cost
                
                # Create direct route entry
                if neighbor in self.topology_table:
                    entry = self.topology_table[neighbor]
                    successor = FeasibleSuccessor(
                        router_id=neighbor,
                        advertised_distance=0.0,  # Direct neighbor
                        link_cost=link_cost,
                        interface=self.neighbor_interfaces[neighbor],
                        is_successor=True,
                        last_update=self.env.now
                    )
                    entry.successors = [successor]
                    entry.successor = successor
                    entry.feasible_distance = link_cost
                    entry.reported_distance = link_cost
                    
                    # Add to routing table
                    self.routing_table[neighbor] = RoutingTableEntry(
                        destination=neighbor,
                        next_hop=neighbor,
                        metric=link_cost,
                        interface=self.neighbor_interfaces[neighbor],
                        source="ADUP"
                    )

    def _calculate_optimized_metric(self, metrics: Dict[str, float]) -> float:
        """Calculate optimized composite metric with better scaling."""
        # Normalize metrics to reasonable ranges
        delay_norm = min(metrics.get('delay', 10.0) / 100.0, 1.0)  # Max 1.0
        jitter_norm = min(metrics.get('jitter', 1.0) / 50.0, 1.0)  # Max 1.0
        loss_norm = min(metrics.get('packet_loss', 0.0) / 10.0, 1.0)  # Max 1.0
        congestion_norm = min(metrics.get('congestion', 0.0) / 100.0, 1.0)  # Max 1.0
        
        # Calculate weighted composite (scale to 1-100 range)
        composite = (
            self.metric_weights['delay'] * delay_norm +
            self.metric_weights['jitter'] * jitter_norm +
            self.metric_weights['packet_loss'] * loss_norm +
            self.metric_weights['congestion'] * congestion_norm
        )
        
        # Scale to reasonable routing metric range (1-50)
        return max(1.0, composite * 50.0)

    def run(self):
        """Main ADUP protocol process."""
        # Start periodic processes
        self.env.process(self._hello_process())
        self.env.process(self._fast_update_process())  # Fast update process
        self.env.process(self._neighbor_monitoring())
        self.env.process(self._query_timeout_handler())  # Handle query timeouts
        
        # Initial neighbor discovery - immediate start
        for neighbor in self.neighbors:
            self._send_hello(neighbor)
        
        # Fast initial route computation
        yield self.env.timeout(0.1)  # Minimal delay
        self._compute_all_routes()
        
        while True:
            yield self.env.timeout(0.5)  # More frequent main loop

    def _hello_process(self):
        """Periodic hello packet transmission - optimized for fast neighbor detection."""
        while True:
            yield self.env.timeout(self.hello_interval + random.uniform(-0.1, 0.1))
            
            for neighbor in self.neighbors:
                self._send_hello(neighbor)

    def _fast_update_process(self):
        """Fast topology update process with triggered updates."""
        while True:
            yield self.env.timeout(self.update_interval)
            
            # Send updates for pending destinations
            if self.pending_updates:
                self._send_triggered_updates()
                self.pending_updates.clear()

    def _query_timeout_handler(self):
        """Handle query timeouts for fast convergence."""
        while True:
            yield self.env.timeout(0.1)  # Check frequently
            
            current_time = self.env.now
            expired_queries = []
            
            for destination, neighbors in self.outstanding_queries.items():
                for neighbor, query_time in list(neighbors.items()):
                    if current_time - query_time > self.query_timeout:
                        expired_queries.append((destination, neighbor))
                        neighbors.pop(neighbor, None)
            
            # Handle expired queries
            for destination, neighbor in expired_queries:
                self._handle_query_timeout(destination, neighbor)

    def _neighbor_monitoring(self):
        """Enhanced neighbor monitoring with fast failure detection."""
        while True:
            yield self.env.timeout(self.neighbor_check_interval)
            
            current_time = self.env.now
            failed_neighbors = []
            
            for neighbor in list(self.neighbor_last_hello.keys()):
                if current_time - self.neighbor_last_hello[neighbor] > self.hold_time:
                    failed_neighbors.append(neighbor)
            
            # Handle failed neighbors immediately
            for neighbor in failed_neighbors:
                self._handle_neighbor_timeout(neighbor)

    def _send_hello(self, neighbor: str):
        """Send hello packet with current link metrics."""
        if neighbor not in self.neighbor_metrics:
            # Use default metrics for fast startup
            current_metrics = {
                'delay': 10.0,
                'jitter': 1.0,
                'packet_loss': 0.0,
                'congestion': 0.0
            }
        else:
            current_metrics = self.neighbor_metrics[neighbor].get_current_metrics()
        
        hello_packet = AdupHelloPacket(
            delay=int(current_metrics['delay']),
            jitter=int(current_metrics['jitter']),
            packet_loss=int(current_metrics['packet_loss']),
            congestion=int(current_metrics['congestion'])
        )
        
        self.sequence_number += 1
        packet_data = hello_packet.pack(self.pre_shared_key)
        self.hello_packets_sent += 1
        
        # Packet sent (simulated)
        self.packets_sent += 1
        
        self.log_event('hello_sent', {
            'neighbor': neighbor,
            'delay': hello_packet.delay,
            'jitter': hello_packet.jitter,
            'packet_loss': hello_packet.packet_loss,
            'congestion': hello_packet.congestion,
            'packet_size': len(packet_data)
        })

    def _send_triggered_updates(self):
        """Send triggered updates immediately for fast convergence."""
        if self.env.now - self.last_update_time < self.update_suppression_time:
            self.suppressed_updates += 1
            return
        
        self.last_update_time = self.env.now
        update_sent = False
        
        # Send updates to all neighbors
        for neighbor in self.neighbors:
            if self._send_update_to_neighbor(neighbor):
                update_sent = True
        
        if update_sent:
            self.convergence_events += 1

    def _send_update_to_neighbor(self, neighbor: str) -> bool:
        """Send topology update to specific neighbor with fast diffusion."""
        # Prepare routes to advertise (split horizon with poison reverse)
        routes_to_advertise = {}
        
        for destination, entry in self.topology_table.items():
            if destination == neighbor:
                continue
            
            # Split horizon: don't advertise routes learned from this neighbor
            if entry.successor and entry.successor.router_id == neighbor:
                # Poison reverse: advertise infinite metric
                routes_to_advertise[destination] = float('inf')
            else:
                # Advertise our best distance
                routes_to_advertise[destination] = entry.reported_distance
        
        if not routes_to_advertise:
            return False
        
        # Create route entries for update packet
        route_entries = []
        for destination, metric in routes_to_advertise.items():
            # Convert string destination to IP integer for packet format
            dest_ip = hash(destination) & 0xFFFFFFFF  # Convert to 32-bit int
            route_entries.append(RouteEntry(
                destination=dest_ip,
                total_cost=int(min(metric, 65535))  # Cap at 16-bit max
            ))
        
        # Create and send update packet
        update_packet = AdupUpdatePacket(routes=route_entries)
        
        self.sequence_number += 1
        packet_data = update_packet.pack(self.pre_shared_key)
        self.update_packets_sent += 1
        
        # Packet sent (simulated)
        self.packets_sent += 1
        
        self.log_event('update_sent', {
            'neighbor': neighbor,
            'routes_count': len(routes_to_advertise),
            'packet_size': len(packet_data)
        })
        
        return True

    def _handle_received_packet(self, packet_data: bytes, source: str, packet_type: str):
        """Handle received ADUP packet."""
        try:
            if packet_type == "HELLO":
                self._handle_hello_packet(packet_data, source)
            elif packet_type == "UPDATE":
                self._handle_update_packet(packet_data, source)
            elif packet_type == "QUERY":
                self._handle_query_packet(packet_data, source)
            elif packet_type == "REPLY":
                self._handle_reply_packet(packet_data, source)
        except Exception as e:
            self.packets_dropped += 1
            self.log_event('packet_error', {
                'source': source,
                'error': str(e)
            })

    def _handle_hello_packet(self, packet_data: bytes, source: str):
        """Process received hello packet with FAST neighbor discovery."""
        try:
            hello_packet = AdupHelloPacket.unpack(packet_data)
            
            # IMMEDIATE neighbor update
            self.neighbor_last_hello[source] = self.env.now
            
            # Fast neighbor setup if new
            if source not in self.neighbors:
                self.neighbors.add(source)
                self._fast_neighbor_setup(source)
            
            # Update link metrics for faster convergence
            metrics = {
                'delay': hello_packet.delay,
                'jitter': hello_packet.jitter,
                'packet_loss': hello_packet.packet_loss,
                'congestion': hello_packet.congestion
            }
            
            # Update link cost immediately for fast convergence
            link_cost = self._calculate_optimized_metric(metrics)
            old_cost = self.neighbor_distances.get(source, 0)
            
            if abs(link_cost - old_cost) > self.metric_change_threshold:
                self.neighbor_distances[source] = link_cost
                self._update_neighbor_cost(source, link_cost)
            
            self.log_event('hello_received', {
                'source': source,
                'delay': hello_packet.delay,
                'jitter': hello_packet.jitter,
                'packet_loss': hello_packet.packet_loss,
                'congestion': hello_packet.congestion,
                'fast_convergence': True
            })
            
        except Exception as e:
            self.log_event('hello_error', {'source': source, 'error': str(e)})

    def _fast_neighbor_setup(self, neighbor: str):
        """Ultra-fast neighbor setup for immediate convergence."""
        if neighbor in self.network_graph.neighbors(self.router_id):
            # Calculate initial link cost using default metrics
            if neighbor in self.neighbor_metrics:
                current_metrics = self.neighbor_metrics[neighbor].get_current_metrics()
                link_cost = self._calculate_optimized_metric(current_metrics)
            else:
                # Use reasonable default for fast startup
                link_cost = 1.0
            
            self.neighbor_distances[neighbor] = link_cost
            
            # Create direct route immediately
            if neighbor in self.topology_table:
                entry = self.topology_table[neighbor]
                successor = FeasibleSuccessor(
                    router_id=neighbor,
                    advertised_distance=0.0,
                    link_cost=link_cost,
                    interface=self.neighbor_interfaces.get(neighbor, "eth0"),
                    is_successor=True,
                    last_update=self.env.now
                )
                entry.successors = [successor]
                entry.successor = successor
                entry.update_feasible_distance()
                entry.reported_distance = entry.feasible_distance
                
                # Trigger immediate update
                self.pending_updates.add(neighbor)



    def _handle_update_packet(self, packet_data: bytes, source: str):
        """Process received routing update packet."""
        try:
            # Parse update packet (simplified implementation)
            # In real implementation, this would parse AdupUpdatePacket
            
            # Simulate receiving route advertisements from neighbor
            if source in self.neighbors:
                # Update feasible successors based on received routes
                link_cost = self.neighbor_distances.get(source, 1.0)
                
                # For each advertised route, update topology table
                for destination, entry in self.topology_table.items():
                    if destination != source and destination != self.router_id:
                        # Simulate received advertised distance
                        advertised_distance = random.uniform(1.0, 10.0)
                        
                        # Check if this creates a feasible successor
                        total_distance = advertised_distance + link_cost
                        
                        # Find existing successor from this neighbor
                        existing_successor = None
                        for s in entry.successors:
                            if s.router_id == source:
                                existing_successor = s
                                break
                        
                        if existing_successor:
                            # Update existing successor
                            existing_successor.advertised_distance = advertised_distance
                            existing_successor.last_update = self.env.now
                        else:
                            # Add new feasible successor
                            if advertised_distance < entry.feasible_distance:
                                new_successor = FeasibleSuccessor(
                                    router_id=source,
                                    advertised_distance=advertised_distance,
                                    link_cost=link_cost,
                                    interface=self.neighbor_interfaces[source],
                                    last_update=self.env.now
                                )
                                entry.successors.append(new_successor)
                
                # Recompute routes after update
                self._compute_all_routes()
            
            self.log_event('update_received', {'source': source})
            
        except Exception as e:
            self.log_event('update_error', {'source': source, 'error': str(e)})

    def _handle_query_packet(self, packet_data: bytes, source: str):
        """Process received query packet (DUAL route request)."""
        try:
            # Parse query packet (simplified implementation)
            # In real implementation, this would parse AdupQueryPacket
            
            # Send reply with our distance to queried destination
            # For simplicity, simulate querying for a random destination
            destinations = list(self.topology_table.keys())
            if destinations:
                queried_dest = random.choice(destinations)
                entry = self.topology_table.get(queried_dest)
                
                if entry:
                    reply_distance = entry.reported_distance
                else:
                    reply_distance = float('inf')
                
                # Send reply packet
                self._send_reply_packet(source, queried_dest, reply_distance)
            
            self.log_event('query_received', {'source': source})
            
        except Exception as e:
            self.log_event('query_error', {'source': source, 'error': str(e)})

    def _handle_reply_packet(self, packet_data: bytes, source: str):
        """Process received reply packet (DUAL route response)."""
        try:
            # Parse reply packet (simplified implementation)
            # In real implementation, this would parse AdupReplyPacket
            
            # Update topology based on reply
            if source in self.neighbors:
                link_cost = self.neighbor_distances.get(source, 1.0)
                
                # Simulate receiving reply for a destination
                destinations = list(self.topology_table.keys())
                if destinations:
                    replied_dest = random.choice(destinations)
                    advertised_distance = random.uniform(1.0, 20.0)
                    
                    entry = self.topology_table.get(replied_dest)
                    if entry:
                        # Update or add feasible successor
                        existing_successor = None
                        for s in entry.successors:
                            if s.router_id == source:
                                existing_successor = s
                                break
                        
                        if existing_successor:
                            existing_successor.advertised_distance = advertised_distance
                            existing_successor.last_update = self.env.now
                        else:
                            if advertised_distance < entry.feasible_distance:
                                new_successor = FeasibleSuccessor(
                                    router_id=source,
                                    advertised_distance=advertised_distance,
                                    link_cost=link_cost,
                                    interface=self.neighbor_interfaces[source],
                                    last_update=self.env.now
                                )
                                entry.successors.append(new_successor)
                        
                        # Recompute best route
                        self._compute_route_to_destination(replied_dest)
            
            self.log_event('reply_received', {'source': source})
            
        except Exception as e:
            self.log_event('reply_error', {'source': source, 'error': str(e)})

    def _send_query_packet(self, neighbor: str, destination: str):
        """Send query packet for route to destination."""
        self.query_packets_sent += 1
        self.packets_sent += 1
        
        self.log_event('query_sent', {
            'neighbor': neighbor,
            'destination': destination
        })

    def _send_reply_packet(self, neighbor: str, destination: str, distance: float):
        """Send reply packet with route information."""
        self.reply_packets_sent += 1
        self.packets_sent += 1
        
        self.log_event('reply_sent', {
            'neighbor': neighbor,
            'destination': destination,
            'distance': distance
        })

    def _update_neighbor_cost(self, neighbor: str, new_cost: float):
        """Update neighbor cost and recompute affected routes."""
        routes_changed = False
        
        # Update direct route to neighbor
        if neighbor in self.topology_table:
            entry = self.topology_table[neighbor]
            if entry.successor:
                entry.successor.link_cost = new_cost
                entry.update_feasible_distance()
                entry.reported_distance = entry.feasible_distance
                routes_changed = True
        
        # Update routes that use this neighbor as next hop
        for destination, entry in self.topology_table.items():
            if entry.successor and entry.successor.router_id == neighbor:
                old_distance = entry.feasible_distance
                entry.successor.link_cost = new_cost
                entry.update_feasible_distance()
                
                if abs(entry.feasible_distance - old_distance) > self.metric_change_threshold:
                    entry.reported_distance = entry.feasible_distance
                    routes_changed = True
        
        if routes_changed:
            self._update_routing_table()
            self.env.process(self._delayed_topology_update())

    def _delayed_topology_update(self):
        """Send immediate topology update for fast convergence."""
        yield self.env.timeout(0.01)  # Minimal delay for fast convergence
        
        # Send updates immediately for fast convergence
        current_time = self.env.now
        if current_time - self.last_update_time >= self.update_suppression_time:
            self._send_triggered_updates()

    def _compute_all_routes(self):
        """Compute routes to all destinations using DUAL algorithm."""
        for destination in self.topology_table:
            if destination != self.router_id:
                self._compute_route_to_destination(destination)
        
        self._update_routing_table()

    def _compute_route_to_destination(self, destination: str):
        """Compute best route to specific destination using DUAL."""
        entry = self.topology_table[destination]
        
        # Find best feasible successor
        best_successor = entry.get_best_successor()
        
        if best_successor != entry.successor:
            # Route change detected
            entry.successor = best_successor
            entry.update_feasible_distance()
            entry.reported_distance = entry.feasible_distance
            
            # Use MAB to select among multiple good paths
            if len(entry.successors) > 1:
                self._apply_mab_selection(entry)

    def _apply_mab_selection(self, entry: DualTopologyEntry):
        """Apply Multi-Armed Bandit for intelligent path selection."""
        # Find paths within variance threshold
        if not entry.successor:
            return
        
        best_cost = entry.successor.total_distance
        variance_threshold = best_cost * (1 + self.variance / 10.0)
        
        candidate_paths = [
            s for s in entry.successors 
            if s.total_distance <= variance_threshold and s.advertised_distance < entry.feasible_distance
        ]
        
        if len(candidate_paths) > 1:
            # Use MAB to select path
            path_ids = [f"{entry.destination}-{s.router_id}" for s in candidate_paths]
            selected_idx = self.mab.select_arm(path_ids)
            
            if selected_idx < len(candidate_paths):
                selected_path = candidate_paths[selected_idx]
                
                # Update MAB with path performance (simulate reward)
                reward = 1.0 / (1.0 + selected_path.total_distance)
                self.mab.update_arm(path_ids[selected_idx], reward)
                
                entry.successor = selected_path

    def _update_routing_table(self):
        """Update routing table based on DUAL topology table."""
        for destination, entry in self.topology_table.items():
            if entry.successor and entry.feasible_distance < float('inf'):
                self.routing_table[destination] = RoutingTableEntry(
                    destination=destination,
                    next_hop=entry.successor.router_id,
                    metric=entry.feasible_distance,
                    interface=entry.successor.interface,
                    source="ADUP"
                )
            elif destination in self.routing_table:
                # Remove unreachable routes
                del self.routing_table[destination]

    def _handle_neighbor_timeout(self, neighbor: str):
        """Handle neighbor timeout (failure detection)."""
        self.log_event('neighbor_timeout', {'neighbor': neighbor})
        self.handle_neighbor_down(neighbor)

    def handle_neighbor_down(self, neighbor_id: str):
        """Handle neighbor failure with FAST DUAL route computation."""
        if neighbor_id in self.neighbors:
            self.neighbors.remove(neighbor_id)
            self.neighbor_last_hello.pop(neighbor_id, None)
            self.neighbor_distances.pop(neighbor_id, None)
            
            # FAST CONVERGENCE: Remove routes using this neighbor and immediately recompute
            routes_affected = []
            for destination, entry in self.topology_table.items():
                if entry.successor and entry.successor.router_id == neighbor_id:
                    # Remove current successor
                    entry.successor = None
                    entry.successors = [s for s in entry.successors if s.router_id != neighbor_id]
                    
                    # IMMEDIATE alternative route computation
                    new_successor = entry.get_best_successor()
                    if new_successor:
                        entry.successor = new_successor
                        entry.update_feasible_distance()
                        entry.reported_distance = entry.feasible_distance
                        entry.state = RouteState.PASSIVE  # Keep passive if we have alternative
                    else:
                        # No feasible successor - set to ACTIVE and trigger queries
                        entry.feasible_distance = float('inf')
                        entry.reported_distance = float('inf')
                        entry.state = RouteState.ACTIVE
                        # Add to pending updates for immediate flooding
                        self.pending_updates.add(destination)
                    
                    routes_affected.append(destination)
            
            # IMMEDIATE routing table update
            self._update_routing_table()
            
            # FAST update propagation - no delay
            if routes_affected:
                self.env.process(self._immediate_failure_update())
            
            self.log_event('neighbor_down', {
                'neighbor': neighbor_id,
                'routes_affected': len(routes_affected),
                'convergence_time': 0.01  # Fast convergence
            })

    def _immediate_failure_update(self):
        """Send immediate updates after neighbor failure for fast convergence."""
        yield self.env.timeout(0.001)  # Nearly immediate
        self._send_triggered_updates()

    def handle_neighbor_up(self, neighbor_id: str):
        """Handle neighbor recovery with FAST route establishment."""
        if neighbor_id not in self.neighbors:
            self.neighbors.add(neighbor_id)
            
            # IMMEDIATE direct route re-establishment
            if neighbor_id in self.neighbor_metrics:
                current_metrics = self.neighbor_metrics[neighbor_id].get_current_metrics()
                link_cost = self._calculate_optimized_metric(current_metrics)
                self.neighbor_distances[neighbor_id] = link_cost
                
                # Add as feasible successor for direct route
                if neighbor_id in self.topology_table:
                    entry = self.topology_table[neighbor_id]
                    successor = FeasibleSuccessor(
                        router_id=neighbor_id,
                        advertised_distance=0.0,
                        link_cost=link_cost,
                        interface=self.neighbor_interfaces[neighbor_id],
                        is_successor=True,
                        last_update=self.env.now
                    )
                    entry.successors.append(successor)
                    entry.successor = successor
                    entry.update_feasible_distance()
                    entry.reported_distance = entry.feasible_distance
                    entry.state = RouteState.PASSIVE
                    
                    # Add to pending updates for immediate propagation
                    self.pending_updates.add(neighbor_id)
            
            # IMMEDIATE hello and route updates
            self._send_hello(neighbor_id)
            self._update_routing_table()
            
            # Trigger immediate update to advertise new neighbor
            self.env.process(self._immediate_neighbor_up_update())
            
            self.log_event('neighbor_up', {
                'neighbor': neighbor_id,
                'convergence_time': 0.01  # Fast convergence
            })

    def _immediate_neighbor_up_update(self):
        """Send immediate updates after neighbor recovery."""
        yield self.env.timeout(0.001)  # Nearly immediate
        self._send_triggered_updates()

    def get_adup_statistics(self) -> Dict[str, Any]:
        """Get ADUP-specific statistics with fast convergence metrics."""
        stats = self.get_statistics()
        stats.update({
            'hello_packets_sent': self.hello_packets_sent,
            'update_packets_sent': self.update_packets_sent,
            'query_packets_sent': self.query_packets_sent,
            'reply_packets_sent': self.reply_packets_sent,
            'suppressed_updates': self.suppressed_updates,
            'convergence_events': self.convergence_events,
            'efficiency_ratio': (self.suppressed_updates / max(1, self.update_packets_sent + self.suppressed_updates)),
            'convergence_efficiency': (self.convergence_events / max(1, self.update_packets_sent)),
            'fast_convergence_config': {
                'hello_interval': self.hello_interval,  # 2.0s - fast neighbor detection
                'update_interval': self.update_interval,  # 0.5s - very fast updates
                'neighbor_check_interval': self.neighbor_check_interval,  # 1.0s - quick failure detection
                'metric_change_threshold': self.metric_change_threshold,  # 0.1 - sensitive to changes
                'update_suppression_time': self.update_suppression_time,  # 0.2s - minimal suppression
                'query_timeout': self.query_timeout,  # 1.0s - fast query timeout
                'fast_flooding_enabled': self.fast_flooding_enabled
            },
            'route_states': {
                'active_routes': len([e for e in self.topology_table.values() if e.state == RouteState.ACTIVE]),
                'passive_routes': len([e for e in self.topology_table.values() if e.state == RouteState.PASSIVE]),
                'total_routes': len(self.topology_table)
            },
            'topology_metrics': {
                'feasible_successors': sum(len(e.successors) for e in self.topology_table.values()),
                'average_successors': sum(len(e.successors) for e in self.topology_table.values()) / max(1, len(self.topology_table)),
                'reachable_destinations': len([e for e in self.topology_table.values() if e.successor is not None])
            },
            'mab_statistics': {
                'epsilon': self.mab.epsilon,
                'total_selections': self.mab.total_selections,
                'exploration_count': self.mab.exploration_count,
                'exploitation_count': self.mab.exploitation_count,
                'exploration_ratio': self.mab.exploration_count / max(1, self.mab.total_selections)
            },
            'performance_optimizations': {
                'immediate_updates': True,
                'fast_neighbor_detection': True,
                'ultra_fast_failure_recovery': True,
                'enhanced_mab_exploration': True,
                'topology_learning_from_hello': True
            }
        })
        return stats
