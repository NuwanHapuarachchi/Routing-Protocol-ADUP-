"""
Simplified OSPF (Open Shortest Path First) implementation.
Used for comparison with ADUP protocol performance.
"""

import simpy
import random
import math
from typing import Dict, List, Set, Any, Tuple

from .base_router import BaseRouter, RoutingTableEntry
from adup_project.utils.packets import OspfHelloPacket, OspfHeader, OspfType


class OspfRouter(BaseRouter):
    """Simplified OSPF Router implementation for comparison testing."""
    
    def __init__(self, env, router_id, network_graph, metric_weights=None):
        super().__init__(env, router_id, network_graph, metric_weights)
        
        # OSPF specific parameters
        self.hello_interval = 10.0
        self.dead_interval = 40.0
        self.area_id = 0  # Backbone area
        
        # OSPF state
        self.neighbor_states: Dict[str, str] = {}  # neighbor_id -> state
        self.neighbor_last_hello: Dict[str, float] = {}
        self.link_state_database: Dict[str, Dict[str, Any]] = {}
        
        # Statistics
        self.hello_packets_sent = 0
        self.lsa_count = 0
        self.spf_calculations = 0
        
        # Initialize LSA for self
        self._initialize_self_lsa()
    
    def _initialize_self_lsa(self):
        """Initialize Link State Advertisement for this router."""
        links = []
        for neighbor in self.neighbors:
            if neighbor in self.neighbor_metrics:
                current_metrics = self.neighbor_metrics[neighbor].get_current_metrics()
                cost = int(self.calculate_composite_metric(current_metrics))
                links.append({
                    'neighbor': neighbor,
                    'cost': cost,
                    'interface': self.neighbor_interfaces[neighbor]
                })
        
        self.link_state_database[self.router_id] = {
            'router_id': self.router_id,
            'sequence': 1,
            'age': 0,
            'links': links,
            'timestamp': self.env.now
        }
    
    def run(self):
        """Main OSPF protocol process."""
        # Start periodic processes
        self.env.process(self._hello_process())
        self.env.process(self._neighbor_monitoring())
        self.env.process(self._lsa_aging_process())
        
        # Initial startup delay
        yield self.env.timeout(random.uniform(1.0, 3.0))
        
        # Send initial hellos
        for neighbor in self.neighbors:
            self._send_hello(neighbor)
        
        # Initial SPF calculation
        yield self.env.timeout(5.0)
        self._calculate_spf()
        
        while True:
            yield self.env.timeout(1.0)
    
    def _hello_process(self):
        """Periodic hello packet transmission."""
        while True:
            yield self.env.timeout(self.hello_interval + random.uniform(-1.0, 1.0))
            
            for neighbor in self.neighbors:
                self._send_hello(neighbor)
    
    def _neighbor_monitoring(self):
        """Monitor neighbor health."""
        while True:
            yield self.env.timeout(5.0)  # Check every 5 seconds
            
            current_time = self.env.now
            dead_neighbors = []
            
            for neighbor, last_hello in self.neighbor_last_hello.items():
                if current_time - last_hello > self.dead_interval:
                    dead_neighbors.append(neighbor)
            
            for neighbor in dead_neighbors:
                self._handle_neighbor_death(neighbor)
    
    def _lsa_aging_process(self):
        """Age LSAs and trigger SPF when needed."""
        while True:
            yield self.env.timeout(30.0)  # Age LSAs every 30 seconds
            
            current_time = self.env.now
            spf_needed = False
            
            for router_id, lsa in self.link_state_database.items():
                lsa['age'] = current_time - lsa['timestamp']
                
                # Remove very old LSAs
                if lsa['age'] > 3600:  # 1 hour max age
                    del self.link_state_database[router_id]
                    spf_needed = True
            
            if spf_needed:
                self._calculate_spf()
    
    def _send_hello(self, neighbor: str):
        """Send hello packet to neighbor."""
        hello_packet = OspfHelloPacket(
            network_mask=0xFFFFFF00,  # /24 network
            hello_interval=int(self.hello_interval),
            router_dead_interval=int(self.dead_interval),
            neighbors=list(self.neighbor_states.keys())
        )
        
        packet_data = hello_packet.pack()
        self.hello_packets_sent += 1
        self.packets_sent += 1  # Update base router counter
        
        self.log_event('ospf_hello_sent', {
            'neighbor': neighbor,
            'neighbor_count': len(self.neighbor_states)
        })
    
    def _handle_received_packet(self, packet_data: bytes, source: str, packet_type: str):
        """Handle received OSPF packet."""
        if packet_type == "OSPF_HELLO":
            try:
                hello_packet = OspfHelloPacket.unpack(packet_data)
                self._process_hello(hello_packet, source)
            except Exception as e:
                self.packets_dropped += 1
                self.log_event('ospf_packet_error', {
                    'source': source,
                    'error': str(e)
                })
    
    def _process_hello(self, hello_packet: OspfHelloPacket, source: str):
        """Process received hello packet."""
        self.neighbor_last_hello[source] = self.env.now
        
        # Update neighbor state
        if source not in self.neighbor_states:
            self.neighbor_states[source] = "Init"
            self.log_event('neighbor_discovered', {'neighbor': source})
        
        # Check if we are in neighbor's hello
        if int(self.router_id, 16) & 0xFFFFFFFF in hello_packet.neighbors:
            if self.neighbor_states[source] == "Init":
                self.neighbor_states[source] = "2-Way"
                self.log_event('neighbor_2way', {'neighbor': source})
                
                # Exchange link state information
                self._exchange_lsa_with_neighbor(source)
        
        self.log_event('ospf_hello_received', {
            'source': source,
            'state': self.neighbor_states.get(source, "Unknown")
        })
    
    def _exchange_lsa_with_neighbor(self, neighbor: str):
        """Exchange LSA information with neighbor."""
        # In a full OSPF implementation, this would involve:
        # 1. Database Description exchange
        # 2. Link State Request/Update
        # 3. Link State Acknowledgment
        
        # Simplified: just share our LSA
        if neighbor not in self.link_state_database:
            # Create LSA for neighbor based on received hello
            if neighbor in self.neighbor_metrics:
                current_metrics = self.neighbor_metrics[neighbor].get_current_metrics()
                cost = int(self.calculate_composite_metric(current_metrics))
                
                self.link_state_database[neighbor] = {
                    'router_id': neighbor,
                    'sequence': 1,
                    'age': 0,
                    'links': [{
                        'neighbor': self.router_id,
                        'cost': cost,
                        'interface': f"eth_{self.router_id}"
                    }],
                    'timestamp': self.env.now
                }
                
                self.lsa_count += 1
                
                # Trigger SPF calculation
                self.env.process(self._delayed_spf_calculation())
    
    def _delayed_spf_calculation(self):
        """Delayed SPF calculation to avoid calculation storms."""
        yield self.env.timeout(random.uniform(1.0, 5.0))
        self._calculate_spf()
    
    def _calculate_spf(self):
        """Calculate Shortest Path First (Dijkstra's algorithm)."""
        self.spf_calculations += 1
        self.start_convergence_timer()
        
        # Dijkstra's algorithm implementation
        distances = {router: float('inf') for router in self.link_state_database.keys()}
        distances[self.router_id] = 0
        previous = {}
        unvisited = set(self.link_state_database.keys())
        
        while unvisited:
            # Find node with minimum distance
            current = min(unvisited, key=lambda x: distances[x])
            
            if distances[current] == float('inf'):
                break  # No more reachable nodes
            
            unvisited.remove(current)
            
            # Check neighbors of current node
            if current in self.link_state_database:
                lsa = self.link_state_database[current]
                for link in lsa['links']:
                    neighbor = link['neighbor']
                    if neighbor in unvisited:
                        alt_distance = distances[current] + link['cost']
                        if alt_distance < distances[neighbor]:
                            distances[neighbor] = alt_distance
                            previous[neighbor] = current
        
        # Update routing table based on SPF results
        self._update_routing_table_from_spf(distances, previous)
        
        self.stop_convergence_timer()
        
        self.log_event('spf_calculated', {
            'reachable_nodes': len([d for d in distances.values() if d < float('inf')]),
            'convergence_time': self.convergence_time
        })
    
    def _update_routing_table_from_spf(self, distances: Dict[str, float], 
                                     previous: Dict[str, str]):
        """Update routing table based on SPF calculation results."""
        new_routes = 0
        updated_routes = 0
        
        for destination, distance in distances.items():
            if destination == self.router_id or distance == float('inf'):
                continue
            
            # Find next hop by tracing back path
            next_hop = destination
            while previous.get(next_hop) != self.router_id:
                if next_hop not in previous:
                    break
                next_hop = previous[next_hop]
            
            # Only update if next hop is a direct neighbor
            if next_hop in self.neighbors:
                interface = self.neighbor_interfaces.get(next_hop, f"eth_{next_hop}")
                
                if destination not in self.routing_table:
                    new_routes += 1
                else:
                    updated_routes += 1
                
                self.routing_table[destination] = RoutingTableEntry(
                    destination=destination,
                    next_hop=next_hop,
                    metric=distance,
                    interface=interface,
                    source="OSPF"
                )
        
        self.log_event('routing_table_updated', {
            'new_routes': new_routes,
            'updated_routes': updated_routes,
            'total_routes': len(self.routing_table)
        })
    
    def _handle_neighbor_death(self, neighbor: str):
        """Handle detected neighbor death."""
        if neighbor in self.neighbor_states:
            del self.neighbor_states[neighbor]
        
        if neighbor in self.neighbor_last_hello:
            del self.neighbor_last_hello[neighbor]
        
        # Remove LSA for dead neighbor
        if neighbor in self.link_state_database:
            del self.link_state_database[neighbor]
        
        # Recalculate SPF
        self._calculate_spf()
        
        self.log_event('neighbor_death_detected', {
            'neighbor': neighbor
        })
    
    def handle_neighbor_down(self, neighbor_id: str):
        """Handle neighbor failure."""
        if neighbor_id in self.neighbors:
            self.neighbors.remove(neighbor_id)
            self._handle_neighbor_death(neighbor_id)
            
            self.log_event('neighbor_down', {
                'neighbor': neighbor_id
            })
    
    def handle_neighbor_up(self, neighbor_id: str):
        """Handle neighbor recovery."""
        if neighbor_id not in self.neighbors:
            self.neighbors.add(neighbor_id)
            
            # Update self LSA with new neighbor
            self._update_self_lsa()
            
            # Send hello to new neighbor
            self._send_hello(neighbor_id)
            
            self.log_event('neighbor_up', {
                'neighbor': neighbor_id
            })
    
    def _update_self_lsa(self):
        """Update self LSA when topology changes."""
        links = []
        for neighbor in self.neighbors:
            if neighbor in self.neighbor_metrics:
                current_metrics = self.neighbor_metrics[neighbor].get_current_metrics()
                cost = int(self.calculate_composite_metric(current_metrics))
                links.append({
                    'neighbor': neighbor,
                    'cost': cost,
                    'interface': self.neighbor_interfaces[neighbor]
                })
        
        if self.router_id in self.link_state_database:
            lsa = self.link_state_database[self.router_id]
            lsa['sequence'] += 1
            lsa['links'] = links
            lsa['timestamp'] = self.env.now
            lsa['age'] = 0
        
        # Trigger SPF recalculation
        self.env.process(self._delayed_spf_calculation())
    
    def get_ospf_statistics(self) -> Dict[str, Any]:
        """Get OSPF-specific statistics."""
        base_stats = self.get_statistics()
        
        ospf_stats = {
            'hello_packets_sent': self.hello_packets_sent,
            'lsa_count': self.lsa_count,
            'spf_calculations': self.spf_calculations,
            'neighbor_count': len(self.neighbor_states),
            'lsdb_size': len(self.link_state_database)
        }
        
        return {**base_stats, **ospf_stats} 