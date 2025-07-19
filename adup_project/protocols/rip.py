"""
RIPv2 (Routing Information Protocol version 2) implementation.
Used for comparison with ADUP protocol performance.
"""

import simpy
import random
from typing import Dict, List, Any

from .base_router import BaseRouter, RoutingTableEntry
from adup_project.utils.packets import RipPacket, RipEntry, RipCommand


class RipRouter(BaseRouter):
    """RIPv2 Router implementation for comparison testing."""
    
    def __init__(self, env, router_id, network_graph, metric_weights=None):
        super().__init__(env, router_id, network_graph, metric_weights)
        
        # RIP specific parameters
        self.update_interval = 30.0  # Periodic update interval
        self.max_hop_count = 15      # Maximum hop count
        
        # RIP routing table
        self.rip_routes: Dict[str, Dict[str, Any]] = {}
        
        # Statistics
        self.update_packets_sent = 0
        self.triggered_updates = 0
        
        # Initialize direct routes
        self._initialize_direct_routes()
    
    def _initialize_direct_routes(self):
        """Initialize directly connected routes."""
        for neighbor in self.neighbors:
            self.rip_routes[neighbor] = {
                'destination': neighbor,
                'next_hop': neighbor,
                'metric': 1,
                'interface': self.neighbor_interfaces[neighbor],
                'valid': True
            }
            
            self.routing_table[neighbor] = RoutingTableEntry(
                destination=neighbor,
                next_hop=neighbor,
                metric=1,
                interface=self.neighbor_interfaces[neighbor],
                source="RIP"
            )
    
    def run(self):
        """Main RIP protocol process."""
        self.env.process(self._periodic_update_process())
        
        yield self.env.timeout(random.uniform(1.0, 5.0))
        self._send_full_update()
        
        while True:
            yield self.env.timeout(1.0)
    
    def _periodic_update_process(self):
        """Send periodic routing updates."""
        while True:
            yield self.env.timeout(self.update_interval)
            self._send_full_update()
    
    def _send_full_update(self):
        """Send full routing table to all neighbors."""
        for neighbor in self.neighbors:
            self._send_update_to_neighbor(neighbor)
    
    def _send_update_to_neighbor(self, neighbor: str):
        """Send routing update to specific neighbor."""
        entries = []
        
        for dest, route_info in self.rip_routes.items():
            if dest == neighbor:
                continue
            
            metric = route_info['metric']
            if route_info['next_hop'] == neighbor:
                metric = 16  # Poison reverse
            
            entry = RipEntry(
                ip_address=hash(dest) & 0xFFFFFFFF,
                metric=min(16, metric),
                subnet_mask=0xFFFFFF00,
                next_hop=0
            )
            entries.append(entry)
        
        if entries:
            rip_packet = RipPacket(command=RipCommand.RESPONSE, entries=entries)
            packet_data = rip_packet.pack()
            self.update_packets_sent += 1
            self.packets_sent += 1  # Update base router counter
            
            self.log_event('rip_update_sent', {
                'neighbor': neighbor,
                'route_count': len(entries)
            })
    
    def _handle_received_packet(self, packet_data: bytes, source: str, packet_type: str):
        """Handle received RIP packet."""
        if packet_type == "RIP_RESPONSE":
            try:
                rip_packet = RipPacket.unpack(packet_data)
                self._process_rip_update(rip_packet, source)
            except Exception as e:
                self.packets_dropped += 1
    
    def _process_rip_update(self, rip_packet: RipPacket, source: str):
        """Process received RIP update."""
        routes_changed = []
        
        for entry in rip_packet.entries:
            destination = str(entry.ip_address)
            advertised_metric = entry.metric
            total_metric = min(16, advertised_metric + 1)
            
            if destination not in self.rip_routes and total_metric < 16:
                # New route
                self.rip_routes[destination] = {
                    'destination': destination,
                    'next_hop': source,
                    'metric': total_metric,
                    'interface': self.neighbor_interfaces.get(source, f"eth_{source}"),
                    'valid': True
                }
                
                self.routing_table[destination] = RoutingTableEntry(
                    destination=destination,
                    next_hop=source,
                    metric=total_metric,
                    interface=self.neighbor_interfaces.get(source, f"eth_{source}"),
                    source="RIP"
                )
                
                routes_changed.append(destination)
            
            elif destination in self.rip_routes:
                current_route = self.rip_routes[destination]
                
                if source == current_route['next_hop'] or total_metric < current_route['metric']:
                    # Update existing route
                    current_route['next_hop'] = source
                    current_route['metric'] = total_metric
                    current_route['valid'] = total_metric < 16
                    
                    if total_metric < 16:
                        self.routing_table[destination].next_hop = source
                        self.routing_table[destination].metric = total_metric
                    else:
                        self.routing_table.pop(destination, None)
                    
                    routes_changed.append(destination)
        
        if routes_changed:
            self.env.process(self._delayed_triggered_update(routes_changed))
    
    def _delayed_triggered_update(self, changed_routes: List[str]):
        """Send triggered update with delay."""
        yield self.env.timeout(random.uniform(1.0, 5.0))
        
        for route in changed_routes:
            self.triggered_updates += 1
            self._send_full_update()
            break  # Send only one update for all changes
    
    def handle_neighbor_down(self, neighbor_id: str):
        """Handle neighbor failure."""
        if neighbor_id in self.neighbors:
            self.neighbors.remove(neighbor_id)
            
            routes_affected = []
            for dest, route_info in self.rip_routes.items():
                if route_info['next_hop'] == neighbor_id:
                    route_info['metric'] = 16
                    route_info['valid'] = False
                    routes_affected.append(dest)
                    self.routing_table.pop(dest, None)
            
            self.log_event('neighbor_down', {
                'neighbor': neighbor_id,
                'routes_affected': len(routes_affected)
            })
    
    def handle_neighbor_up(self, neighbor_id: str):
        """Handle neighbor recovery."""
        if neighbor_id not in self.neighbors:
            self.neighbors.add(neighbor_id)
            
            self.rip_routes[neighbor_id] = {
                'destination': neighbor_id,
                'next_hop': neighbor_id,
                'metric': 1,
                'interface': self.neighbor_interfaces.get(neighbor_id, f"eth_{neighbor_id}"),
                'valid': True
            }
            
            self.routing_table[neighbor_id] = RoutingTableEntry(
                destination=neighbor_id,
                next_hop=neighbor_id,
                metric=1,
                interface=self.neighbor_interfaces.get(neighbor_id, f"eth_{neighbor_id}"),
                source="RIP"
            )
            
            self._send_update_to_neighbor(neighbor_id)
            
            self.log_event('neighbor_up', {'neighbor': neighbor_id})