"""
Standard network topologies for ADUP simulation testing.
"""

import networkx as nx
import random
from typing import Dict, Any


def create_linear_topology(num_nodes: int = 5) -> nx.Graph:
    """Create a linear chain topology for convergence speed testing."""
    G = nx.Graph()
    
    # Add nodes
    for i in range(num_nodes):
        G.add_node(f"R{i}")
    
    # Add edges to form a chain
    for i in range(num_nodes - 1):
        G.add_edge(f"R{i}", f"R{i+1}", 
                  delay=10.0 + random.uniform(0, 5),
                  jitter=random.uniform(0.5, 2.0),
                  packet_loss=random.uniform(0, 0.5),
                  congestion=random.uniform(0, 10),
                  bandwidth=100.0)
    
    return G


def create_diamond_topology() -> nx.Graph:
    """Create a diamond topology for path intelligence testing."""
    G = nx.Graph()
    
    # Add nodes
    nodes = ["A", "B", "C", "D"]
    for node in nodes:
        G.add_node(node)
    
    # Path A-B-D: Higher delay but lower congestion
    G.add_edge("A", "B", delay=25, jitter=1.0, packet_loss=0.1, congestion=5.0, bandwidth=100.0)
    G.add_edge("B", "D", delay=25, jitter=1.0, packet_loss=0.1, congestion=5.0, bandwidth=100.0)
    
    # Path A-C-D: Lower delay but higher congestion
    G.add_edge("A", "C", delay=15, jitter=2.0, packet_loss=0.5, congestion=25.0, bandwidth=100.0)
    G.add_edge("C", "D", delay=15, jitter=2.0, packet_loss=0.5, congestion=25.0, bandwidth=100.0)
    
    return G


def create_mesh_topology(num_nodes: int = 6) -> nx.Graph:
    """Create a partial mesh topology for stability testing."""
    G = nx.Graph()
    
    # Add nodes
    for i in range(num_nodes):
        G.add_node(f"R{i}")
    
    # Create partial mesh
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < 0.6:  # 60% connectivity
                G.add_edge(f"R{i}", f"R{j}",
                          delay=10.0 + random.uniform(0, 10),
                          jitter=random.uniform(0.5, 3.0),
                          packet_loss=random.uniform(0, 1.0),
                          congestion=random.uniform(0, 30),
                          bandwidth=random.uniform(50, 150))
    
    # Ensure connectivity
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        for i in range(len(components) - 1):
            node1 = list(components[i])[0]
            node2 = list(components[i + 1])[0]
            G.add_edge(node1, node2, delay=10.0, jitter=1.0, packet_loss=0.1, 
                      congestion=5.0, bandwidth=100.0)
    
    return G


def create_scalability_topology(num_nodes: int) -> nx.Graph:
    """Create large-scale topology for scalability testing."""
    # Use Barabási-Albert model for scale-free network
    m = max(2, min(5, num_nodes // 10))
    G = nx.barabasi_albert_graph(num_nodes, m)
    
    # Ensure connectivity
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        for i in range(len(components) - 1):
            node1 = list(components[i])[0]
            node2 = list(components[i + 1])[0]
            G.add_edge(node1, node2)
    
    # Rename nodes and add attributes
    mapping = {i: f"R{i}" for i in range(num_nodes)}
    G = nx.relabel_nodes(G, mapping)
    
    # Add link attributes
    for u, v in G.edges():
        G[u][v]['delay'] = random.uniform(8.0, 15.0)
        G[u][v]['jitter'] = random.uniform(0.5, 2.5)
        G[u][v]['packet_loss'] = random.uniform(0, 0.8)
        G[u][v]['congestion'] = random.uniform(0, 20)
        G[u][v]['bandwidth'] = random.uniform(80, 120)
    
    return G


def get_scenario_topology(scenario: str, **kwargs) -> nx.Graph:
    """Get topology for specific test scenario."""
    scenario = scenario.upper()
    
    if scenario == "A" or scenario == "CONVERGENCE":
        return create_linear_topology(**kwargs)
    elif scenario == "B" or scenario == "PATH_INTELLIGENCE":
        return create_diamond_topology()
    elif scenario == "C" or scenario == "STABILITY":
        return create_mesh_topology(**kwargs)
    elif scenario == "D" or scenario == "SCALABILITY":
        return create_scalability_topology(**kwargs)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")


def analyze_topology(graph: nx.Graph) -> Dict[str, Any]:
    """Analyze topology characteristics."""
    return {
        'num_nodes': graph.number_of_nodes(),
        'num_edges': graph.number_of_edges(),
        'density': nx.density(graph),
        'is_connected': nx.is_connected(graph),
        'diameter': nx.diameter(graph) if nx.is_connected(graph) else float('inf')
    }