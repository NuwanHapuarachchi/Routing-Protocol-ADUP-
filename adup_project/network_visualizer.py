#!/usr/bin/env python3
"""
Network Topology and Path Selection Visualizer
Shows how RIP, OSPF, and ADUP protocols behave differently in path selection.
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
from typing import Dict, List, Tuple, Any
import os
from adup_project.simulation_manager import SimulationManager
from adup_project.topologies.standard_topologies import get_scenario_topology
import pandas as pd


def setup_visualization():
    """Setup matplotlib for high-quality network visualizations."""
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (15, 10)
    plt.rcParams['font.size'] = 10


def create_random_network_topology(num_nodes: int = 8, edge_probability: float = 0.4) -> nx.Graph:
    """Create a random network topology for testing."""
    G = nx.erdos_renyi_graph(num_nodes, edge_probability, seed=42)
    
    # Ensure connectivity
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        for i in range(len(components) - 1):
            node1 = random.choice(list(components[i]))
            node2 = random.choice(list(components[i + 1]))
            G.add_edge(node1, node2)
    
    # Add random weights to edges
    for u, v in G.edges():
        G[u][v]['weight'] = random.randint(1, 10)
        G[u][v]['delay'] = random.uniform(5, 50)
        G[u][v]['jitter'] = random.uniform(0.5, 5.0)
        G[u][v]['packet_loss'] = random.uniform(0, 2.0)
        G[u][v]['congestion'] = random.uniform(0, 20.0)
    
    # Label nodes with letters
    mapping = {i: chr(65 + i) for i in range(num_nodes)}
    G = nx.relabel_nodes(G, mapping)
    
    return G


def simulate_protocol_routing(protocol: str, topology: nx.Graph, source: str, destination: str) -> Dict[str, Any]:
    """Simulate routing behavior for a specific protocol."""
    
    # Create simulation
    sim = SimulationManager(protocol, topology.copy(), simulation_time=30.0)
    events_df = sim.run_simulation()
    router_stats = sim.get_router_statistics()
    
    # Get the path each protocol would choose
    path = []
    protocol_cost = float('inf')
    
    if protocol == "RIP":
        # RIP uses hop count (shortest path in terms of hops)
        try:
            path = nx.shortest_path(topology, source, destination)
            protocol_cost = len(path) - 1  # Hop count
        except:
            path, protocol_cost = [], float('inf')
    
    elif protocol == "OSPF":
        # OSPF uses weighted shortest path
        try:
            path = nx.shortest_path(topology, source, destination, weight='weight')
            protocol_cost = nx.shortest_path_length(topology, source, destination, weight='weight')
        except:
            path, protocol_cost = [], float('inf')
    
    elif protocol == "ADUP":
        # ADUP uses optimized composite metrics
        try:
            # Calculate optimized composite weights using ADUP's method
            for u, v in topology.edges():
                edge_data = topology[u][v]
                
                # Use ADUP's optimized metric calculation
                delay_norm = min(edge_data.get('delay', 10) / 100.0, 1.0)
                jitter_norm = min(edge_data.get('jitter', 1) / 50.0, 1.0)
                loss_norm = min(edge_data.get('packet_loss', 0) / 10.0, 1.0)
                congestion_norm = min(edge_data.get('congestion', 0) / 100.0, 1.0)
                
                # ADUP's optimized weights
                composite = (
                    0.4 * delay_norm +      # Prioritize delay
                    0.15 * jitter_norm +    # Reduce jitter weight
                    0.35 * loss_norm +      # High weight for packet loss
                    0.1 * congestion_norm   # Lower congestion weight
                )
                
                # Scale to reasonable routing metric range (1-50)
                adup_metric = max(1.0, composite * 50.0)
                topology[u][v]['adup_metric'] = adup_metric
            
            path = nx.shortest_path(topology, source, destination, weight='adup_metric')
            protocol_cost = nx.shortest_path_length(topology, source, destination, weight='adup_metric')
        except:
            path, protocol_cost = [], float('inf')
    
    # Calculate actual network performance metrics for the chosen path
    if len(path) > 1:
        total_delay = 0
        total_jitter = 0
        total_packet_loss = 0
        total_congestion = 0
        total_weight = 0
        hop_count = len(path) - 1
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge_data = topology[u][v]
            
            total_delay += edge_data.get('delay', 10)
            total_jitter += edge_data.get('jitter', 1)
            total_packet_loss += edge_data.get('packet_loss', 0)
            total_congestion += edge_data.get('congestion', 0)
            total_weight += edge_data.get('weight', 1)
        
        # Calculate a normalized quality score (0-100, lower is better)
        # Use reasonable normalization ranges based on typical network values
        
        # Average per-hop values
        avg_delay_per_hop = total_delay / hop_count if hop_count > 0 else total_delay
        avg_jitter_per_hop = total_jitter / hop_count if hop_count > 0 else total_jitter
        avg_loss_per_hop = total_packet_loss / hop_count if hop_count > 0 else total_packet_loss
        avg_congestion_per_hop = total_congestion / hop_count if hop_count > 0 else total_congestion
        
        # Normalize to 0-1 scale using reasonable maximum values
        delay_norm = min(avg_delay_per_hop / 50.0, 1.0)      # Max 50ms per hop = bad
        jitter_norm = min(avg_jitter_per_hop / 10.0, 1.0)    # Max 10ms jitter per hop = bad  
        loss_norm = min(avg_loss_per_hop / 5.0, 1.0)         # Max 5% loss per hop = bad
        congestion_norm = min(avg_congestion_per_hop / 50.0, 1.0)  # Max 50 congestion per hop = bad
        
        # Combined quality score (0-100, lower is better)
        # Weight: delay=40%, loss=30%, jitter=20%, congestion=10%
        quality_score = (delay_norm * 40 + loss_norm * 30 + jitter_norm * 20 + congestion_norm * 10)
        
    else:
        total_delay = float('inf')
        total_jitter = float('inf')
        total_packet_loss = float('inf')
        total_congestion = float('inf')
        total_weight = float('inf')
        hop_count = float('inf')
        quality_score = 100.0  # Worst possible score
    
    return {
        'protocol': protocol,
        'path': path,
        'protocol_cost': protocol_cost,  # Original protocol cost
        'hop_count': hop_count,
        'total_delay': total_delay,
        'total_jitter': total_jitter,
        'total_packet_loss': total_packet_loss,
        'total_congestion': total_congestion,
        'total_weight': total_weight,
        'quality_score': quality_score,  # Normalized comparison metric
        'packets_sent': sum(stats['packets_sent'] for stats in router_stats.values()),
        'convergence_time': sum(stats['convergence_time'] for stats in router_stats.values()) / len(router_stats)
    }


def visualize_network_comparison(topology: nx.Graph, source: str, destination: str, output_dir: str = "plots"):
    """Create a comprehensive visualization comparing all three protocols."""
    setup_visualization()
    
    # Simulate routing for all protocols
    protocols = ["RIP", "OSPF", "ADUP"]
    results = {}
    
    print(f"Simulating routing from {source} to {destination}...")
    for protocol in protocols:
        results[protocol] = simulate_protocol_routing(protocol, topology, source, destination)
        print(f"{protocol}: Path = {' -> '.join(results[protocol]['path'])}")
    
    # Print comparison table
    print("\nPERFORMANCE COMPARISON:")
    print("-" * 80)
    print(f"{'Protocol':<8} {'Hops':<5} {'Delay':<8} {'Loss%':<7} {'Jitter':<8} {'Quality':<8} {'Path'}")
    print("-" * 80)
    
    for protocol in protocols:
        r = results[protocol]
        if r['path']:
            path_str = ' → '.join(r['path'])
            print(f"{protocol:<8} {r['hop_count']:<5} {r['total_delay']:<8.1f} {r['total_packet_loss']:<7.2f} "
                  f"{r['total_jitter']:<8.1f} {r['quality_score']:<8.1f} {path_str}")
        else:
            print(f"{protocol:<8} {'∞':<5} {'∞':<8} {'∞':<7} {'∞':<8} {'100.0':<8} No path")
    print("-" * 80)
    
    # Create clean, professional visualization
    fig = plt.figure(figsize=(16, 10))
    plt.style.use('default')
    
    # Define colors for different protocols
    colors = {'RIP': '#E74C3C', 'OSPF': '#3498DB', 'ADUP': '#2ECC71'}
    
    # Create consistent node positioning
    pos = nx.spring_layout(topology, seed=42, k=2, iterations=100)
    
    # Main title
    fig.suptitle(f'Protocol Comparison: {source} → {destination}', 
                fontsize=18, fontweight='bold', y=0.95)
    
    # Create 2x2 grid layout
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.2)
    
    # Plot 1: Network topology (top-left)
    ax_topo = fig.add_subplot(gs[0, 0])
    nx.draw_networkx_nodes(topology, pos, ax=ax_topo, node_color='lightsteelblue', 
                          node_size=800, alpha=0.9, edgecolors='navy', linewidths=2)
    nx.draw_networkx_edges(topology, pos, ax=ax_topo, edge_color='gray', alpha=0.6, width=1.5)
    nx.draw_networkx_labels(topology, pos, ax=ax_topo, font_size=11, font_weight='bold')
    
    # Highlight source and destination
    nx.draw_networkx_nodes(topology, pos, nodelist=[source], node_color='#27AE60', 
                          node_size=1000, alpha=1.0, ax=ax_topo, edgecolors='darkgreen', linewidths=3)
    nx.draw_networkx_nodes(topology, pos, nodelist=[destination], node_color='#E67E22', 
                          node_size=1000, alpha=1.0, ax=ax_topo, edgecolors='darkorange', linewidths=3)
    
    ax_topo.set_title('Network Topology', fontweight='bold', fontsize=12, pad=20)
    ax_topo.axis('off')
    
    # Plot each protocol's path (remaining positions)
    positions = [(0, 1), (1, 0), (1, 1)]
    
    for i, protocol in enumerate(protocols):
        row, col = positions[i]
        ax = fig.add_subplot(gs[row, col])
        
        # Draw base network
        nx.draw_networkx_nodes(topology, pos, ax=ax, node_color='lightgray', 
                              node_size=600, alpha=0.4)
        nx.draw_networkx_edges(topology, pos, ax=ax, edge_color='lightgray', alpha=0.3, width=1)
        nx.draw_networkx_labels(topology, pos, ax=ax, font_size=9, alpha=0.7)
        
        # Highlight the chosen path
        path = results[protocol]['path']
        if len(path) > 1:
            path_edges = [(path[j], path[j+1]) for j in range(len(path)-1)]
            nx.draw_networkx_edges(topology, pos, edgelist=path_edges, 
                                 edge_color=colors[protocol], width=4, alpha=0.9, ax=ax)
            
            # Highlight nodes in path
            nx.draw_networkx_nodes(topology, pos, nodelist=path, 
                                 node_color=colors[protocol], node_size=700, alpha=0.9, ax=ax,
                                 edgecolors='white', linewidths=2)
        
        # Always highlight source and destination
        nx.draw_networkx_nodes(topology, pos, nodelist=[source], node_color='#27AE60', 
                              node_size=800, alpha=1.0, ax=ax, edgecolors='darkgreen', linewidths=2)
        nx.draw_networkx_nodes(topology, pos, nodelist=[destination], node_color='#E67E22', 
                              node_size=800, alpha=1.0, ax=ax, edgecolors='darkorange', linewidths=2)
        
        r = results[protocol]
        path_str = ' → '.join(path) if path else 'No path'
        
        # Create info text
        info_text = f"Path: {path_str}\n"
        info_text += f"Hops: {r['hop_count']}\n"
        info_text += f"Quality: {r['quality_score']:.1f}/100"
        
        ax.set_title(f'{protocol}', fontweight='bold', fontsize=12, 
                    color=colors[protocol], pad=10)
        
        # Add info box
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
               facecolor=colors[protocol], alpha=0.1, edgecolor=colors[protocol]))
        
        ax.axis('off')
    
    # Add performance summary at the bottom
    summary_text = "PERFORMANCE SUMMARY (Lower Quality Score = Better):\n"
    
    # Sort by quality score for ranking
    sorted_results = sorted([(p, r) for p, r in results.items() if r['path']], 
                           key=lambda x: x[1]['quality_score'])
    
    for i, (protocol, data) in enumerate(sorted_results):
        rank_emoji = ["🥇", "🥈", "🥉"][i] if i < 3 else f"{i+1}."
        summary_text += f"{rank_emoji} {protocol}: Quality {data['quality_score']:.1f} | "
        summary_text += f"Delay {data['total_delay']:.1f}ms | Loss {data['total_packet_loss']:.2f}%\n"
    
    fig.text(0.5, 0.02, summary_text, ha='center', va='bottom', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for summary
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/network_path_comparison_{source}_{destination}.png", 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return results


def save_comparison_data(results: Dict[str, Dict], source: str, destination: str, 
                        topology_name: str = "Random", output_dir: str = "plots"):
    """Save detailed comparison data to CSV file."""
    # Prepare data for CSV
    data_rows = []
    for protocol, data in results.items():
        if data['path']:
            row = {
                'Topology': topology_name,
                'Source': source,
                'Destination': destination,
                'Protocol': protocol,
                'Path': ' → '.join(data['path']),
                'Hop_Count': data['hop_count'],
                'Total_Delay_ms': round(data['total_delay'], 2),
                'Total_Packet_Loss_percent': round(data['total_packet_loss'], 3),
                'Total_Jitter_ms': round(data['total_jitter'], 2),
                'Total_Congestion': round(data['total_congestion'], 2),
                'Total_Weight': round(data['total_weight'], 2),
                'Quality_Score': round(data['quality_score'], 2),
                'Protocol_Cost': round(data['protocol_cost'], 2),
                'Packets_Sent': data['packets_sent'],
                'Convergence_Time': round(data['convergence_time'], 3)
            }
        else:
            row = {
                'Topology': topology_name,
                'Source': source,
                'Destination': destination,
                'Protocol': protocol,
                'Path': 'No path',
                'Hop_Count': float('inf'),
                'Total_Delay_ms': float('inf'),
                'Total_Packet_Loss_percent': float('inf'),
                'Total_Jitter_ms': float('inf'),
                'Total_Congestion': float('inf'),
                'Total_Weight': float('inf'),
                'Quality_Score': 100.0,
                'Protocol_Cost': float('inf'),
                'Packets_Sent': data['packets_sent'],
                'Convergence_Time': round(data['convergence_time'], 3)
            }
        data_rows.append(row)
    
    # Create DataFrame and save
    df = pd.DataFrame(data_rows)
    os.makedirs(output_dir, exist_ok=True)
    
    csv_file = f"{output_dir}/protocol_comparison_results.csv"
    
    # Check if file exists to append or create new
    if os.path.exists(csv_file):
        # Append to existing file
        existing_df = pd.read_csv(csv_file)
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        combined_df.to_csv(csv_file, index=False)
    else:
        # Create new file
        df.to_csv(csv_file, index=False)
    
    print(f"   💾 Comparison data saved to {csv_file}")
    return df


def create_multiple_network_scenarios(output_dir: str = "plots"):
    """Create multiple random network scenarios showing protocol behavior."""
    setup_visualization()
    
    scenarios = [
        {"nodes": 6, "prob": 0.4, "name": "Sparse Network"},
        {"nodes": 8, "prob": 0.6, "name": "Dense Network"},
        {"nodes": 10, "prob": 0.3, "name": "Large Sparse Network"}
    ]
    
    # Create a clean figure with more vertical space
    fig = plt.figure(figsize=(18, 12))
    plt.style.use('default')
    
    # Create grid with minimal spacing
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.2, 
                         left=0.05, right=0.95, top=0.95, bottom=0.05)
    
    colors = {'RIP': '#E74C3C', 'OSPF': '#3498DB', 'ADUP': '#2ECC71'}
    protocols = ["RIP", "OSPF", "ADUP"]
    
    for scenario_idx, scenario in enumerate(scenarios):
        # Create random topology
        topology = create_random_network_topology(scenario["nodes"], scenario["prob"])
        nodes = list(topology.nodes())
        source, destination = nodes[0], nodes[-1]
        
        pos = nx.spring_layout(topology, seed=42 + scenario_idx, k=2, iterations=100)
        
        # Simulate all protocols
        protocol_results = {}
        for protocol in protocols:
            protocol_results[protocol] = simulate_protocol_routing(protocol, topology, source, destination)
        
        # Plot topology (first column) - NO TITLE
        ax = fig.add_subplot(gs[scenario_idx, 0])
        nx.draw_networkx_nodes(topology, pos, ax=ax, node_color='lightsteelblue', 
                              node_size=450, alpha=0.8, edgecolors='navy', linewidths=1.5)
        nx.draw_networkx_edges(topology, pos, ax=ax, edge_color='gray', alpha=0.6, width=1.5)
        nx.draw_networkx_labels(topology, pos, ax=ax, font_size=9, font_weight='bold')
        
        # Highlight source and destination
        nx.draw_networkx_nodes(topology, pos, nodelist=[source], node_color='#27AE60', 
                              node_size=550, alpha=1.0, ax=ax, edgecolors='darkgreen', linewidths=2)
        nx.draw_networkx_nodes(topology, pos, nodelist=[destination], node_color='#E67E22', 
                              node_size=550, alpha=1.0, ax=ax, edgecolors='darkorange', linewidths=2)
        
        ax.axis('off')
        
        # Plot each protocol (columns 2-4) - NO TITLES
        for protocol_idx, protocol in enumerate(protocols):
            ax = fig.add_subplot(gs[scenario_idx, protocol_idx + 1])
            
            # Draw base network (faded)
            nx.draw_networkx_nodes(topology, pos, ax=ax, node_color='lightgray', 
                                  node_size=350, alpha=0.4)
            nx.draw_networkx_edges(topology, pos, ax=ax, edge_color='lightgray', alpha=0.3, width=1)
            nx.draw_networkx_labels(topology, pos, ax=ax, font_size=8, alpha=0.6)
            
            # Highlight chosen path
            path = protocol_results[protocol]['path']
            if len(path) > 1:
                path_edges = [(path[j], path[j+1]) for j in range(len(path)-1)]
                nx.draw_networkx_edges(topology, pos, edgelist=path_edges, 
                                     edge_color=colors[protocol], width=3, alpha=0.9, ax=ax)
                nx.draw_networkx_nodes(topology, pos, nodelist=path, 
                                     node_color=colors[protocol], node_size=450, alpha=0.9, ax=ax,
                                     edgecolors='white', linewidths=1.5)
            
            # Always highlight source and destination
            nx.draw_networkx_nodes(topology, pos, nodelist=[source], node_color='#27AE60', 
                                  node_size=550, alpha=1.0, ax=ax, edgecolors='darkgreen', linewidths=2)
            nx.draw_networkx_nodes(topology, pos, nodelist=[destination], node_color='#E67E22', 
                                  node_size=550, alpha=1.0, ax=ax, edgecolors='darkorange', linewidths=2)
            
            # Add small quality score box in corner
            quality_score = protocol_results[protocol]['quality_score']
            hops = len(path) - 1 if path else 0
            
            # Create very compact info text
            info_text = f"{protocol}\nQ:{quality_score:.1f}\nH:{hops}"
            
            # Add smaller text box in top-left corner
            ax.text(0.01, 0.99, info_text, transform=ax.transAxes, fontsize=7,
                   verticalalignment='top', horizontalalignment='left',
                   bbox=dict(boxstyle="round,pad=0.15", facecolor=colors[protocol], 
                            alpha=0.8, edgecolor='white'),
                   color='white', fontweight='bold')
            
            ax.axis('off')
    
    # Save the plot - NO LEGEND BOX
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/multiple_network_scenarios.png", dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Generate comprehensive network visualizations."""
    print("🌐 Generating Network Path Selection Visualizations")
    print("=" * 60)
    
    # Create a medium-sized random network
    print("\n1. Creating random network topology...")
    topology = create_random_network_topology(8, 0.4)
    print(f"   Created network with {topology.number_of_nodes()} nodes and {topology.number_of_edges()} edges")
    
    # Test multiple source-destination pairs
    nodes = list(topology.nodes())
    test_pairs = [
        (nodes[0], nodes[-1]),  # First to last
        (nodes[2], nodes[-2]),  # Middle pairs
    ]
    
    print("\n2. Generating path comparison visualizations...")
    for source, dest in test_pairs:
        print(f"   Analyzing routes: {source} → {dest}")
        results = visualize_network_comparison(topology, source, dest)
        
        # Save comparison data to CSV
        save_comparison_data(results, source, dest, "Random_8_Node", "plots")
        
        # Print summary with comprehensive metrics
        print("   PERFORMANCE SUMMARY:")
        best_quality = min(r['quality_score'] for r in results.values() if r['path'])
        best_hops = min(r['hop_count'] for r in results.values() if r['path'])
        best_delay = min(r['total_delay'] for r in results.values() if r['path'])
        
        for protocol, data in results.items():
            if data['path']:
                path_str = ' → '.join(data['path'])
                quality_indicator = "🥇" if data['quality_score'] == best_quality else "🥈" if data['quality_score'] <= best_quality + 10 else "🥉"
                print(f"     {quality_indicator} {protocol}: {path_str}")
                print(f"        Quality: {data['quality_score']:.1f}/100 | Hops: {data['hop_count']} | Delay: {data['total_delay']:.1f}ms | Loss: {data['total_packet_loss']:.2f}%")
            else:
                print(f"     ❌ {protocol}: No path found")
        print()
    
    print("\n3. Creating multiple network scenarios...")
    create_multiple_network_scenarios()
    
    print("\n4. Testing with standard topologies...")
    
    # Test with diamond topology
    diamond_topology = get_scenario_topology("PATH_INTELLIGENCE")
    print("   Testing diamond topology...")
    diamond_results = visualize_network_comparison(diamond_topology, 'A', 'D')
    save_comparison_data(diamond_results, 'A', 'D', "Diamond_Topology", "plots")
    
    print(f"\nAll visualizations saved to 'plots/' directory")
    print("\nGenerated files:")
    plot_files = [f for f in os.listdir('plots') if f.endswith(('.png', '.csv'))]
    for f in sorted(plot_files):
        file_type = "[PLOT]" if f.endswith('.png') else "[DATA]"
        print(f"   {file_type} {f}")


if __name__ == "__main__":
    main() 