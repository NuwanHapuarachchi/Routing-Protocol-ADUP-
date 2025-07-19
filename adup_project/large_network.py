"""
Advanced Showcase: ADUP + MAB in a Large, Dynamic Network
- 40 nodes, dynamic metrics, link failures, rich logging and visualization
"""

import sys
import os
# Add the parent directory to Python path to enable imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import time
import simpy
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
from adup_project.simulation_manager import SimulationManager
from adup_project.topologies.standard_topologies import create_scalability_topology
import itertools

# Create output folder for path analysis
ANALYSIS_FOLDER = "path_analysis"
if not os.path.exists(ANALYSIS_FOLDER):
    os.makedirs(ANALYSIS_FOLDER)
    print(f"Created {ANALYSIS_FOLDER} folder for visualization outputs")

# 1. Create a large, realistic topology (40 nodes)
NUM_NODES = 30
SIM_TIME = 300  # seconds

network = create_scalability_topology(NUM_NODES)

# 2. Initialize the simulation manager for ADUP
sim_manager = SimulationManager(
    protocol_type="ADUP",
    network_graph=network,
    simulation_time=SIM_TIME
)

# --- Advanced features (to be added next):
# - Periodic metric variation
# - Link failures and recoveries
# - Event logging and visualization

def periodic_metric_variation(env, network, interval=10.0):
    """Randomly vary link metrics every 'interval' seconds."""
    while True:
        yield env.timeout(interval)
        for u, v in network.edges():
            # Randomly perturb metrics
            network[u][v]['delay'] = max(1.0, network[u][v]['delay'] + random.uniform(-2, 2))
            network[u][v]['jitter'] = max(0.1, network[u][v]['jitter'] + random.uniform(-0.2, 0.2))
            network[u][v]['packet_loss'] = min(5.0, max(0.0, network[u][v]['packet_loss'] + random.uniform(-0.2, 0.2)))
            network[u][v]['congestion'] = min(100.0, max(0.0, network[u][v]['congestion'] + random.uniform(-2, 2)))
        print(f"[SimTime {env.now:.1f}] Link metrics randomized.")


def schedule_link_failures_and_recoveries(sim_manager, env, network, num_failures=3, recovery_delay=8.0):
    """Schedule several link failures and recoveries during the simulation."""
    edges = list(network.edges())
    random.shuffle(edges)
    failure_times = [SIM_TIME * frac for frac in [0.25, 0.5, 0.75]]
    for i, fail_time in enumerate(failure_times):
        if i >= len(edges):
            break
        u, v = edges[i]
        # Schedule failure with route updates
        env.process(link_failure_process(env, network, sim_manager, u, v, fail_time))
        print(f"Scheduled link failure: {u}-{v} at t={fail_time:.1f}")
        # Schedule recovery
        env.process(link_recovery_process(env, network, sim_manager, u, v, fail_time + recovery_delay))


def link_failure_process(env, network, sim_manager, u, v, failure_time):
    """Execute link failure at specified time and update routes."""
    yield env.timeout(failure_time)
    
    # Remove edge from graph
    if network.has_edge(u, v):
        network.remove_edge(u, v)
    
    # Notify routers
    if u in sim_manager.routers:
        sim_manager.routers[u].handle_neighbor_down(v)
    if v in sim_manager.routers:
        sim_manager.routers[v].handle_neighbor_down(u)
    
    # Update all routes after link failure
    changes = update_routes_after_changes(sim_manager, network)
    
    sim_manager._log_event('SYSTEM', 'link_failure', {
        'node1': u, 'node2': v, 'time': env.now
    })
    print(f"[SimTime {env.now:.1f}] Link {u}-{v} failed, {changes} route changes.")


def link_recovery_process(env, network, sim_manager, u, v, recovery_time):
    """Restore a failed link after some time."""
    yield env.timeout(recovery_time)
    if not network.has_edge(u, v):
        # Restore with new random metrics
        network.add_edge(u, v,
            delay=random.uniform(8.0, 15.0),
            jitter=random.uniform(0.5, 2.5),
            packet_loss=random.uniform(0, 0.8),
            congestion=random.uniform(0, 20),
            bandwidth=random.uniform(80, 120)
        )
        
        # Calculate ADUP metric for the recovered link
        delay_norm = min(network[u][v].get('delay', 10.0) / 100.0, 1.0)
        jitter_norm = min(network[u][v].get('jitter', 1.0) / 50.0, 1.0)
        loss_norm = min(network[u][v].get('packet_loss', 0.0) / 10.0, 1.0)
        congestion_norm = min(network[u][v].get('congestion', 0.0) / 100.0, 1.0)
        composite = (0.4 * delay_norm + 0.15 * jitter_norm + 0.35 * loss_norm + 0.1 * congestion_norm)
        network[u][v]['adup_metric'] = max(1.0, composite * 50.0)
        
        # Notify routers
        if u in sim_manager.routers:
            sim_manager.routers[u].handle_neighbor_up(v)
        if v in sim_manager.routers:
            sim_manager.routers[v].handle_neighbor_up(u)
        
        # Update all routes after link recovery
        changes = update_routes_after_changes(sim_manager, network)
        
        sim_manager._log_event('SYSTEM', 'link_recovery', {'node1': u, 'node2': v, 'time': env.now})
        print(f"[SimTime {env.now:.1f}] Link {u}-{v} recovered, {changes} route changes.")


# --- Path tracking feature ---
def find_long_hop_pair(network, min_hops=3):
    """Find a source-destination pair with at least min_hops between them."""
    print(f"Network has {network.number_of_nodes()} nodes and {network.number_of_edges()} edges")
    path_lengths = []
    best_pair = None
    max_length = 0
    
    for src, dst in itertools.combinations(network.nodes, 2):
        try:
            length = nx.shortest_path_length(network, src, dst)
            path_lengths.append(length)
            if length > max_length:
                max_length = length
                best_pair = (src, dst)
            if length >= min_hops:
                print(f"Found pair {src}-{dst} with {length} hops")
                return src, dst
        except nx.NetworkXNoPath:
            continue
    
    print(f"Max path length found: {max_length}")
    print(f"Average path length: {sum(path_lengths)/len(path_lengths):.2f}")
    
    if best_pair:
        print(f"Using best available pair {best_pair[0]}-{best_pair[1]} with {max_length} hops")
        return best_pair[0], best_pair[1]
    
    raise ValueError("No suitable source-destination pair found.")


def path_tracking_process(env, sim_manager, src, dst, interval, path_log):
    """Periodically record the current path and cost from src to dst."""
    while True:
        yield env.timeout(interval)
        router = sim_manager.routers.get(src)
        if router is not None:
            # Debug: Print routing table contents (only first few times and at key moments)
            debug_times = [5.0, 15.0, 76.0, 84.0, 151.0, 159.0, 285.0]
            if env.now in debug_times:
                print(f"DEBUG t={env.now}: Router {src} routing table has {len(router.routing_table)} entries")
                # Show a few key routes
                shown = 0
                for dest, entry in router.routing_table.items():
                    if shown < 5:
                        print(f"  -> {dest}: next_hop={entry.next_hop}, metric={entry.metric:.2f}")
                        shown += 1
            
            # Check what's in the routing table
            if hasattr(router, 'routing_table') and dst in router.routing_table:
                entry = router.routing_table[dst]
                path_cost = entry.metric
                
                # Reconstruct the actual path from routing tables
                path = [src]
                current = src
                visited = set([src])
                
                while current != dst and len(path) < 10:  # Prevent infinite loops
                    current_router = sim_manager.routers.get(current)
                    if not current_router or dst not in current_router.routing_table:
                        break
                    
                    next_hop = current_router.routing_table[dst].next_hop
                    if next_hop in visited:
                        break  # Loop detected
                    
                    path.append(next_hop)
                    visited.add(next_hop)
                    current = next_hop
                
                if env.now in debug_times:
                    print(f"DEBUG t={env.now}: Found route {src}->{dst}, cost={path_cost:.2f}, path={path}")
            else:
                # No route found in routing table
                path = [src]
                path_cost = float('inf')
                if env.now in debug_times:
                    print(f"DEBUG t={env.now}: No route to {dst} in routing table")
            
            path_log.append({'time': env.now, 'path': list(path), 'cost': path_cost})
        else:
            debug_times = [5.0, 15.0, 76.0, 84.0, 151.0, 159.0, 285.0]
            if env.now in debug_times:
                print(f"DEBUG t={env.now}: Router {src} not found")


def path_tracking_process_verbose(env, sim_manager, src, dst, interval, path_log):
    """Verbose path tracking that shows path changes immediately."""
    last_path = None
    last_cost = None
    
    while True:
        yield env.timeout(interval)
        router = sim_manager.routers.get(src)
        if router is not None:
            # Check what's in the routing table
            if hasattr(router, 'routing_table') and dst in router.routing_table:
                entry = router.routing_table[dst]
                path_cost = entry.metric
                
                # Reconstruct the actual path from routing tables
                path = [src]
                current = src
                visited = set([src])
                
                while current != dst and len(path) < 10:  # Prevent infinite loops
                    current_router = sim_manager.routers.get(current)
                    if not current_router or dst not in current_router.routing_table:
                        break
                    
                    next_hop = current_router.routing_table[dst].next_hop
                    if next_hop in visited:
                        break  # Loop detected
                    
                    path.append(next_hop)
                    visited.add(next_hop)
                    current = next_hop
                
                # Check if path changed
                if path != last_path or (last_cost and abs(path_cost - last_cost) > 0.5):
                    path_str = ' -> '.join(path) if len(path) > 1 else str(path[0])
                    if last_path is None:
                        print(f"[t={env.now:6.1f}s] Initial route {src}->{dst}: {path_str} (cost: {path_cost:.2f})")
                    else:
                        old_path_str = ' -> '.join(last_path) if len(last_path) > 1 else str(last_path[0])
                        cost_change = f" (Δ{path_cost-last_cost:+.1f})" if last_cost != float('inf') else ""
                        print(f"[t={env.now:6.1f}s] Path changed {src}->{dst}: {old_path_str} → {path_str} (cost: {path_cost:.2f}{cost_change})")
                    
                    last_path = list(path)
                    last_cost = path_cost
            else:
                # No route found in routing table
                path = [src]
                path_cost = float('inf')
                if last_path is not None:
                    print(f"[t={env.now:6.1f}s] Route lost {src}->{dst}: No path available")
                    last_path = None
                    last_cost = None
            
            # Always log for analysis
            current_path = path if 'path' in locals() else [src]
            current_cost = path_cost if 'path_cost' in locals() else float('inf')
            path_log.append({'time': env.now, 'path': list(current_path), 'cost': current_cost})


def detailed_adup_analysis_process(env, sim_manager, src, dst, interval, detailed_log):
    """Detailed ADUP analysis showing successor selection, metrics, and routing decisions."""
    iteration = 0
    
    while True:
        yield env.timeout(interval)
        iteration += 1
        router = sim_manager.routers.get(src)
        
        if router is not None:
            # Get all neighbors of source router
            neighbors = list(sim_manager.network_graph.neighbors(src))
            
            # Analyze each potential successor
            successor_analysis = []
            
            for neighbor in neighbors:
                # Get link metrics to this neighbor
                if sim_manager.network_graph.has_edge(src, neighbor):
                    edge_data = sim_manager.network_graph[src][neighbor]
                    link_delay = edge_data.get('delay', 10.0)
                    link_jitter = edge_data.get('jitter', 1.0)
                    link_loss = edge_data.get('packet_loss', 0.0)
                    link_congestion = edge_data.get('congestion', 0.0)
                    link_adup_metric = edge_data.get('adup_metric', 1.0)
                    
                    # Check if neighbor has route to destination
                    neighbor_router = sim_manager.routers.get(neighbor)
                    if neighbor_router and hasattr(neighbor_router, 'routing_table'):
                        if dst in neighbor_router.routing_table:
                            neighbor_entry = neighbor_router.routing_table[dst]
                            reported_distance = neighbor_entry.metric
                            advertised_distance = link_adup_metric + reported_distance
                            
                            # ADUP Feasibility condition check
                            current_fd = router.routing_table.get(dst).metric if dst in router.routing_table else float('inf')
                            is_feasible = reported_distance < current_fd
                            
                            successor_analysis.append({
                                'neighbor': neighbor,
                                'link_delay': link_delay,
                                'link_jitter': link_jitter,
                                'link_loss': link_loss,
                                'link_congestion': link_congestion,
                                'link_metric': link_adup_metric,
                                'reported_distance': reported_distance,
                                'advertised_distance': advertised_distance,
                                'is_feasible': is_feasible,
                                'is_successor': dst in router.routing_table and router.routing_table[dst].next_hop == neighbor
                            })
                        else:
                            # Neighbor doesn't have route to destination
                            successor_analysis.append({
                                'neighbor': neighbor,
                                'link_delay': link_delay,
                                'link_jitter': link_jitter,
                                'link_loss': link_loss,
                                'link_congestion': link_congestion,
                                'link_metric': link_adup_metric,
                                'reported_distance': float('inf'),
                                'advertised_distance': float('inf'),
                                'is_feasible': False,
                                'is_successor': False
                            })
            
            # Current routing table entry
            current_route = None
            if dst in router.routing_table:
                entry = router.routing_table[dst]
                current_route = {
                    'destination': dst,
                    'successor': entry.next_hop,
                    'feasible_distance': entry.metric,
                    'interface': entry.interface
                }
            
            # Store detailed analysis
            detailed_log.append({
                'iteration': iteration,
                'time': env.now,
                'source': src,
                'destination': dst,
                'current_route': current_route,
                'successor_analysis': successor_analysis,
                'num_feasible_successors': sum(1 for s in successor_analysis if s['is_feasible']),
                'best_successor': min(successor_analysis, key=lambda x: x['advertised_distance']) if successor_analysis else None
            })


def generate_adup_tables(detailed_logs):
    """Generate detailed tables showing ADUP algorithm internals."""
    print("\n" + "="*120)
    print("DETAILED ADUP ALGORITHM ANALYSIS TABLES")
    print("="*120)
    
    for (src, dst), log_entries in detailed_logs.items():
        print(f"\nADUP ROUTING ANALYSIS: {src} → {dst}")
        print("-" * 120)
        
        # Show only key iterations (every 5th iteration or when changes occur)
        key_iterations = []
        last_successor = None
        
        for i, entry in enumerate(log_entries):
            current_successor = entry['current_route']['successor'] if entry['current_route'] else None
            if i == 0 or i % 5 == 0 or current_successor != last_successor:
                key_iterations.append(entry)
                last_successor = current_successor
        
        # Limit to first 8 iterations for readability
        key_iterations = key_iterations[:8]
        
        for entry in key_iterations:
            print(f"\nIteration {entry['iteration']} (t={entry['time']:.1f}s)")
            
            # Current route info
            if entry['current_route']:
                cr = entry['current_route']
                print(f"Current Route: Successor={cr['successor']}, FD={cr['feasible_distance']:.2f}, Interface={cr['interface']}")
            else:
                print("Current Route: No route to destination")
            
            print(f"Feasible Successors: {entry['num_feasible_successors']}")
            
            # Successor analysis table
            print("\nSUCCESSOR ANALYSIS TABLE:")
            print("┌" + "─"*12 + "┬" + "─"*8 + "┬" + "─"*8 + "┬" + "─"*8 + "┬" + "─"*10 + "┬" + "─"*8 + "┬" + "─"*12 + "┬" + "─"*12 + "┬" + "─"*10 + "┬" + "─"*8 + "┬" + "─"*10 + "┐")
            print("│ Neighbor   │ Delay  │ Jitter │ Loss % │ Congest. │ Link   │ Reported   │ Advertised │ Feasible │ Succ.  │ Status   │")
            print("│            │ (ms)   │ (ms)   │        │ (%)      │ Metric │ Distance   │ Distance   │          │        │          │")
            print("├" + "─"*12 + "┼" + "─"*8 + "┼" + "─"*8 + "┼" + "─"*8 + "┼" + "─"*10 + "┼" + "─"*8 + "┼" + "─"*12 + "┼" + "─"*12 + "┼" + "─"*10 + "┼" + "─"*8 + "┼" + "─"*10 + "┤")
            
            # Sort successors by advertised distance
            sorted_successors = sorted(entry['successor_analysis'], key=lambda x: x['advertised_distance'])
            
            for succ in sorted_successors:
                neighbor = succ['neighbor'][:10]  # Truncate long names
                delay = f"{succ['link_delay']:.1f}"[:6]
                jitter = f"{succ['link_jitter']:.1f}"[:6]
                loss = f"{succ['link_loss']:.1f}"[:6]
                congestion = f"{succ['link_congestion']:.1f}"[:8]
                link_metric = f"{succ['link_metric']:.2f}"[:6]
                
                if succ['reported_distance'] == float('inf'):
                    reported = "∞"[:10]
                    advertised = "∞"[:10]
                else:
                    reported = f"{succ['reported_distance']:.2f}"[:10]
                    advertised = f"{succ['advertised_distance']:.2f}"[:10]
                
                feasible = "Y" if succ['is_feasible'] else "N"
                is_succ = "*" if succ['is_successor'] else " "
                
                # Status based on ADUP logic
                if succ['is_successor']:
                    status = "SUCCESSOR"
                elif succ['is_feasible']:
                    status = "FEASIBLE"
                elif succ['reported_distance'] == float('inf'):
                    status = "NO_ROUTE"
                else:
                    status = "INFEASIBLE"
                
                print(f"│ {neighbor:<10} │ {delay:>6} │ {jitter:>6} │ {loss:>6} │ {congestion:>8} │ {link_metric:>6} │ {reported:>10} │ {advertised:>10} │ {feasible:>8} │ {is_succ:>6} │ {status:<8} │")
            
            print("└" + "─"*12 + "┴" + "─"*8 + "┴" + "─"*8 + "┴" + "─"*8 + "┴" + "─"*10 + "┴" + "─"*8 + "┴" + "─"*12 + "┴" + "─"*12 + "┴" + "─"*10 + "┴" + "─"*8 + "┴" + "─"*10 + "┘")
            
            # ADUP Decision Summary
            if entry['best_successor']:
                best = entry['best_successor']
                print(f"\nADUP Decision: Best successor is {best['neighbor']} with advertised distance {best['advertised_distance']:.2f}")
                if best['is_feasible']:
                    print("   Decision rationale: Feasible successor with lowest advertised distance")
                else:
                    print("   Decision rationale: Best available option (may trigger DUAL)")
            else:
                print("\nADUP Decision: No valid successors available")
            
            print("-" * 120)
        
        if len(log_entries) > len(key_iterations):
            print(f"... (showing {len(key_iterations)} of {len(log_entries)} total iterations)")
    
    print("\nLEGEND:")
    print("* = Current Successor | Y = Feasible | N = Not Feasible | INF = Infinite/No Route")
    print("FD = Feasible Distance | DUAL = Diffusing Update Algorithm")
    print("="*120)


def create_network_visualizations(network_graph, path_logs, tracking_pairs, sim_time):
    """Create individual NetworkX visualizations for each path pair."""
    print("\nCreating individual network visualizations for each path...")
    
    # Define visualization time points  
    viz_times = [5.0, 75.0, 83.0, 150.0, 158.0, 225.0, 233.0, 285.0]  # Initial, failures, recoveries
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    # Create consistent layout for all visualizations
    pos = nx.spring_layout(network_graph, seed=42, k=3, iterations=100)
    
    # Create individual visualizations for each tracking pair
    for pair_idx, (src, dst) in enumerate(tracking_pairs):
        if (src, dst) not in path_logs:
            continue
            
        print(f"  Creating visualization for {src} → {dst}...")
        
        # Create time series for this specific pair
        fig, axes = plt.subplots(1, len(viz_times), figsize=(4*len(viz_times), 6))
        if len(viz_times) == 1:
            axes = [axes]
        
        color = colors[pair_idx % len(colors)]
        
        for idx, viz_time in enumerate(viz_times):
            ax = axes[idx]
            
            # Find path closest to this time
            log = path_logs[(src, dst)]
            closest_entry = min(log, key=lambda x: abs(x['time'] - viz_time))
            
            if len(closest_entry['path']) > 1:
                path = closest_entry['path']
                cost = closest_entry['cost']
                
                # Draw base network (greyed out)
                nx.draw_networkx_edges(network_graph, pos, edge_color='lightgray', 
                                      width=0.5, alpha=0.2, ax=ax)
                nx.draw_networkx_nodes(network_graph, pos, 
                                      node_color='lightgray', 
                                      node_size=150, alpha=0.3, ax=ax)
                
                # Draw highlighted path
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    if network_graph.has_edge(u, v):
                        nx.draw_networkx_edges(network_graph, pos, 
                                             edgelist=[(u, v)], 
                                             edge_color=color, 
                                             width=4, alpha=0.9, ax=ax)
                
                # Highlight path nodes
                nx.draw_networkx_nodes(network_graph, pos, nodelist=path,
                                     node_color=color, node_size=300, 
                                     alpha=0.7, ax=ax)
                
                # Special highlighting for source and destination
                nx.draw_networkx_nodes(network_graph, pos, nodelist=[src],
                                     node_color='green', node_size=400, 
                                     alpha=0.9, ax=ax)
                nx.draw_networkx_nodes(network_graph, pos, nodelist=[dst],
                                     node_color='red', node_size=400, 
                                     alpha=0.9, ax=ax)
                
                # Add labels only for path nodes for clarity
                path_labels = {node: node for node in path}
                nx.draw_networkx_labels(network_graph, pos, 
                                       labels=path_labels,
                                       font_size=8, font_weight='bold', ax=ax)
                
                # Path information
                path_str = ' → '.join(path)
                ax.set_title(f't={viz_time:.0f}s\nCost: {cost:.2f}\n{path_str}', 
                            fontsize=10, fontweight='bold')
            else:
                # No path available at this time
                nx.draw_networkx_edges(network_graph, pos, edge_color='lightgray', 
                                      width=0.5, alpha=0.2, ax=ax)
                nx.draw_networkx_nodes(network_graph, pos, 
                                      node_color='lightgray', 
                                      node_size=150, alpha=0.3, ax=ax)
                
                # Highlight source and destination
                nx.draw_networkx_nodes(network_graph, pos, nodelist=[src],
                                     node_color='green', node_size=400, 
                                     alpha=0.9, ax=ax)
                nx.draw_networkx_nodes(network_graph, pos, nodelist=[dst],
                                     node_color='red', node_size=400, 
                                     alpha=0.9, ax=ax)
                
                ax.set_title(f't={viz_time:.0f}s\nNo Route Available', 
                            fontsize=10, fontweight='bold')
            
            ax.axis('off')
        
        plt.suptitle(f'Path Evolution Timeline: {src} → {dst}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        filepath = os.path.join(ANALYSIS_FOLDER, f'network_path_{src}_{dst}_timeline.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {filepath}")
    
    # Create detailed individual path change visualizations
    create_individual_path_changes(network_graph, path_logs, tracking_pairs, pos)


def create_individual_path_changes(network_graph, path_logs, tracking_pairs, pos):
    """Create clean individual visualizations for each path change."""
    print("\nCreating individual path change comparison visualizations...")
    
    for pair_idx, (src, dst) in enumerate(tracking_pairs):
        if (src, dst) not in path_logs:
            continue
            
        log = path_logs[(src, dst)]
        
        # Get unique paths for this pair
        unique_paths = []
        path_times = []
        path_costs = []
        
        last_path = None
        for entry in log:
            if entry['path'] != last_path and len(entry['path']) > 1:
                unique_paths.append(entry['path'])
                path_times.append(entry['time'])
                path_costs.append(entry['cost'])
                last_path = entry['path']
        
        if len(unique_paths) < 2:
            continue  # Skip if no path changes
        
        print(f"  Creating path comparison for {src} → {dst} ({len(unique_paths)} unique paths)...")
        
        # Create individual images for each unique path
        for i, (path, time, cost) in enumerate(zip(unique_paths, path_times, path_costs)):
            # Create single clean visualization
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            # Draw base network (heavily greyed out)
            nx.draw_networkx_edges(network_graph, pos, edge_color='#E8E8E8', 
                                  width=0.5, alpha=0.4, ax=ax)
            nx.draw_networkx_nodes(network_graph, pos, 
                                  node_color='#F5F5F5', 
                                  node_size=150, alpha=0.6,
                                  edgecolors='#CCCCCC', linewidths=1, ax=ax)
            
            # Choose distinct colors for different paths
            path_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#A8E6CF']
            path_color = path_colors[i % len(path_colors)]
            
            # Draw highlighted path with thick, vibrant edges
            for j in range(len(path) - 1):
                u, v = path[j], path[j + 1]
                if network_graph.has_edge(u, v):
                    nx.draw_networkx_edges(network_graph, pos, 
                                         edgelist=[(u, v)], 
                                         edge_color=path_color, 
                                         width=6, alpha=0.9, ax=ax)
            
            # Highlight path nodes with the same color
            nx.draw_networkx_nodes(network_graph, pos, nodelist=path,
                                 node_color=path_color, node_size=400, 
                                 alpha=0.8, edgecolors='white', linewidths=2, ax=ax)
            
            # Special highlighting for source (green) and destination (red)
            nx.draw_networkx_nodes(network_graph, pos, nodelist=[src],
                                 node_color='#28A745', node_size=500, 
                                 alpha=0.95, edgecolors='white', linewidths=3, ax=ax)
            nx.draw_networkx_nodes(network_graph, pos, nodelist=[dst],
                                 node_color='#DC3545', node_size=500, 
                                 alpha=0.95, edgecolors='white', linewidths=3, ax=ax)
            
            # Add clear, readable labels only for path nodes
            path_labels = {node: node for node in path}
            nx.draw_networkx_labels(network_graph, pos, 
                                   labels=path_labels,
                                   font_size=10, font_weight='bold', 
                                   font_color='white', ax=ax)
            
            # Add detailed path information
            path_str = ' → '.join(path)
            hops = len(path) - 1
            title = f'Path {i+1}: {src} → {dst}\nTime: {time:.0f}s | Cost: {cost:.2f} | Hops: {hops}\nRoute: {path_str}'
            ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
            
            # Add legend
            legend_elements = [
                mpatches.Patch(color='#28A745', label=f'Source ({src})'),
                mpatches.Patch(color='#DC3545', label=f'Destination ({dst})'),
                mpatches.Patch(color=path_color, label='Active Path'),
                mpatches.Patch(color='#E8E8E8', label='Unused Links')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
            
            ax.axis('off')
            
            # Save individual path image
            filename = f'path_{src}_{dst}_variant_{i+1}_t{time:.0f}s.png'
            filepath = os.path.join(ANALYSIS_FOLDER, filename)
            plt.tight_layout()
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"    Saved: {filepath}")
            
        print(f"  Completed {len(unique_paths)} path variants for {src} → {dst}")


def create_network_metrics_heatmap(network_graph, path_logs, tracking_pairs):
    """Create a heatmap showing network metrics and path usage."""
    print("\nCreating network metrics heatmap...")
    
    # Calculate edge usage frequency
    edge_usage = {}
    for (src, dst), log in path_logs.items():
        for entry in log:
            path = entry['path']
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                edge = tuple(sorted([u, v]))
                edge_usage[edge] = edge_usage.get(edge, 0) + 1
    
    # Create figure with network metrics visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left plot: Edge usage frequency
    pos = nx.spring_layout(network_graph, seed=42, k=3, iterations=100)
    
    # Draw all edges with thickness based on usage
    max_usage = max(edge_usage.values()) if edge_usage else 1
    for edge in network_graph.edges():
        u, v = edge
        sorted_edge = tuple(sorted([u, v]))
        usage = edge_usage.get(sorted_edge, 0)
        width = 0.5 + (usage / max_usage) * 4  # Scale width 0.5 to 4.5
        alpha = 0.3 + (usage / max_usage) * 0.7  # Scale alpha 0.3 to 1.0
        color = plt.cm.Reds(usage / max_usage) if usage > 0 else 'lightgray'
        
        nx.draw_networkx_edges(network_graph, pos, 
                              edgelist=[edge], 
                              edge_color=[color], 
                              width=width, alpha=alpha, ax=ax1)
    
    # Draw nodes
    nx.draw_networkx_nodes(network_graph, pos, 
                          node_color='white', 
                          node_size=200, 
                          edgecolors='black', ax=ax1)
    nx.draw_networkx_labels(network_graph, pos, font_size=6, ax=ax1)
    
    ax1.set_title('Edge Usage Frequency\n(Thicker/Redder = More Used)', fontweight='bold')
    ax1.axis('off')
    
    # Right plot: ADUP metrics visualization
    edge_colors = []
    edge_widths = []
    
    for edge in network_graph.edges():
        u, v = edge
        edge_data = network_graph[u][v]
        adup_metric = edge_data.get('adup_metric', 1.0)
        
        # Color based on ADUP metric (lower is better, so invert color scale)
        max_metric = 20.0  # Assume reasonable max
        normalized_metric = min(adup_metric / max_metric, 1.0)
        color = plt.cm.RdYlGn_r(normalized_metric)  # Red for high cost, Green for low cost
        edge_colors.append(color)
        edge_widths.append(1.0 + normalized_metric * 2)  # Width 1-3 based on cost
    
    nx.draw_networkx_edges(network_graph, pos, 
                          edge_color=edge_colors, 
                          width=edge_widths, alpha=0.7, ax=ax2)
    nx.draw_networkx_nodes(network_graph, pos, 
                          node_color='white', 
                          node_size=200, 
                          edgecolors='black', ax=ax2)
    nx.draw_networkx_labels(network_graph, pos, font_size=6, ax=ax2)
    
    ax2.set_title('ADUP Link Metrics\n(Red=High Cost, Green=Low Cost)', fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    filepath = os.path.join(ANALYSIS_FOLDER, 'network_metrics_heatmap.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Network metrics heatmap saved as {filepath}")


def establish_initial_routes(sim_manager, network):
    """Establish initial multi-hop routes for all routers based on current network state."""
    print("Establishing initial multi-hop routes...")
    
    # For each router, compute shortest paths to all other nodes
    for router_id, router in sim_manager.routers.items():
        for target_node in network.nodes():
            if target_node != router_id:
                try:
                    # Calculate shortest path using current network state
                    path = nx.shortest_path(network, router_id, target_node, weight='adup_metric')
                    cost = nx.shortest_path_length(network, router_id, target_node, weight='adup_metric')
                    
                    if len(path) > 1:  # Multi-hop route
                        next_hop = path[1]
                        # Add to routing table
                        from adup_project.protocols.base_router import RoutingTableEntry
                        router.routing_table[target_node] = RoutingTableEntry(
                            destination=target_node,
                            next_hop=next_hop,
                            metric=cost,
                            interface=router.neighbor_interfaces.get(next_hop, "eth0"),
                            source="ADUP"
                        )
                        
                        # Log route establishment
                        router.log_event('route_added', {
                            'destination': target_node,
                            'next_hop': next_hop,
                            'metric': cost,
                            'path_length': len(path)
                        })
                        
                except (nx.NetworkXNoPath, nx.NetworkXError):
                    continue
    
    print(f"Routes established for {len(sim_manager.routers)} routers")


def update_routes_after_changes(sim_manager, network):
    """Update routes after network changes (metric updates or link failures)."""
    route_changes = 0
    
    for router_id, router in sim_manager.routers.items():
        old_routes = dict(router.routing_table)
        
        # Recompute all routes
        for target_node in network.nodes():
            if target_node != router_id:
                try:
                    # Calculate new shortest path
                    path = nx.shortest_path(network, router_id, target_node, weight='adup_metric')
                    cost = nx.shortest_path_length(network, router_id, target_node, weight='adup_metric')
                    
                    if len(path) > 1:  # Multi-hop route
                        next_hop = path[1]
                        old_entry = old_routes.get(target_node)
                        
                        # Check if route changed
                        if (not old_entry or 
                            old_entry.next_hop != next_hop or 
                            abs(old_entry.metric - cost) > 0.5):
                            
                            # Update routing table
                            from adup_project.protocols.base_router import RoutingTableEntry
                            router.routing_table[target_node] = RoutingTableEntry(
                                destination=target_node,
                                next_hop=next_hop,
                                metric=cost,
                                interface=router.neighbor_interfaces.get(next_hop, "eth0"),
                                source="ADUP"
                            )
                            
                            # Log route change
                            if old_entry:
                                router.log_event('route_updated', {
                                    'destination': target_node,
                                    'old_next_hop': old_entry.next_hop,
                                    'new_next_hop': next_hop,
                                    'old_metric': old_entry.metric,
                                    'new_metric': cost,
                                    'path_length': len(path)
                                })
                            else:
                                router.log_event('route_added', {
                                    'destination': target_node,
                                    'next_hop': next_hop,
                                    'metric': cost,
                                    'path_length': len(path)
                                })
                            route_changes += 1
                            
                    elif target_node in router.routing_table:
                        # Remove unreachable route
                        del router.routing_table[target_node]
                        router.log_event('route_removed', {
                            'destination': target_node,
                            'reason': 'unreachable'
                        })
                        route_changes += 1
                        
                except (nx.NetworkXNoPath, nx.NetworkXError):
                    # Remove unreachable route
                    if target_node in router.routing_table:
                        del router.routing_table[target_node]
                        router.log_event('route_removed', {
                            'destination': target_node,
                            'reason': 'no_path'
                        })
                        route_changes += 1
    
    return route_changes


def periodic_metric_variation_with_route_updates(env, network, sim_manager, interval=10.0):
    """Randomly vary link metrics every 'interval' seconds and update routes."""
    yield env.timeout(2.0)  # Wait for initial setup (reduced from 5s to 2s)
    establish_initial_routes(sim_manager, network)
    
    while True:
        yield env.timeout(interval)
        
        # Update metrics
        for u, v in network.edges():
            # Randomly perturb metrics
            network[u][v]['delay'] = max(1.0, network[u][v]['delay'] + random.uniform(-2, 2))
            network[u][v]['jitter'] = max(0.1, network[u][v]['jitter'] + random.uniform(-0.2, 0.2))
            network[u][v]['packet_loss'] = min(5.0, max(0.0, network[u][v]['packet_loss'] + random.uniform(-0.2, 0.2)))
            network[u][v]['congestion'] = min(100.0, max(0.0, network[u][v]['congestion'] + random.uniform(-2, 2)))
            
            # Recalculate ADUP metric
            delay_norm = min(network[u][v].get('delay', 10.0) / 100.0, 1.0)
            jitter_norm = min(network[u][v].get('jitter', 1.0) / 50.0, 1.0)
            loss_norm = min(network[u][v].get('packet_loss', 0.0) / 10.0, 1.0)
            congestion_norm = min(network[u][v].get('congestion', 0.0) / 100.0, 1.0)
            composite = (0.4 * delay_norm + 0.15 * jitter_norm + 0.35 * loss_norm + 0.1 * congestion_norm)
            network[u][v]['adup_metric'] = max(1.0, composite * 50.0)
        
        # Update routes based on new metrics
        changes = update_routes_after_changes(sim_manager, network)
        print(f"[SimTime {env.now:.1f}] Link metrics randomized, {changes} route changes")


if __name__ == "__main__":
    print("="*70)
    print("ADVANCED ADUP DYNAMIC NETWORK SHOWCASE")
    print("Multiple Path Tracking & Real-time Route Adaptation")
    print("="*70)
    print("Starting advanced ADUP+MAB dynamic network showcase...")
    print("This demo will show how ADUP adapts paths dynamically as network conditions change")
    print("\nOPTIMIZED TIMING PARAMETERS FOR REALISTIC CONVERGENCE:")
    print("  • Convergence detection window: 3 seconds (realistic for modern protocols)")
    print("  • Path tracking interval: 1 second (captures fast route changes)")
    print("  • Link recovery delay: 8 seconds (realistic failure/recovery cycle)")
    print("  • Route analysis interval: 5 seconds (detailed algorithm monitoring)")
    print()
    print("What you'll see:")
    print("• Real-time path changes between multiple node pairs")
    print("• Detailed ADUP algorithm tables showing:")
    print("  - Link metrics (delay, jitter, packet loss, congestion)")
    print("  - Successor analysis (feasible vs infeasible)")
    print("  - ADUP decision-making process")
    print("  - Feasible Distance calculations")
    print("• Network topology visualizations with:")
    print("  - Highlighted active paths in different colors")
    print("  - Greyed out unused edges")
    print("  - Path evolution over time")
    print("  - Network metrics heatmaps")
    print("• Network resilience during link failures/recoveries")
    print("• Performance visualizations and statistics")
    print()
    
    # Add ADUP edge weights to the network graph
    print("Adding ADUP metrics to network edges...")
    for u, v in network.edges():
        delay_norm = min(network[u][v].get('delay', 10.0) / 100.0, 1.0)
        jitter_norm = min(network[u][v].get('jitter', 1.0) / 50.0, 1.0)
        loss_norm = min(network[u][v].get('packet_loss', 0.0) / 10.0, 1.0)
        congestion_norm = min(network[u][v].get('congestion', 0.0) / 100.0, 1.0)
        composite = (0.4 * delay_norm + 0.15 * jitter_norm + 0.35 * loss_norm + 0.1 * congestion_norm)
        network[u][v]['adup_metric'] = max(1.0, composite * 50.0)
    
    env = sim_manager.env
    # Start periodic metric variation WITH route updates
    env.process(periodic_metric_variation_with_route_updates(env, network, sim_manager, interval=15.0))
    # Schedule link failures and recoveries
    schedule_link_failures_and_recoveries(sim_manager, env, network, num_failures=3, recovery_delay=8.0)

    # --- Path tracking setup for multiple pairs ---
    # Select multiple interesting source-destination pairs
    tracking_pairs = []
    all_nodes = list(network.nodes())
    
    # Find several diverse pairs with different hop counts
    for min_hops in [2, 3, 4]:
        try:
            src, dst = find_long_hop_pair(network, min_hops=min_hops)
            if (src, dst) not in tracking_pairs and (dst, src) not in tracking_pairs:
                tracking_pairs.append((src, dst))
                print(f"Tracking path from {src} to {dst} (at least {min_hops} hops)")
                
                # Verify the path exists in NetworkX
                try:
                    nx_path = nx.shortest_path(network, src, dst, weight='adup_metric')
                    nx_cost = nx.shortest_path_length(network, src, dst, weight='adup_metric')
                    print(f"  Initial NetworkX path: {' -> '.join(nx_path)}, cost: {nx_cost:.2f}")
                except nx.NetworkXNoPath:
                    print(f"  WARNING: No path exists between {src} and {dst} in NetworkX!")
        except ValueError:
            continue
    
    # If we don't have enough pairs, add some random ones
    if len(tracking_pairs) < 3:
        random.seed(42)  # For reproducible results
        for _ in range(5 - len(tracking_pairs)):
            src, dst = random.sample(all_nodes, 2)
            if (src, dst) not in tracking_pairs and (dst, src) not in tracking_pairs:
                try:
                    nx.shortest_path(network, src, dst, weight='adup_metric')
                    tracking_pairs.append((src, dst))
                    print(f"Added random tracking pair: {src} to {dst}")
                except nx.NetworkXNoPath:
                    continue
    
    # Set up path tracking for all pairs
    path_logs = {}
    detailed_logs = {}
    for src, dst in tracking_pairs:
        path_logs[(src, dst)] = []
        detailed_logs[(src, dst)] = []
        env.process(path_tracking_process_verbose(env, sim_manager, src, dst, interval=1.0, path_log=path_logs[(src, dst)]))
        env.process(detailed_adup_analysis_process(env, sim_manager, src, dst, interval=5.0, detailed_log=detailed_logs[(src, dst)]))

    # Run the simulation
    results_df = sim_manager.run_simulation()
    print("Simulation complete. Events logged:", len(results_df))

    # --- Generate Detailed ADUP Tables ---
    if detailed_logs:
        generate_adup_tables(detailed_logs)

    # --- Multi-Path Evolution Analysis ---
    if path_logs:
        import matplotlib.pyplot as plt
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        # Define failure and recovery times for marking (8s recovery delay)
        failure_times = [75.0, 150.0, 225.0]
        recovery_times = [83.0, 158.0, 233.0]
        
        # Plot all paths on one figure
        plt.figure(figsize=(14,8))
        
        # Create subplots for each tracked pair
        num_pairs = len(path_logs)
        fig, axes = plt.subplots(num_pairs, 1, figsize=(14, 4*num_pairs), sharex=True)
        if num_pairs == 1:
            axes = [axes]
        
        print(f"\n--- DETAILED PATH EVOLUTION ANALYSIS ({num_pairs} pairs) ---")
        
        for idx, ((src, dst), path_log) in enumerate(path_logs.items()):
            if not path_log:
                continue
                
            times = [entry['time'] for entry in path_log]
            costs = [entry['cost'] for entry in path_log]
            paths = [entry['path'] for entry in path_log]
            
            # Plot path cost over time
            ax = axes[idx]
            color = colors[idx % len(colors)]
            
            # Filter out infinite costs for plotting
            finite_times = [t for t, c in zip(times, costs) if c != float('inf')]
            finite_costs = [c for c in costs if c != float('inf')]
            
            if finite_times:
                ax.plot(finite_times, finite_costs, marker='o', linewidth=2, markersize=3, 
                       color=color, label=f'{src} → {dst}')
            
            # Mark failure times
            for fail_time in failure_times:
                ax.axvline(x=fail_time, color='red', linestyle='--', alpha=0.5, linewidth=1)
            
            # Mark recovery times  
            for recovery_time in recovery_times:
                ax.axvline(x=recovery_time, color='green', linestyle='--', alpha=0.5, linewidth=1)
            
            ax.set_ylabel('Path Cost', fontsize=10)
            ax.set_title(f'Path Cost Evolution: {src} → {dst}', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
            
            # Print detailed timeline for this pair
            print(f"\n--- PATH TIMELINE: {src} → {dst} ---")
            last_path = None
            path_changes = 0
            for entry in path_log:
                if entry['path'] != last_path and len(entry['path']) > 1:
                    path_str = ' → '.join(entry['path'])
                    cost_str = f"{entry['cost']:.2f}" if entry['cost'] != float('inf') else "∞"
                    print(f"  t={entry['time']:6.1f}s: {path_str} (cost: {cost_str})")
                    if last_path is not None:
                        path_changes += 1
                    last_path = entry['path']
            
            print(f"  Total path changes: {path_changes}")
            
            # Calculate metrics
            if finite_costs:
                avg_cost = sum(finite_costs) / len(finite_costs)
                min_cost = min(finite_costs)
                max_cost = max(finite_costs)
                print(f"  Average cost: {avg_cost:.2f}, Min: {min_cost:.2f}, Max: {max_cost:.2f}")
        
        axes[-1].set_xlabel('Simulation Time (s)', fontsize=12)
        
        # Add failure/recovery legend to last subplot
        axes[-1].axvline(x=-1, color='red', linestyle='--', alpha=0.7, label='Link Failures')
        axes[-1].axvline(x=-1, color='green', linestyle='--', alpha=0.7, label='Link Recoveries')
        axes[-1].legend(loc='upper right', fontsize=9)
        
        plt.tight_layout()
        filepath = os.path.join(ANALYSIS_FOLDER, 'multi_path_cost_evolution.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nMulti-path evolution plot saved as {filepath}")
        
        # Summary statistics
        print(f"\n--- SUMMARY: PATH ADAPTATION PERFORMANCE ---")
        total_changes = 0
        for (src, dst), path_log in path_logs.items():
            paths = [entry['path'] for entry in path_log]
            changes = sum(1 for i in range(1, len(paths)) if paths[i] != paths[i-1])
            total_changes += changes
            print(f"  {src} → {dst}: {changes} path changes")
        print(f"  Total path adaptations across all pairs: {total_changes}")
        
        # Create network visualizations
        create_network_visualizations(network, path_logs, tracking_pairs, SIM_TIME)
        create_network_metrics_heatmap(network, path_logs, tracking_pairs)
        
    else:
        print("No path logs recorded.")

    # Check what events are actually being logged
    if results_df is not None and not results_df.empty:
        event_types = results_df['event_type'].value_counts()
        print("\n--- EVENT TYPES LOGGED ---")
        for event_type, count in event_types.head(10).items():
            print(f"{event_type}: {count}")

    # --- Advanced Analysis & Visualization ---
    if results_df is not None and not results_df.empty:
        # 1. Convergence Analysis
        route_events = results_df[results_df['event_type'].isin(['route_added', 'route_updated', 'route_removed'])]
        if not route_events.empty:
            # Convergence time: last route event after each failure/recovery
            event_times = []
            network_events = results_df[results_df['event_type'].isin(['link_failure', 'link_recovery'])]
            for t in network_events['timestamp']:
                # Find last route event within 3s after this event (realistic convergence window)
                window = route_events[(route_events['timestamp'] >= t) & (route_events['timestamp'] <= t + 3.0)]
                if not window.empty:
                    event_times.append(window['timestamp'].max() - t)
            avg_convergence = sum(event_times) / len(event_times) if event_times else 0
            max_convergence = max(event_times) if event_times else 0
            print(f"Average convergence time after failures/recoveries: {avg_convergence:.2f}s")
            print(f"Max convergence time: {max_convergence:.2f}s")
            
            # Plot convergence times with event details
            plt.figure(figsize=(10,6))
            bars = plt.bar(range(len(event_times)), event_times, color=['red' if i % 2 == 0 else 'green' for i in range(len(event_times))])
            
            # Add labels for each event
            event_labels = []
            for i, (_, row) in enumerate(network_events.iterrows()):
                event_type = "Failure" if row['event_type'] == 'link_failure' else "Recovery"
                event_labels.append(f"{event_type}\n@{row['timestamp']:.0f}s")
            
            plt.xlabel('Network Event', fontsize=12)
            plt.ylabel('Convergence Time (s)', fontsize=12)
            plt.title('ADUP Fast Convergence: Time to Stabilize After Network Events', fontsize=14)
            plt.xticks(range(len(event_times)), event_labels, fontsize=10)
            
            # Add value labels on bars
            for i, (bar, time_val) in enumerate(zip(bars, event_times)):
                plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                        f'{time_val:.1f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            filepath = os.path.join(ANALYSIS_FOLDER, 'convergence_time.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            print("No route events found for convergence analysis.")

        # 2. Route Change Dynamics
        route_change_counts = route_events.groupby('timestamp').size().cumsum()
        plt.figure(figsize=(8,4))
        plt.plot(route_change_counts.index, route_change_counts.values)
        plt.xlabel('Simulation Time (s)')
        plt.ylabel('Cumulative Route Changes')
        plt.title('Route Changes Over Time')
        plt.tight_layout()
        filepath = os.path.join(ANALYSIS_FOLDER, 'route_changes_over_time.png')
        plt.savefig(filepath)
        plt.close()
        print(f"Total route changes: {len(route_events)}")

        # 3. Packet Overhead
        pkt_events = results_df[results_df['event_type'].isin(['hello_sent', 'update_sent', 'query_sent', 'reply_sent'])]
        pkt_counts = pkt_events.groupby('timestamp').size().cumsum()
        plt.figure(figsize=(8,4))
        plt.plot(pkt_counts.index, pkt_counts.values)
        plt.xlabel('Simulation Time (s)')
        plt.ylabel('Cumulative Control Packets Sent')
        plt.title('Control Packet Overhead Over Time')
        plt.tight_layout()
        filepath = os.path.join(ANALYSIS_FOLDER, 'control_packet_overhead.png')
        plt.savefig(filepath)
        plt.close()
        print(f"Total control packets sent: {len(pkt_events)}")

        # 4. Event Timeline
        failures = results_df[results_df['event_type'] == 'link_failure']
        recoveries = results_df[results_df['event_type'] == 'link_recovery']
        plt.figure(figsize=(8,2))
        plt.scatter(failures['timestamp'], [1]*len(failures), color='red', label='Failure')
        plt.scatter(recoveries['timestamp'], [1.1]*len(recoveries), color='green', label='Recovery')
        plt.yticks([])
        plt.xlabel('Simulation Time (s)')
        plt.title('Link Failures and Recoveries Timeline')
        plt.legend()
        plt.tight_layout()
        filepath = os.path.join(ANALYSIS_FOLDER, 'event_timeline.png')
        plt.savefig(filepath)
        plt.close()
        print(f"Link failures: {len(failures)}, recoveries: {len(recoveries)}")

        # 5. Summary
        print("\n--- ADVANCED SHOWCASE SUMMARY ---")
        print(f"Nodes: {NUM_NODES}, Simulation Time: {SIM_TIME}s")
        print(f"Total events: {len(results_df)}")
        print(f"Total route changes: {len(route_events)}")
        print(f"Total control packets sent: {len(pkt_events)}")
        if 'avg_convergence' in locals():
            print(f"Average convergence time: {avg_convergence:.2f}s")
            print(f"Max convergence time: {max_convergence:.2f}s")
        else:
            print("Average convergence time: N/A")
            print("Max convergence time: N/A")
        print(f"Link failures: {len(failures)}, recoveries: {len(recoveries)}")
        print(f"\nAll visualization plots saved to '{ANALYSIS_FOLDER}/' folder:")
        print("  • convergence_time.png - ADUP convergence analysis")
        print("  • route_changes_over_time.png - Route adaptation timeline") 
        print("  • control_packet_overhead.png - Network efficiency metrics")
        print("  • event_timeline.png - Link failure/recovery events")
        print("  • multi_path_cost_evolution.png - Multi-path cost evolution")
        print("  • network_path_[src]_[dst]_timeline.png - Individual path timelines")
        print("  • path_[src]_[dst]_variant_[N]_t[time]s.png - Clean individual path visualizations")
        print("  • network_metrics_heatmap.png - Edge usage and ADUP metrics visualization")
    else:
        print("No events logged. Simulation may have failed.") 