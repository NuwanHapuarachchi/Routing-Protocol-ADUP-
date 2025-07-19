"""
Main entry point for ADUP protocol simulation and analysis.
"""

import random
import time
import sys
import os
from typing import Dict, Any, List

# Add the parent directory to Python path to enable imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adup_project.simulation_manager import SimulationManager
from adup_project.topologies.standard_topologies import get_scenario_topology, analyze_topology
from adup_project.analysis.plotter import create_summary_report


def run_scenario_tests(scenarios: List[str], protocols: List[str], 
                      simulation_time: float = 200.0) -> Dict[str, Any]:
    """Run tests for multiple scenarios and protocols."""
    all_results = {}
    
    print("=" * 60)
    print("ADUP Protocol Simulation Suite")
    print("=" * 60)
    
    for scenario in scenarios:
        print(f"\n--- Testing Scenario {scenario} ---")
        
        # Create topology for scenario
        if scenario == "A":
            topology = get_scenario_topology("CONVERGENCE", num_nodes=5)
        elif scenario == "B":
            topology = get_scenario_topology("PATH_INTELLIGENCE")
        elif scenario == "C":
            topology = get_scenario_topology("STABILITY", num_nodes=6)
        elif scenario == "D":
            topology = get_scenario_topology("SCALABILITY", num_nodes=10)
        else:
            print(f"Unknown scenario: {scenario}")
            continue
        
        # Analyze topology
        topo_analysis = analyze_topology(topology)
        print(f"Topology: {topo_analysis['num_nodes']} nodes, "
              f"{topo_analysis['num_edges']} edges")
        
        scenario_results = {}
        
        for protocol in protocols:
            print(f"\nRunning {protocol} simulation...")
            
            try:
                # Create simulation manager
                sim_manager = SimulationManager(
                    protocol_type=protocol,
                    network_graph=topology.copy(),
                    simulation_time=simulation_time
                )
                
                # Schedule link failures for convergence testing
                if scenario in ["A", "C"]:  # Convergence and stability scenarios
                    nodes = list(topology.nodes())
                    if len(nodes) >= 2:
                        node1, node2 = random.sample(nodes, 2)
                        if topology.has_edge(node1, node2):
                            failure_time = simulation_time * 0.3
                            sim_manager.schedule_link_failure(failure_time, node1, node2)
                            print(f"Scheduled link failure: {node1} - {node2} at {failure_time:.1f}s")
                
                # Run simulation
                events_df = sim_manager.run_simulation()
                
                # Analyze results
                analysis = sim_manager.analyze_convergence()
                router_stats = sim_manager.get_router_statistics()
                
                # Store results
                scenario_results[protocol] = {
                    'analysis': analysis,
                    'router_stats': router_stats,
                    'events_df': events_df
                }
                
                print(f"{protocol} completed: {analysis.get('total_events', 0)} events, "
                      f"{analysis.get('packets_sent', 0)} packets sent")
                
            except Exception as e:
                print(f"Error running {protocol} simulation: {e}")
                scenario_results[protocol] = {'error': str(e)}
        
        all_results[f"Scenario_{scenario}"] = scenario_results
    
    return all_results


def generate_comparative_analysis(results: Dict[str, Any]) -> None:
    """Generate comparative analysis and plots from simulation results."""
    print("\n" + "=" * 60)
    print("GENERATING COMPARATIVE ANALYSIS")
    print("=" * 60)
    
    # Aggregate results across scenarios
    overall_results = {}
    protocol_names = set()
    
    for scenario_results in results.values():
        protocol_names.update(scenario_results.keys())
    
    for protocol in protocol_names:
        total_packets_sent = 0
        total_packets_received = 0
        total_runtime = 0
        total_convergence_time = 0.0
        scenario_count = 0
        
        for scenario_results in results.values():
            if protocol in scenario_results and 'analysis' in scenario_results[protocol]:
                analysis = scenario_results[protocol]['analysis']
                total_packets_sent += analysis.get('packets_sent', 0)
                total_packets_received += analysis.get('packets_received', 0)
                total_runtime += analysis.get('actual_runtime', 0)
                total_convergence_time += analysis.get('avg_convergence_time', 0.0)
                scenario_count += 1
        
        if scenario_count > 0:
            overall_results[protocol] = {
                'packets_sent': total_packets_sent,
                'packets_received': total_packets_received,
                'packet_delivery_ratio': total_packets_received / max(1, total_packets_sent),
                'actual_runtime': total_runtime / scenario_count,
                'avg_convergence_time': total_convergence_time / scenario_count,  # Add convergence time
                'scenarios_tested': scenario_count
            }
    
    # Print summary
    print("\nOverall Results Summary:")
    print("-" * 40)
    for protocol, data in overall_results.items():
        packets_sent = data.get('packets_sent', 0)
        packets_received = data.get('packets_received', 0)
        runtime = data.get('actual_runtime', 0)
        delivery_ratio = data.get('packet_delivery_ratio', 0)
        
        print(f"{protocol:>8}: {packets_sent:>6} sent, {packets_received:>6} received, "
              f"{delivery_ratio:>6.3f} ratio, {runtime:>6.2f}s runtime")
    
    # Create overall comparison report
    if overall_results:
        create_summary_report(overall_results, "plots")
    
    print(f"\nComparative analysis complete!")
    print(f"Check the 'plots/' directory for generated visualizations.")


def run_comprehensive_evaluation() -> None:
    """Run comprehensive evaluation of all protocols across all scenarios."""
    random.seed(42)  # For reproducible results
    
    protocols = ["ADUP", "RIP", "OSPF"]
    scenarios = ["A", "B", "C"]  # Reduced for demo
    simulation_time = 150.0
    
    print("ADUP Protocol Evaluation Suite")
    print("Testing protocols:", ", ".join(protocols))
    print("Testing scenarios:", ", ".join(scenarios))
    print(f"Simulation time per test: {simulation_time} seconds")
    print()
    
    start_time = time.time()
    
    # Run all scenario tests
    results = run_scenario_tests(scenarios, protocols, simulation_time)
    
    # Generate comparative analysis
    generate_comparative_analysis(results)
    
    end_time = time.time()
    
    print(f"\nTotal evaluation time: {end_time - start_time:.2f} seconds")
    print("Evaluation complete!")


def run_quick_demo() -> None:
    """Run a quick demonstration with a simple scenario."""
    print("ADUP Quick Demonstration")
    print("=" * 40)
    
    protocols = ["ADUP", "RIP"]
    simulation_time = 100.0
    
    topology = get_scenario_topology("PATH_INTELLIGENCE")
    topo_analysis = analyze_topology(topology)
    
    print(f"Testing diamond topology: {topo_analysis['num_nodes']} nodes, "
          f"{topo_analysis['num_edges']} edges")
    
    results = {}
    
    for protocol in protocols:
        print(f"\nRunning {protocol}...")
        
        sim_manager = SimulationManager(
            protocol_type=protocol,
            network_graph=topology.copy(),
            simulation_time=simulation_time
        )
        
        # Schedule a link failure for testing
        sim_manager.schedule_link_failure(50.0, "A", "B")
        
        events_df = sim_manager.run_simulation()
        analysis = sim_manager.analyze_convergence()
        
        results[protocol] = analysis
        
        print(f"{protocol}: {analysis.get('packets_sent', 0)} packets sent, "
              f"{analysis.get('actual_runtime', 0):.2f}s runtime")
    
    # Generate quick comparison
    create_summary_report(results, "plots")
    
    print("\nQuick demo complete! Check 'plots/' for results.")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        run_quick_demo()
    else:
        run_comprehensive_evaluation()