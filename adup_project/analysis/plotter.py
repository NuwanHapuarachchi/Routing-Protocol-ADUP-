"""
Analysis and plotting functions for ADUP simulation results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
import os


def setup_plotting():
    """Setup matplotlib and seaborn for high-quality plots."""
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (10, 6)


def plot_convergence_comparison(results: Dict[str, Dict[str, Any]], 
                              output_dir: str = "plots") -> None:
    """Plot convergence time comparison across protocols."""
    setup_plotting()
    
    protocols = []
    convergence_times = []
    
    for protocol, data in results.items():
        protocols.append(protocol)
        convergence_times.append(data.get('avg_convergence_time', 0.0))
    
    fig, ax = plt.subplots()
    bars = ax.bar(protocols, convergence_times, 
                  color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
    
    ax.set_ylabel('Average Convergence Time (seconds)')
    ax.set_title('Protocol Convergence Speed Comparison')
    ax.grid(True, alpha=0.3)
    
    for bar, value in zip(bars, convergence_times):
        if value > 0:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                   f'{value:.2f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/convergence_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_packet_overhead(results: Dict[str, Dict[str, Any]], 
                        output_dir: str = "plots") -> None:
    """Plot packet overhead comparison across protocols."""
    setup_plotting()
    
    protocols = []
    packets_sent = []
    
    for protocol, data in results.items():
        protocols.append(protocol)
        packets_sent.append(data.get('packets_sent', 0))
    
    fig, ax = plt.subplots()
    bars = ax.bar(protocols, packets_sent, 
                  color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
    
    ax.set_ylabel('Total Packets Sent')
    ax.set_title('Network Overhead Comparison')
    ax.grid(True, alpha=0.3)
    
    for bar, value in zip(bars, packets_sent):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{value}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/packet_overhead.png", dpi=300, bbox_inches='tight')
    plt.show()


def create_summary_report(results: Dict[str, Any], output_dir: str = "plots") -> None:
    """Create summary report with visualizations."""
    print("Generating analysis report...")
    
    if results:
        plot_convergence_comparison(results, output_dir)
        plot_packet_overhead(results, output_dir)
        
        # Create summary table with convergence time data
        summary_data = []
        for protocol, data in results.items():
            summary = {
                'Protocol': protocol,
                'Packets Sent': data.get('packets_sent', 0),
                'Packets Received': data.get('packets_received', 0),
                'Convergence Time (s)': f"{data.get('avg_convergence_time', 0.0):.2f}",
                'Runtime (s)': f"{data.get('actual_runtime', 0):.2f}"
            }
            summary_data.append(summary)
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(summary_data)
        os.makedirs(output_dir, exist_ok=True)
        csv_path = f"{output_dir}/protocol_comparison_results.csv"
        
        # Debug: Print DataFrame before saving
        print(f"Final DataFrame to save:\n{df}")
        
        # Save with explicit encoding and flushing
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        # Also save the old filename for backward compatibility
        df.to_csv(f"{output_dir}/summary_statistics.csv", index=False, encoding='utf-8')
        
        print(f"Analysis report saved to {output_dir}/")
        print(f"Summary CSV saved with {len(summary_data)} protocols to {csv_path}")
    else:
        print("No results to plot")
