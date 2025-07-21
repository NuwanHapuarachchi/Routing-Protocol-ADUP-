# ADUP Routing Protocol Implementation

## Advanced Diffusion Update Routing Protocol (ADUP)

This repository contains a comprehensive implementation and simulation environment for the ADUP ( Advanced Diffusion Update Routing Protocol) routing protocol, featuring dynamic network adaptation, multi-path analysis, and real-time visualization capabilities.

## Features

### Core Protocol Implementation
- **ADUP Protocol**: Advanced diffusion update routing protocol with enhanced metrics
- **Multi-Armed Bandit (MAB)**: Intelligent successor selection algorithm
- **Dynamic Adaptation**: Real-time adaptation to network changes
- **Fast Convergence**: Optimized convergence detection and recovery

### Advanced Simulation Capabilities
- **Large-Scale Networks**: Support for 30+ node networks
- **Dynamic Metrics**: Real-time link metric variations
- **Link Failures/Recoveries**: Realistic network fault simulation
- **Multi-Path Tracking**: Simultaneous monitoring of multiple source-destination pairs

### Visualization & Analysis
- **Real-time Path Visualization**: Network topology with highlighted active paths
- **ADUP Algorithm Tables**: Detailed successor analysis and decision-making process
- **Performance Metrics**: Convergence time, route changes, packet overhead
- **Network Heatmaps**: Link usage frequency and metric visualization

## Project Structure

```
adup_project/
├── protocols/           # Routing protocol implementations
│   ├── base_router.py  # Base router class
│   ├── adup.py        # ADUP protocol implementation
│   ├── rip.py         # RIP protocol (for comparison)
│   └── ospf.py        # OSPF protocol (for comparison)
├── topologies/         # Network topology generators
│   └── standard_topologies.py
├── analysis/           # Analysis and visualization tools
├── utils/             # Utility functions
├── simulation_manager.py  # Simulation orchestration
├── network_visualizer.py # Network visualization tools
├── large_network.py   # Advanced demo with dynamic networks
└── main.py           # Basic simulation entry point
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/NuwanHapuarachchi/Routing-Protocol-ADUP-.git
cd Routing-Protocol-ADUP-
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Simulation
Run a basic ADUP simulation:
```bash
python adup_project/main.py
```

### Advanced Dynamic Network Showcase
Experience the full ADUP capabilities with dynamic network adaptation:
```bash
python adup_project/large_network.py
```

This advanced demo includes:
- 30-node dynamic network
- Real-time metric variations
- Link failures and recoveries
- Multi-path tracking and visualization
- Detailed ADUP algorithm analysis
- Performance benchmarking

## Key Features Demonstrated

### 1. Dynamic Path Adaptation
- **Real-time Route Changes**: Monitor how ADUP adapts paths as network conditions change
- **Multi-Path Analysis**: Track multiple source-destination pairs simultaneously
- **Path Evolution Timeline**: Visualize how routes evolve over time

### 2. ADUP Algorithm Internals
- **Successor Analysis Tables**: Detailed view of feasible vs infeasible successors
- **Metric Calculations**: Link delay, jitter, packet loss, and congestion analysis
- **Decision Rationale**: Understanding ADUP's route selection logic

### 3. Network Resilience
- **Fast Convergence**: Typically converges within 3 seconds after network events
- **Fault Tolerance**: Automatic recovery from link failures
- **Efficient Overhead**: Minimal control packet exchange

### 4. Performance Visualization
- **Convergence Analysis**: Time to stabilize after network changes
- **Route Change Dynamics**: Cumulative route adaptations over time
- **Control Packet Overhead**: Network efficiency metrics
- **Network Heatmaps**: Visual representation of link usage and metrics

## Output Analysis

The simulation generates comprehensive analysis in the `path_analysis/` folder:

- `convergence_time.png` - ADUP convergence performance
- `multi_path_cost_evolution.png` - Path cost changes over time
- `network_path_[src]_[dst]_timeline.png` - Individual path evolution
- `path_[src]_[dst]_variant_[N]_t[time]s.png` - Clean path visualizations
- `network_metrics_heatmap.png` - Network usage and metrics analysis
- `route_changes_over_time.png` - Route adaptation timeline
- `control_packet_overhead.png` - Protocol efficiency metrics

## Technical Specifications

### Protocol Parameters
- **Convergence Window**: 3 seconds (realistic for modern protocols)
- **Path Tracking Interval**: 1 second (captures fast route changes)
- **Link Recovery Delay**: 8 seconds (realistic failure/recovery cycle)
- **Metric Update Interval**: 15 seconds (dynamic network conditions)

### ADUP Metrics
- **Delay Normalization**: Delay/100ms (40% weight)
- **Jitter Normalization**: Jitter/50ms (15% weight)
- **Packet Loss**: Loss%/10% (35% weight)
- **Congestion**: Congestion%/100% (10% weight)

## Research Applications

This implementation is suitable for:
- **Protocol Comparison Studies**: ADUP vs RIP, OSPF, EIGRP
- **Network Performance Analysis**: Large-scale network behavior
- **Algorithm Research**: Distance-vector protocol enhancements
- **Educational Purposes**: Understanding modern routing protocols

## Contributing

Contributions are welcome! Please feel free to submit pull requests, report bugs, or suggest features.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**Nuwan Hapuarachchi**
- GitHub: [@NuwanHapuarachchi](https://github.com/NuwanHapuarachchi)

## Acknowledgments

- SimPy for discrete-event simulation framework
- NetworkX for graph algorithms and network analysis
- Matplotlib for comprehensive visualization capabilities
