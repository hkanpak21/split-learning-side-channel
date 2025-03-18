# Split Learning Simulation

This directory contains code for simulating a split learning setup and measuring timing information at the cut layer.

## Contents

- `SideChannel.py`: Main simulation script implementing a simple MNIST neural network (784-128-32-10) with the split between layers 128 and 32
- `results/`: Directory containing output files from the timing analysis:
  - `timing_histogram.png`: Histogram showing distribution of processing times for each digit
  - `timing_boxplot.png`: Box plot comparing timing distributions by digit
  - `transition_heatmap.png`: Heatmap showing average timing for transitions between digits
  - `transition_counts.png`: Heatmap showing frequency of transitions
  - `transition_times.csv`: CSV file with average timing data for transitions
  - `detailed_transition_stats.csv`: Detailed statistics for each transition

## How It Works

The simulation:

1. Loads the MNIST dataset
2. Implements a neural network split into client and server components
3. Trains the network using a simulated split learning approach
4. Measures processing time at the server side for each sample
5. Analyzes timing patterns by label and transitions between labels
6. Generates visualizations and statistics files for further analysis

The network architecture is:
- Input layer (784 neurons) → Hidden layer 1 (128 neurons) [CLIENT SIDE]
- Hidden layer 1 (128 neurons) → Hidden layer 2 (32 neurons) → Output layer (10 neurons) [SERVER SIDE]

## Running the Simulation

To run the simulation:

```bash
python SideChannel.py
```

The first run will download the MNIST dataset, which requires an internet connection. After running, check the `results/` directory for generated files. 