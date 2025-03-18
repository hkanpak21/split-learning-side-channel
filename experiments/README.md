# Split Learning Side Channel Experiments

This directory contains comprehensive experimental results for the side channel attack on split learning.

## Experiment Configuration

The experiments systematically vary several parameters:

1. **Model Architecture**: Different network depths and widths
   - Original: [784, 128, 32, 10]
   - Deeper: [784, 256, 128, 64, 10]
   - Much Deeper: [784, 512, 256, 128, 64, 10]
   - Smaller: [784, 64, 32, 10]

2. **Cut Layer Position**: Different positions for the split between client and server
   - Early layers
   - Middle layers
   - Later layers

3. **Batch Size**: 16, 32, 64, 128

4. **Training Parameters**: 
   - Epochs: 3, 5
   - Learning Rate: 0.01, 0.001

## Experiment Structure

Each experiment is organized as follows:

```
experiments/
├── results.csv                # Raw results from all experiments
├── aggregated_results.csv     # Aggregated statistics across runs
├── exp_0/                     # Experiment folder
│   ├── config.json            # Experiment configuration
│   ├── run_0/                 # First run
│   │   ├── simulation/        # Simulation results
│   │   │   ├── results/       # Timing data and visualizations
│   │   │   └── config.json    # Simulation configuration
│   │   └── recovery/          # Recovery results
│   │       ├── results/       # Recovery analysis and visualizations
│   │       └── config.json    # Recovery configuration
│   ├── run_1/                 # Second run
│   └── run_2/                 # Third run
├── exp_1/                     # Another experiment with different parameters
...
```

## Results Analysis

The `results.csv` file contains detailed results for each run of each experiment, including:

- Experiment configuration (architecture, cut layer, batch size, etc.)
- Recovery accuracy for each digit
- Overall accuracy compared to random baseline
- Improvement percentage over random guessing

The `aggregated_results.csv` file contains statistics aggregated across multiple runs:

- Mean and standard deviation of accuracy
- Mean and standard deviation of per-digit accuracies
- Mean and standard deviation of improvement over random baseline

## Running Experiments

The experiments are managed by the `run_experiments.py` script in the root directory. You can run it with:

```
python run_experiments.py [options]
```

Options:
- `--limit N`: Limit to first N experiment configurations
- `--repeat N`: Number of times to repeat each experiment (default: 3)

## Key Findings

From these experiments, we can observe:
- How different model architectures affect the information leakage
- The impact of cut layer position on privacy
- How batch size and other training parameters influence the side channel
- Which digits are most vulnerable to recovery

The most effective configurations for both attack and defense can be identified from the aggregated results. 