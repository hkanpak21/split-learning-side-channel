# Split Learning Side Channel Analysis

This repository contains the implementation and analysis of timing-based side channel experiments in Split Learning architectures. The project investigates potential security implications of timing patterns in split neural network configurations.

## Project Structure

```
.
├── SideChannel.py          # Main implementation of split learning network
├── recover.py             # Script for label recovery based on timing analysis
├── run_experiments.py     # Script to run multiple experiments with different configurations
├── merge_results.py       # Script to merge and analyze results from multiple experiments
├── experiments/          # Directory containing individual experiment results
├── merged_results/       # Directory containing aggregated results and statistics
└── docs/                # Documentation including LaTeX report
```

## Key Features

- Implementation of split learning with MNIST dataset
- Timing analysis of network transitions
- Multiple architecture configurations:
  - Basic (784-128-32-10)
  - Deeper (784-256-128-64-10)
- Configurable parameters:
  - Cut layer positions
  - Batch sizes (16, 32, 64, 128)
  - Learning rates (0.01, 0.001)
  - Number of epochs

## Results

The experiments revealed several interesting patterns in the timing behavior of split learning:

- Architectural differences create distinguishable timing patterns
- Cut layer position significantly affects transition times
- Batch size variations lead to consistent timing differences
- Detailed results are available in the LaTeX report and summary statistics

## Usage

1. Run experiments:
```bash
python run_experiments.py
```

2. Merge results:
```bash
python merge_results.py
```

3. Generate report:
```bash
pdflatex experiment_report.tex
```

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Pandas
- Matplotlib
- LaTeX (for report generation)

## Documentation

For detailed analysis and findings, refer to `experiment_report.pdf` in the docs directory. 