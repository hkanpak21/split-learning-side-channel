# Side Channel Recovery Attack

This directory contains code for implementing a timing-based side channel attack on split learning. The attack uses the timing information collected during the split learning simulation to recover private client labels.

## Contents

- `recover.py`: Main attack script that implements label recovery using timing side channel information
- `results/`: Directory containing output files from the recovery analysis:
  - `recovery_confusion_matrix.png`: Confusion matrix showing true vs. predicted labels
  - `recovery_confidence.png`: Box plot showing prediction confidence by digit
  - `recovery_success_by_digit.png`: Bar chart showing recovery success rate by digit
  - `model_comparison.png`: Comparison of simple vs. detailed timing models
  - `recovery_results.txt`: Text file summarizing attack results

## How It Works

The recovery attack:

1. Loads timing data from the split learning simulation
2. Creates probabilistic models of timing patterns for each digit transition
3. Simulates observing timing information during client training
4. Uses timing models to infer the most likely label for each sample
5. Evaluates attack success by comparing predicted labels to true labels
6. Analyzes the effectiveness of the attack compared to random guessing

Two different recovery methods are implemented:
- A simple distance-based approach
- A more sophisticated statistical model using Gaussian distributions 

## Running the Attack

To run the recovery attack:

```bash
python recover.py
```

The script assumes that the timing data files from the split learning simulation are available. Make sure to run the simulation in the `split_learning_simulation` directory first.

## Security Implications

The success of this attack demonstrates that timing patterns in split learning can leak information about private labels. This highlights the need for countermeasures such as:

1. Adding random delays to normalize processing time
2. Batch padding to ensure uniform computation time
3. Using differential privacy approaches to mask timing patterns

The attack effectiveness (measured by improvement over random guessing) provides a quantitative assessment of how much information is leaked through the timing side channel. 