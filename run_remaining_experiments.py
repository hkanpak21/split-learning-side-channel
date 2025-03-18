#!/usr/bin/env python3
import os
import sys
import json
import time
import pandas as pd
import numpy as np
import itertools
import subprocess
from pathlib import Path
import argparse

# Ensure the experiments directory exists
os.makedirs("experiments", exist_ok=True)

def get_remaining_experiment_configs():
    """Define the configurations for the remaining experiments."""
    
    configs = []
    
    # Only the untested model architectures
    model_architectures = [
        [784, 256, 128, 64, 10],  # Deeper network
        [784, 512, 256, 128, 64, 10],  # Much deeper network
        [784, 64, 32, 10],  # Smaller network
    ]
    
    # Keep all other parameters the same
    cut_positions = [1, 2, -2]
    batch_sizes = [16, 32, 64, 128]
    epochs = [3, 5]
    learning_rates = [0.01, 0.001]
    
    # Create all valid combinations
    for arch, batch_size, epoch, lr in itertools.product(
        model_architectures, batch_sizes, epochs, learning_rates
    ):
        for cut_pos in cut_positions:
            # Convert relative positions to absolute indices
            if cut_pos < 0:
                actual_cut_pos = len(arch) + cut_pos - 1
            else:
                actual_cut_pos = cut_pos
                
            # Skip invalid positions (must be between first and last layer)
            if actual_cut_pos <= 0 or actual_cut_pos >= len(arch) - 1:
                continue
                
            # Create a configuration
            config = {
                "model_architecture": arch,
                "cut_layer": actual_cut_pos,
                "batch_size": batch_size,
                "epochs": epoch,
                "learning_rate": lr,
                "timestamp": int(time.time()),
            }
            
            configs.append(config)
    
    return configs

def run_experiment(config, experiment_id, repeat=3):
    """Run an experiment with the given configuration."""
    
    print(f"\n{'-'*80}")
    print(f"Running experiment {experiment_id}")
    print(f"Configuration: {config}")
    print(f"{'-'*80}\n")
    
    # Create directory for this experiment
    experiment_dir = Path(f"experiments/exp_{experiment_id}")
    experiment_dir.mkdir(exist_ok=True)
    
    # Save the configuration
    with open(experiment_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)
    
    # Results for this experiment
    results = []
    
    for run in range(repeat):
        print(f"\nRun {run+1}/{repeat}")
        
        run_dir = experiment_dir / f"run_{run}"
        run_dir.mkdir(exist_ok=True)
        
        sim_dir = run_dir / "simulation"
        sim_dir.mkdir(exist_ok=True)
        
        recovery_dir = run_dir / "recovery"
        recovery_dir.mkdir(exist_ok=True)
        
        # Run the simulation
        print("\nRunning split learning simulation...")
        
        # Prepare command with all configuration parameters
        sim_cmd = [
            "python", "split_learning_simulation/SideChannel.py",
            "--output_dir", str(sim_dir),
            "--architecture", ",".join(map(str, config["model_architecture"])),
            "--cut_layer", str(config["cut_layer"]),
            "--batch_size", str(config["batch_size"]),
            "--epochs", str(config["epochs"]),
            "--learning_rate", str(config["learning_rate"]),
        ]
        
        # Execute simulation
        subprocess.run(sim_cmd, check=True)
        
        # Run the recovery attack
        print("\nRunning recovery attack...")
        
        recovery_cmd = [
            "python", "recovery_attack/recover.py",
            "--timing_data", str(sim_dir / "results/transition_times.csv"),
            "--detailed_stats", str(sim_dir / "results/detailed_transition_stats.csv"),
            "--output_dir", str(recovery_dir),
            "--architecture", ",".join(map(str, config["model_architecture"])),
            "--cut_layer", str(config["cut_layer"]),
        ]
        
        # Execute recovery
        subprocess.run(recovery_cmd, check=True)
        
        # Read recovery results
        try:
            with open(recovery_dir / "results/recovery_results.txt", "r") as f:
                recovery_results = f.read()
                
            # Extract accuracy information
            accuracy_line = [line for line in recovery_results.split("\n") if "Best model accuracy" in line][0]
            accuracy = float(accuracy_line.split(":")[-1].strip())
            
            random_line = [line for line in recovery_results.split("\n") if "Random baseline accuracy" in line][0]
            random_accuracy = float(random_line.split(":")[-1].strip())
            
            improvement_line = [line for line in recovery_results.split("\n") if "Improvement over random" in line][0]
            improvement = float(improvement_line.split(":")[-1].strip().rstrip("%"))
            
            # Extract per-digit success rates
            digit_accuracies = {}
            digit_lines = [line for line in recovery_results.split("\n") if line.startswith("Digit ")]
            for line in digit_lines:
                parts = line.split(":")
                digit = int(parts[0].split(" ")[1])
                digit_acc = float(parts[1].strip())
                digit_accuracies[digit] = digit_acc
            
            # Store results for this run
            run_result = {
                "experiment_id": experiment_id,
                "run": run,
                **config,
                "accuracy": accuracy,
                "random_baseline": random_accuracy,
                "improvement_percent": improvement,
            }
            
            # Add per-digit accuracies
            for digit, acc in digit_accuracies.items():
                run_result[f"digit_{digit}_accuracy"] = acc
                
            results.append(run_result)
            
        except Exception as e:
            print(f"Error processing recovery results: {e}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Run remaining split learning side channel experiments")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of experiments to run")
    parser.add_argument("--repeat", type=int, default=3, help="Number of times to repeat each experiment")
    parser.add_argument("--start_id", type=int, default=80, help="Starting experiment ID (to avoid conflicts with existing experiments)")
    args = parser.parse_args()
    
    # Load existing results if available
    existing_results = []
    if os.path.exists("experiments/results.csv"):
        existing_results = pd.read_csv("experiments/results.csv").to_dict('records')
    
    # Get remaining experiment configurations
    configs = get_remaining_experiment_configs()
    print(f"Generated {len(configs)} remaining experiment configurations")
    
    if args.limit:
        print(f"Limiting to first {args.limit} experiments")
        configs = configs[:args.limit]
    
    # Results storage
    all_results = existing_results.copy()  # Start with existing results
    
    # Run each remaining experiment
    for i, config in enumerate(configs):
        try:
            # Use start_id to avoid conflicts with existing experiment IDs
            results = run_experiment(config, i + args.start_id, repeat=args.repeat)
            all_results.extend(results)
            
            # Save intermediate results
            df = pd.DataFrame(all_results)
            df.to_csv("experiments/results.csv", index=False)
            
        except Exception as e:
            print(f"Error in experiment {i + args.start_id}: {e}")
    
    # Final results processing
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Calculate aggregate statistics across runs
        agg_results = []
        for exp_id in df["experiment_id"].unique():
            exp_df = df[df["experiment_id"] == exp_id]
            
            agg_row = {
                "experiment_id": exp_id,
                **{k: v for k, v in exp_df.iloc[0].items() if k not in ["run", "accuracy", "random_baseline", "improvement_percent"] and not k.startswith("digit_")},
                "mean_accuracy": exp_df["accuracy"].mean(),
                "std_accuracy": exp_df["accuracy"].std(),
                "mean_improvement": exp_df["improvement_percent"].mean(),
                "std_improvement": exp_df["improvement_percent"].std(),
            }
            
            # Average per-digit accuracies
            for digit in range(10):
                digit_col = f"digit_{digit}_accuracy"
                if digit_col in exp_df.columns:
                    agg_row[f"mean_{digit_col}"] = exp_df[digit_col].mean()
                    agg_row[f"std_{digit_col}"] = exp_df[digit_col].std()
            
            agg_results.append(agg_row)
        
        agg_df = pd.DataFrame(agg_results)
        agg_df.to_csv("experiments/aggregated_results.csv", index=False)
        
        print("\nExperiments completed!")
        print(f"Results saved to experiments/results.csv and experiments/aggregated_results.csv")
    else:
        print("No results were collected!")

if __name__ == "__main__":
    main() 