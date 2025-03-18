import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
from collections import defaultdict
from scipy.stats import norm
import os
import argparse
from pathlib import Path
import json
import sys

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Label Recovery Using Timing Side Channel Analysis')
    
    # Input data
    parser.add_argument('--timing_data', type=str, default='../split_learning_simulation/results/transition_times.csv',
                        help='Path to timing data CSV (default: ../split_learning_simulation/results/transition_times.csv)')
    parser.add_argument('--detailed_stats', type=str, default='../split_learning_simulation/results/detailed_transition_stats.csv',
                        help='Path to detailed statistics CSV (default: ../split_learning_simulation/results/detailed_transition_stats.csv)')
    
    # Model architecture (must match simulation)
    parser.add_argument('--architecture', type=str, default='784,128,32,10',
                        help='Comma-separated list of layer sizes including input and output (default: 784,128,32,10)')
    parser.add_argument('--cut_layer', type=int, default=1,
                        help='Position of the cut layer (0-indexed, default: 1)')
    
    # Recovery parameters
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples to use for recovery testing (default: 1000)')
    
    # Output directory
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results (default: results)')
    
    args = parser.parse_args()
    
    # Parse architecture string to list of integers
    args.architecture = [int(x) for x in args.architecture.split(',')]
    
    return args

# Custom flattening module
class Flatten(nn.Module):
    def __init__(self, in_features):
        super(Flatten, self).__init__()
        self.in_features = in_features
        
    def forward(self, x):
        return x.view(-1, self.in_features)

# Function to create a fully connected layer (same as in SideChannel.py)
def create_fc_block(in_features, out_features):
    """Create a fully connected block with ReLU activation."""
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.ReLU()
    )

# Dynamic client model (matching the SideChannel.py implementation)
class ClientModel(nn.Module):
    def __init__(self, architecture, cut_layer):
        super(ClientModel, self).__init__()
        
        # Create layers up to the cut point
        layers = []
        for i in range(cut_layer):
            if i == 0:
                # First layer flattens the input and applies linear + ReLU
                layers.append(Flatten(architecture[0]))
                layers.append(create_fc_block(architecture[i], architecture[i+1]))
            else:
                # Subsequent layers
                layers.append(create_fc_block(architecture[i], architecture[i+1]))
        
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Dynamic server model (matching the SideChannel.py implementation)
class ServerModel(nn.Module):
    def __init__(self, architecture, cut_layer):
        super(ServerModel, self).__init__()
        
        # Create layers after the cut point
        layers = []
        for i in range(cut_layer, len(architecture) - 2):
            layers.append(create_fc_block(architecture[i], architecture[i+1]))
        
        # Add the final layer without ReLU
        layers.append(nn.Linear(architecture[-2], architecture[-1]))
        
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Create base timing model from the data
def create_timing_distribution_model(transition_times, detailed_stats=None):
    # Convert dataframe to numpy array for easier indexing
    timing_matrix = transition_times.values
    
    # Create probabilistic models for each label
    timing_models = {}
    
    if detailed_stats is not None:
        # Use more detailed statistics if available
        for i in range(10):  # 'to' labels
            label_stats = detailed_stats[detailed_stats['to_label'] == i]
            if not label_stats.empty:
                means = {}
                stds = {}
                for _, row in label_stats.iterrows():
                    from_label = int(row['from_label'])
                    means[from_label] = row['mean']
                    stds[from_label] = max(row['std'], 1e-6)  # Avoid zero std
                timing_models[i] = {'means': means, 'stds': stds}
    else:
        # Use basic transition matrix timing data
        for i in range(10):  # For each possible label
            # Get timing statistics for transitions to this label
            label_timings = {}
            for j in range(10):  # From each previous label
                if timing_matrix[j, i] > 0:  # If we have data for this transition
                    label_timings[j] = timing_matrix[j, i]
            
            if label_timings:
                timing_models[i] = label_timings
    
    return timing_models

# Function to predict label based on timing and previous label
def predict_label(timing, prev_label, timing_models, detailed=False):
    probabilities = np.zeros(10)
    
    if detailed and 'means' in timing_models.get(0, {}):
        # Use detailed statistical model (assumes Gaussian distribution)
        for label in range(10):
            if label in timing_models and prev_label in timing_models[label]['means']:
                mean = timing_models[label]['means'][prev_label]
                std = timing_models[label]['stds'][prev_label]
                # Calculate probability using Gaussian PDF
                probabilities[label] = norm.pdf(timing, mean, std)
    else:
        # Simple distance-based approach
        for label in range(10):
            if label in timing_models and prev_label in timing_models[label]:
                expected_timing = timing_models[label][prev_label]
                # Inverted distance (closer timing = higher probability)
                probabilities[label] = 1.0 / (abs(timing - expected_timing) + 1e-6)
    
    # Normalize probabilities
    if np.sum(probabilities) > 0:
        probabilities = probabilities / np.sum(probabilities)
    else:
        # Fallback to uniform distribution if no data
        probabilities = np.ones(10) / 10
        
    return probabilities

# Simulate the server observing client's training and collecting timing information
def simulate_observation(client_model, server_model, test_loader, num_samples=1000):
    print(f"\nSimulating observation of {num_samples} samples...")
    
    observed_timings = []
    true_labels = []
    prev_label = np.random.randint(0, 10)  # Start with a random label
    
    # Use test dataset iterator
    dataiter = iter(test_loader)
    
    for i in range(num_samples):
        try:
            inputs, labels = next(dataiter)
        except StopIteration:
            dataiter = iter(test_loader)
            inputs, labels = next(dataiter)
            
        true_label = labels.item()
        
        # Client forward pass
        with torch.no_grad():  # No need to track gradients for this simulation
            client_outputs = client_model(inputs)
        
        # Measure server processing time
        start_time = time.time()
        
        # Server processing
        with torch.no_grad():
            server_outputs = server_model(client_outputs)
        
        # Record time
        end_time = time.time()
        observed_time = end_time - start_time
        
        # Save the timing and label
        observed_timings.append(observed_time)
        true_labels.append(true_label)
        
        # Update previous label
        prev_label = true_label
        
        if (i+1) % 100 == 0:
            print(f"Observed {i+1} samples")
    
    return observed_timings, true_labels

# Function to recover labels from observed timings
def recover_labels(observed_timings, timing_models, use_detailed=True):
    print("\nRecovering labels from timing information...")
    
    recovered_labels = []
    label_probabilities = []
    
    # Initial label is unknown, start with a random guess
    prev_label = np.random.randint(0, 10)
    
    for i, timing in enumerate(observed_timings):
        # Calculate probability distribution for this timing
        probs = predict_label(timing, prev_label, timing_models, detailed=use_detailed)
        
        # Make a prediction (most likely label)
        pred_label = np.argmax(probs)
        
        recovered_labels.append(pred_label)
        label_probabilities.append(probs)
        
        # Update previous label (using our prediction)
        prev_label = pred_label
        
        if (i+1) % 100 == 0:
            print(f"Processed {i+1} timings")
    
    return recovered_labels, label_probabilities

# Function to analyze recovery success
def analyze_recovery(true_labels, recovered_labels, label_probabilities, output_dir):
    print("\n--- Recovery Analysis ---")
    
    # Calculate accuracy
    accuracy = accuracy_score(true_labels, recovered_labels)
    print(f"Overall accuracy: {accuracy:.4f}")
    
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, recovered_labels)
    
    # Classification report
    print("\nClassification Report:")
    report = classification_report(true_labels, recovered_labels)
    print(report)
    
    # Visualize confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10),
                yticklabels=range(10))
    plt.title('Confusion Matrix: True Labels vs Recovered Labels')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(output_dir / 'recovery_confusion_matrix.png')
    plt.close()
    
    # Analyze prediction confidence
    confidence_by_digit = defaultdict(list)
    for i, (true, pred, probs) in enumerate(zip(true_labels, recovered_labels, label_probabilities)):
        confidence = probs[pred]  # Probability of the predicted class
        confidence_by_digit[true].append(confidence)
    
    # Visualize confidence by digit
    plt.figure(figsize=(12, 6))
    box_data = [confidence_by_digit[d] for d in range(10)]
    plt.boxplot(box_data, labels=range(10))
    plt.title('Prediction Confidence by True Digit')
    plt.xlabel('True Digit')
    plt.ylabel('Confidence')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(output_dir / 'recovery_confidence.png')
    plt.close()
    
    # Calculate success rate by digit
    success_by_digit = {}
    for digit in range(10):
        digit_indices = [i for i, x in enumerate(true_labels) if x == digit]
        if digit_indices:
            correct = sum(1 for i in digit_indices if recovered_labels[i] == true_labels[i])
            success_by_digit[digit] = correct / len(digit_indices)
        else:
            success_by_digit[digit] = 0
    
    print("\nRecovery success rate by digit:")
    for digit, rate in success_by_digit.items():
        print(f"Digit {digit}: {rate:.4f}")
    
    # Visualize success by digit
    plt.figure(figsize=(10, 6))
    plt.bar(range(10), [success_by_digit[d] for d in range(10)])
    plt.title('Recovery Success Rate by Digit')
    plt.xlabel('Digit')
    plt.ylabel('Success Rate')
    plt.xticks(range(10))
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(output_dir / 'recovery_success_by_digit.png')
    plt.close()
    
    return accuracy, success_by_digit, report

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    results_dir = output_dir / "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save configuration to file
    config = {
        "architecture": args.architecture,
        "cut_layer": args.cut_layer,
        "num_samples": args.num_samples,
        "timing_data": args.timing_data,
        "detailed_stats": args.detailed_stats
    }
    
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("Label Recovery Using Timing Side Channel Analysis")
    print("================================================")
    print(f"Architecture: {args.architecture}")
    print(f"Cut layer: {args.cut_layer}")
    
    # Load timing data collected from SideChannel.py
    try:
        transition_times = pd.read_csv(args.timing_data, index_col=0)
        print("Loaded transition timing data")
    except Exception as e:
        print(f"Error loading timing data: {e}")
        sys.exit(1)
    
    # Load detailed statistics if available
    detailed_stats = None
    try:
        detailed_stats = pd.read_csv(args.detailed_stats)
        has_detailed_stats = True
        print("Loaded detailed transition statistics")
    except Exception as e:
        has_detailed_stats = False
        print(f"Detailed statistics not found or error loading: {e}")
        print("Using basic timing model")
    
    # Define transformations for MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load MNIST test dataset (to simulate client's private data)
    test_dataset = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    # Initialize models using the specified architecture and cut layer
    client_model = ClientModel(args.architecture, args.cut_layer)
    server_model = ServerModel(args.architecture, args.cut_layer)
    
    # Print model architecture
    print("\nClient model architecture:")
    print(client_model)
    
    print("\nServer model architecture:")
    print(server_model)
    
    # Create timing distribution model from collected data
    timing_models = create_timing_distribution_model(transition_times, detailed_stats)
    print(f"Created timing model for {len(timing_models)} labels")
    
    # Simulate observation of client's training
    observed_timings, true_labels = simulate_observation(
        client_model, server_model, test_loader, num_samples=args.num_samples
    )
    
    # Try to recover labels using both simple and detailed models
    print("\n--- Using Simple Timing Model ---")
    recovered_labels_simple, label_probs_simple = recover_labels(
        observed_timings, timing_models, use_detailed=False
    )
    accuracy_simple, success_simple, report_simple = analyze_recovery(
        true_labels, recovered_labels_simple, label_probs_simple, results_dir
    )
    
    if has_detailed_stats:
        print("\n--- Using Detailed Statistical Model ---")
        recovered_labels_detailed, label_probs_detailed = recover_labels(
            observed_timings, timing_models, use_detailed=True
        )
        accuracy_detailed, success_detailed, report_detailed = analyze_recovery(
            true_labels, recovered_labels_detailed, label_probs_detailed, results_dir
        )
        
        # Compare models
        print("\n--- Model Comparison ---")
        print(f"Simple model accuracy: {accuracy_simple:.4f}")
        print(f"Detailed model accuracy: {accuracy_detailed:.4f}")
        
        # Create comparison visualization
        plt.figure(figsize=(12, 6))
        width = 0.35
        x = np.arange(10)
        plt.bar(x - width/2, [success_simple[d] for d in range(10)], width, label='Simple Model')
        plt.bar(x + width/2, [success_detailed[d] for d in range(10)], width, label='Detailed Model')
        plt.title('Recovery Success Rate Comparison')
        plt.xlabel('Digit')
        plt.ylabel('Success Rate')
        plt.xticks(range(10))
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(results_dir / 'model_comparison.png')
        plt.close()
    
    # For comparison, create a baseline random classifier
    random_labels = np.random.randint(0, 10, size=len(true_labels))
    random_accuracy = accuracy_score(true_labels, random_labels)
    print(f"\nRandom baseline accuracy: {random_accuracy:.4f}")
    
    # Summary
    if has_detailed_stats:
        best_accuracy = max(accuracy_simple, accuracy_detailed)
        improvement = (best_accuracy - random_accuracy) / (1 - random_accuracy) * 100
    else:
        best_accuracy = accuracy_simple
        improvement = (best_accuracy - random_accuracy) / (1 - random_accuracy) * 100
    
    print("\n--- Summary ---")
    print(f"Best model accuracy: {best_accuracy:.4f}")
    print(f"Random baseline accuracy: {random_accuracy:.4f}")
    print(f"Improvement over random: {improvement:.2f}%")
    
    # Save results to a file
    with open(results_dir / 'recovery_results.txt', 'w') as f:
        f.write("Label Recovery Results Using Timing Side Channels\n")
        f.write("===============================================\n\n")
        f.write(f"Model architecture: {args.architecture}\n")
        f.write(f"Cut layer: {args.cut_layer}\n\n")
        f.write(f"Best model accuracy: {best_accuracy:.4f}\n")
        f.write(f"Random baseline accuracy: {random_accuracy:.4f}\n")
        f.write(f"Improvement over random: {improvement:.2f}%\n\n")
        f.write("Success rate by digit:\n")
        for digit in range(10):
            if has_detailed_stats:
                best_rate = max(success_simple[digit], success_detailed[digit])
                f.write(f"Digit {digit}: {best_rate:.4f}\n")
            else:
                f.write(f"Digit {digit}: {success_simple[digit]:.4f}\n")
        
        f.write("\n\nClassification Report (Simple Model):\n")
        f.write(report_simple)
        
        if has_detailed_stats:
            f.write("\n\nClassification Report (Detailed Model):\n")
            f.write(report_detailed)
    
    print(f"\nAnalysis complete. Results saved to {results_dir / 'recovery_results.txt'}")

if __name__ == "__main__":
    main() 