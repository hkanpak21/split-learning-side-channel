import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from collections import defaultdict
from scipy import stats
import os
import argparse
from pathlib import Path
import json

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Split Learning Side Channel Simulation')
    
    # Model architecture
    parser.add_argument('--architecture', type=str, default='784,128,32,10',
                        help='Comma-separated list of layer sizes including input and output (default: 784,128,32,10)')
    
    # Cut layer position
    parser.add_argument('--cut_layer', type=int, default=1,
                        help='Position of the cut layer (0-indexed, default: 1 which means cut after first hidden layer)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Training batch size (default: 64)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs (default: 5)')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate (default: 0.01)')
    
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

# Function to create a fully connected layer
def create_fc_block(in_features, out_features):
    """Create a fully connected block with ReLU activation."""
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.ReLU()
    )

# Dynamic client model that supports arbitrary architectures and cut positions
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

# Dynamic server model that supports arbitrary architectures and cut positions
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
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
    }
    
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Define transformations for MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Initialize models using the specified architecture and cut layer
    client_model = ClientModel(args.architecture, args.cut_layer)
    server_model = ServerModel(args.architecture, args.cut_layer)
    
    # Print model architecture
    print("\nClient model architecture:")
    print(client_model)
    
    print("\nServer model architecture:")
    print(server_model)
    
    # Initialize loss function and optimizers
    criterion = nn.CrossEntropyLoss()
    client_optimizer = optim.SGD(client_model.parameters(), lr=args.learning_rate, momentum=0.9)
    server_optimizer = optim.SGD(server_model.parameters(), lr=args.learning_rate, momentum=0.9)
    
    # Dictionary to store timing information by label
    timing_by_label = defaultdict(list)
    
    # To track consecutive label pairs and their timing differences
    timing_sequence = []
    previous_label = None
    previous_time = None
    
    # Create a dictionary for transition timing differences
    transition_times = {(i, j): [] for i in range(10) for j in range(10)}
    
    print("Starting split learning simulation...")
    print(f"Architecture: {args.architecture}")
    print(f"Cut layer: {args.cut_layer}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    
    # Training loop
    for epoch in range(args.epochs):
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):
            batch_times = {}
            
            # Process each sample in the batch individually to measure precise timing
            for idx in range(len(inputs)):
                input_sample = inputs[idx:idx+1]
                label = int(labels[idx].item())
                
                # Forward pass on client side
                client_outputs = client_model(input_sample)
                
                # Simulate sending activations to server and measure time
                # Record starting time for forward pass
                start_time = time.time()
                
                # Server forward pass
                server_outputs = server_model(client_outputs)
                loss = criterion(server_outputs, labels[idx:idx+1])
                
                # Server backward pass
                loss.backward()
                
                # Get gradients at the cut layer
                cut_layer_gradients = client_outputs.grad
                
                # Record time when gradients are ready to be sent back to client
                end_time = time.time()
                
                # Calculate time difference
                current_time = end_time - start_time
                
                # Store in our dictionary
                timing_by_label[label].append(current_time)
                
                # Store current time and label for sequence analysis
                timing_sequence.append((label, current_time))
                
                # Calculate transition time if we have a previous sample
                if previous_label is not None:
                    transition_times[(previous_label, label)].append(current_time)
                
                # Update previous label and time
                previous_label = label
                previous_time = current_time
            
            # Client backward pass for the whole batch
            client_optimizer.step()
            server_optimizer.step()
            
            # Zero the parameter gradients
            client_optimizer.zero_grad()
            server_optimizer.zero_grad()
            
            running_loss += loss.item()
            
            if i % 10 == 9:  # Print more frequently
                print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss/10:.4f}')
                running_loss = 0.0
    
    print("Training completed")
    
    # Basic timing analysis
    print("\n--- Basic Timing Analysis ---")
    print("\nAverage timing by label:")
    for label in sorted(timing_by_label.keys()):
        avg_time = np.mean(timing_by_label[label])
        std_time = np.std(timing_by_label[label])
        print(f"Label {label}: {avg_time:.8f}s ± {std_time:.8f}s")
    
    # Visualize timing information
    plt.figure(figsize=(12, 8))
    
    # Plot histograms for each label
    for label in sorted(timing_by_label.keys()):
        plt.hist(timing_by_label[label], alpha=0.5, bins=20, label=f'Label {label}')
    
    plt.title('Time Difference Distribution by Label at Cut Layer')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(results_dir / 'timing_histogram.png')
    plt.close()
    
    # Box plot of timing by label
    plt.figure(figsize=(12, 6))
    box_data = [timing_by_label[label] for label in sorted(timing_by_label.keys())]
    plt.boxplot(box_data, labels=[f'Label {label}' for label in sorted(timing_by_label.keys())])
    plt.title('Time Difference Distribution by Label at Cut Layer')
    plt.xlabel('Label')
    plt.ylabel('Time (seconds)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(results_dir / 'timing_boxplot.png')
    plt.close()
    
    # Transition timing analysis - Creating a matrix of average times
    print("\n--- Label Transition Timing Analysis ---")
    transition_matrix = np.zeros((10, 10))
    transition_count_matrix = np.zeros((10, 10))
    p_value_matrix = np.zeros((10, 10))
    
    # Fill transition matrix with average times
    for i in range(10):
        for j in range(10):
            if transition_times[(i, j)]:
                transition_matrix[i, j] = np.mean(transition_times[(i, j)])
                transition_count_matrix[i, j] = len(transition_times[(i, j)])
    
    # Print transition statistics
    print("\nAverage timing for label transitions (from row to column):")
    df_transitions = pd.DataFrame(transition_matrix, 
                                 index=[f'From {i}' for i in range(10)], 
                                 columns=[f'To {j}' for j in range(10)])
    print(df_transitions)
    
    # Save transition matrix to CSV
    df_transitions.to_csv(results_dir / 'transition_times.csv')
    
    # Conduct statistical tests for each transition pair
    for i in range(10):
        for j in range(10):
            for k in range(10):
                if i != k and len(transition_times[(i, j)]) > 5 and len(transition_times[(k, j)]) > 5:
                    # T-test between transitions i->j and k->j
                    t_stat, p_val = stats.ttest_ind(transition_times[(i, j)], transition_times[(k, j)])
                    if p_val < 0.05:
                        print(f"Statistically significant difference between transitions {i}->{j} and {k}->{j}: p={p_val:.4f}")
    
    # Create a heatmap of transition times
    plt.figure(figsize=(12, 10))
    sns.heatmap(transition_matrix, annot=True, fmt=".8f", cmap="viridis",
                xticklabels=[f'To {j}' for j in range(10)],
                yticklabels=[f'From {i}' for i in range(10)])
    plt.title('Average Timing for Label Transitions (seconds)')
    plt.tight_layout()
    plt.savefig(results_dir / 'transition_heatmap.png')
    plt.close()
    
    # Create a heatmap of transition counts
    plt.figure(figsize=(12, 10))
    sns.heatmap(transition_count_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=[f'To {j}' for j in range(10)],
                yticklabels=[f'From {i}' for i in range(10)])
    plt.title('Number of Observed Label Transitions')
    plt.tight_layout()
    plt.savefig(results_dir / 'transition_counts.png')
    plt.close()
    
    # Advanced analysis: Identifying the most distinctive transitions
    print("\n--- Advanced Timing Analysis ---")
    
    # Find pairs with the largest time differences
    largest_diffs = []
    for i in range(10):
        for j in range(10):
            for k in range(10):
                if i != k and transition_matrix[i, j] > 0 and transition_matrix[k, j] > 0:
                    time_diff = abs(transition_matrix[i, j] - transition_matrix[k, j])
                    largest_diffs.append(((i, j, k), time_diff))
    
    largest_diffs.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 largest time differences between transitions:")
    for (i, j, k), diff in largest_diffs[:10]:
        print(f"Transition {i}->{j} vs {k}->{j}: {diff:.8f}s difference")
    
    # Calculate and write detailed statistics to CSV
    all_stats = []
    for i in range(10):
        for j in range(10):
            if transition_times[(i, j)]:
                times = transition_times[(i, j)]
                stats_dict = {
                    'from_label': i,
                    'to_label': j,
                    'count': len(times),
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': np.min(times),
                    'max': np.max(times),
                    'median': np.median(times),
                    '25th_percentile': np.percentile(times, 25),
                    '75th_percentile': np.percentile(times, 75)
                }
                all_stats.append(stats_dict)
    
    stats_df = pd.DataFrame(all_stats)
    stats_df.to_csv(results_dir / 'detailed_transition_stats.csv', index=False)
    print(f"\nDetailed statistics saved to '{results_dir / 'detailed_transition_stats.csv'}'")
    
    # Scaling factor analysis to detect if patterns are consistent
    print("\n--- Scaling Factor Analysis ---")
    # Compare the timing patterns across different epochs
    epoch_timings = defaultdict(lambda: defaultdict(list))
    
    # Recreate epoch-by-epoch timing data
    epoch_size = len(timing_sequence) // args.epochs
    for epoch in range(args.epochs):
        start_idx = epoch * epoch_size
        end_idx = start_idx + epoch_size if epoch < args.epochs - 1 else len(timing_sequence)
        
        for idx in range(start_idx, end_idx - 1):
            label1, time1 = timing_sequence[idx]
            label2, time2 = timing_sequence[idx + 1]
            epoch_timings[epoch][(label1, label2)].append(time2)
    
    # Calculate correlation between epochs
    for epoch1 in range(args.epochs):
        for epoch2 in range(epoch1 + 1, args.epochs):
            common_transitions = []
            epoch1_times = []
            epoch2_times = []
            
            for transition in set(epoch_timings[epoch1].keys()) & set(epoch_timings[epoch2].keys()):
                if len(epoch_timings[epoch1][transition]) > 0 and len(epoch_timings[epoch2][transition]) > 0:
                    common_transitions.append(transition)
                    epoch1_times.append(np.mean(epoch_timings[epoch1][transition]))
                    epoch2_times.append(np.mean(epoch_timings[epoch2][transition]))
            
            if common_transitions:
                correlation, p_value = stats.pearsonr(epoch1_times, epoch2_times)
                print(f"Correlation between epoch {epoch1+1} and {epoch2+1}: {correlation:.4f} (p={p_value:.4f})")
    
    print(f"\nAnalysis complete. Check the generated files in '{results_dir}' for detailed results.")

if __name__ == "__main__":
    main() 