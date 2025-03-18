#!/usr/bin/env python3
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob

def load_config(exp_dir):
    """Load experiment configuration from config.json"""
    config_path = os.path.join(exp_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return None

def process_experiment(exp_path):
    """Process a single experiment directory"""
    # Get experiment ID from path
    path_parts = Path(exp_path).parts
    exp_dir_name = [p for p in path_parts if p.startswith('exp_')][0]
    exp_id = int(exp_dir_name.replace('exp_', ''))
    
    # Load configuration
    exp_dir = os.path.join('experiments', exp_dir_name)
    config = load_config(exp_dir)
    
    if config is None:
        print(f"Warning: No config found for experiment {exp_id}")
        return None
    
    # Read transition times
    try:
        df = pd.read_csv(exp_path)
        
        # Convert wide format to long format
        df_long = pd.melt(df, 
                         value_vars=[f'To {i}' for i in range(10)],
                         var_name='to_label',
                         value_name='transition_time')
        
        # Extract the target label (removing "To " prefix)
        df_long['label'] = df_long['to_label'].str.extract('To (\d+)').astype(int)
        
        # Add experiment metadata
        df_long['experiment_id'] = exp_id
        df_long['architecture'] = '_'.join(map(str, config['model_architecture']))
        df_long['cut_layer'] = config['cut_layer']
        df_long['batch_size'] = config['batch_size']
        df_long['epochs'] = config['epochs']
        df_long['learning_rate'] = config['learning_rate']
        
        return df_long
    except Exception as e:
        print(f"Error processing {exp_path}: {e}")
        return None

def merge_results(output_dir='merged_results'):
    """Merge all experiment results into comprehensive CSV files"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all transition_times.csv files
    pattern = 'experiments/exp_*/run_*/simulation/results/transition_times.csv'
    result_files = glob(pattern)
    
    if not result_files:
        print("No result files found!")
        return
    
    print(f"Found {len(result_files)} result files")
    
    # Process each experiment
    dfs = []
    for result_file in result_files:
        df = process_experiment(result_file)
        if df is not None:
            dfs.append(df)
    
    if not dfs:
        print("No valid results to merge!")
        return
    
    # Merge all results
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Save complete dataset
    merged_df.to_csv(os.path.join(output_dir, 'all_results.csv'), index=False)
    print(f"Saved complete results to {output_dir}/all_results.csv")
    
    # Create summary statistics
    summary_stats = []
    
    # Group by experiment configuration
    grouped = merged_df.groupby(['experiment_id', 'architecture', 'cut_layer', 'batch_size', 'epochs', 'learning_rate'])
    
    for name, group in grouped:
        exp_id, arch, cut_layer, batch_size, epochs, lr = name
        
        # Calculate statistics
        stats = {
            'experiment_id': exp_id,
            'architecture': arch,
            'cut_layer': cut_layer,
            'batch_size': batch_size,
            'epochs': epochs,
            'learning_rate': lr,
            'mean_transition_time': group['transition_time'].mean(),
            'std_transition_time': group['transition_time'].std(),
            'min_transition_time': group['transition_time'].min(),
            'max_transition_time': group['transition_time'].max(),
            'total_transitions': len(group),
            'unique_labels': group['label'].nunique()
        }
        
        # Add per-label statistics
        label_stats = group.groupby('label')['transition_time'].agg(['mean', 'std', 'count']).round(4)
        for label, row in label_stats.iterrows():
            stats[f'label_{label}_mean'] = row['mean']
            stats[f'label_{label}_std'] = row['std']
            stats[f'label_{label}_count'] = row['count']
        
        summary_stats.append(stats)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_stats)
    
    # Save summary statistics
    summary_df.to_csv(os.path.join(output_dir, 'summary_statistics.csv'), index=False)
    print(f"Saved summary statistics to {output_dir}/summary_statistics.csv")
    
    # Print basic statistics
    print("\nBasic Statistics:")
    print(f"Total number of experiments: {len(summary_df)}")
    print(f"Total number of transitions: {merged_df.shape[0]}")
    print("\nUnique configurations:")
    print(f"Architectures: {summary_df['architecture'].nunique()}")
    print(f"Cut layers: {summary_df['cut_layer'].nunique()}")
    print(f"Batch sizes: {summary_df['batch_size'].nunique()}")
    print(f"Learning rates: {summary_df['learning_rate'].nunique()}")
    
    return merged_df, summary_df

if __name__ == '__main__':
    merge_results() 