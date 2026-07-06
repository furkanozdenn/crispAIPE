"""
Script to create uncertainty analysis figure similar to the reference figure,
but adapted for pe-uncert with PCA-based uncertainty intervals.

Creates three plots:
A. Box plot: Editing Activity vs PCA-based Uncertainty Intervals
B. Scatter plot: Mismatch Count vs PCA-based Uncertainty
C. Heatmap: PCA-based Uncertainty by Mismatch Type and Position
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from scipy.stats import dirichlet
from tqdm import tqdm
import json

# Add parent directory to path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pe_uncert_models.models.crispAIPE import crispAIPE
from pe_uncert_models.data_utils.data import PE_Dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Generate uncertainty analysis figure for crispAIPE')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./figures', help='Output directory for figures')
    parser.add_argument('--n_samples', type=int, default=5000, help='Number of samples to draw from distributions')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for inference')
    parser.add_argument('--max_examples', type=int, default=10000, help='Maximum number of examples to process')
    return parser.parse_args()


def load_model_and_data(config_path, checkpoint_path):
    """Load the model from checkpoint and prepare the test dataset."""
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    config_model = config['model_parameters']
    config_data = config['data_parameters']
    config_training = config['training_parameters']
    
    # Fix data paths - make them relative to the config file location
    config_dir = os.path.dirname(os.path.abspath(config_path))
    
    # Convert relative paths to absolute paths
    for path_key in ['train_data_path', 'val_data_path', 'test_data_path', 'vocab_path']:
        if path_key in config_data:
            if not os.path.isabs(config_data[path_key]):
                config_data[path_key] = os.path.normpath(
                    os.path.join(config_dir, config_data[path_key])
                )
    
    # Load dataset
    data_module = PE_Dataset(data_config=config_data)
    
    # Load model from checkpoint
    model = crispAIPE.load_from_checkpoint(
        checkpoint_path,
        hparams={**config_model, **config_data, **config_training}
    )
    model.eval()
    
    return model, data_module


def simplex_to_2d(simplex_points):
    """Extract first two dimensions from 3D simplex points (edited %, unedited %)."""
    return simplex_points[:, :2]


def compute_2d_coefficient_of_variation(samples_2d):
    """
    Compute 2D Coefficient of Variation (CV) from 2D samples.
    
    CV in 2D is computed as: CV = std(distance) / mean(distance)
    where distance is the Euclidean distance from each point to the mean.
    
    Returns:
        cv: 2D coefficient of variation
    """
    if len(samples_2d) < 2:
        return 0.0
    
    # Compute mean of the 2D distribution
    mean = np.mean(samples_2d, axis=0)
    
    # Compute Euclidean distance from each point to the mean
    distances = np.sqrt(np.sum((samples_2d - mean) ** 2, axis=1))
    
    # Compute mean distance
    mean_distance = np.mean(distances)
    
    # Avoid division by zero
    if mean_distance < 1e-10:
        return 0.0
    
    # Compute standard deviation of distances
    std_distance = np.std(distances)
    
    # Coefficient of Variation = std / mean
    cv = std_distance / mean_distance
    
    return cv


def extract_sequences_from_batch(batch):
    """Extract initial and mutated sequences from batch."""
    initial_seq = batch[0]  # Shape: (batch_size, seq_len, input_dim)
    mutated_seq = batch[1]  # Shape: (batch_size, seq_len, input_dim)
    
    # Convert one-hot to indices
    initial_indices = torch.argmax(initial_seq, dim=-1).cpu().numpy()
    mutated_indices = torch.argmax(mutated_seq, dim=-1).cpu().numpy()
    
    # Convert indices to sequences (A=0, G=1, C=2, T=3, N=4)
    index_to_nuc = {0: 'A', 1: 'G', 2: 'C', 3: 'T', 4: 'N'}
    
    initial_seqs = []
    mutated_seqs = []
    for i in range(len(initial_indices)):
        initial_seq_str = ''.join([index_to_nuc.get(idx, 'N') for idx in initial_indices[i]])
        mutated_seq_str = ''.join([index_to_nuc.get(idx, 'N') for idx in mutated_indices[i]])
        initial_seqs.append(initial_seq_str)
        mutated_seqs.append(mutated_seq_str)
    
    return initial_seqs, mutated_seqs


def count_mismatches(initial_seq, mutated_seq):
    """Count the number of mismatches between initial and mutated sequences."""
    return sum(1 for i, m in zip(initial_seq, mutated_seq) if i != m)


def extract_mismatch_info(initial_seq, mutated_seq):
    """
    Extract mismatch type and position information.
    
    Returns:
        mismatches: List of tuples (position, initial_nuc, mutated_nuc, mismatch_type)
    """
    mismatches = []
    for pos, (init_nuc, mut_nuc) in enumerate(zip(initial_seq, mutated_seq)):
        if init_nuc != mut_nuc:
            mismatch_type = f"{init_nuc}-{mut_nuc}"
            mismatches.append((pos, init_nuc, mut_nuc, mismatch_type))
    return mismatches


def generate_predictions_and_uncertainty(model, data_module, n_samples=5000, batch_size=64, max_examples=10000):
    """
    Generate predictions, samples, and extract sequence information.
    
    Returns:
        predictions: (N, 3) array of predicted means
        samples: (N, n_samples, 3) array of samples
        ground_truth: (N, 3) array of ground truth proportions
        edited_percentages: (N,) array of edited percentages
        cv_2d_values: (N,) array of 2D Coefficient of Variation values
        mismatch_counts: (N,) array of mismatch counts
        mismatch_info: List of lists of mismatch tuples
    """
    device = next(model.parameters()).device
    test_loader = data_module.test_dataloader()
    
    all_predictions = []
    all_samples = []
    all_ground_truth = []
    all_edited_percentages = []
    all_cv_2d = []
    all_mismatch_counts = []
    all_mismatch_info = []
    
    samples_collected = 0
    
    print("Generating predictions and computing uncertainty...")
    for batch in tqdm(test_loader):
        if samples_collected >= max_examples:
            break
        
        batch = [b.to(device) for b in batch]
        
        with torch.no_grad():
            # Get Dirichlet parameters
            _, alpha_params = model(batch)
            
            # Get ground truth proportions
            _, edited_pct, unedited_pct, indel_pct = batch[2:6]
            ground_truth = torch.stack([edited_pct, unedited_pct, indel_pct], dim=1)
            ground_truth = ground_truth / torch.sum(ground_truth, dim=1, keepdim=True)
            
            # Calculate predicted means
            alpha_sum = torch.sum(alpha_params, dim=1, keepdim=True)
            predictions = alpha_params / alpha_sum
            
            # Sample from Dirichlet distributions
            alpha_np = alpha_params.cpu().numpy()
            samples = np.array([dirichlet.rvs(alpha, size=n_samples) for alpha in alpha_np])
            
            # Extract sequences
            initial_seqs, mutated_seqs = extract_sequences_from_batch(batch)
            
            # Process each sample in the batch
            for i in range(len(alpha_np)):
                # Convert samples to 2D
                samples_2d = simplex_to_2d(samples[i])
                
                # Compute 2D Coefficient of Variation
                cv_2d = compute_2d_coefficient_of_variation(samples_2d)
                
                # Count mismatches
                mismatch_count = count_mismatches(initial_seqs[i], mutated_seqs[i])
                
                # Extract mismatch info
                mismatch_info = extract_mismatch_info(initial_seqs[i], mutated_seqs[i])
                
                all_predictions.append(predictions[i].cpu().numpy())
                all_samples.append(samples[i])
                all_ground_truth.append(ground_truth[i].cpu().numpy())
                all_edited_percentages.append(edited_pct[i].cpu().item())
                all_cv_2d.append(cv_2d)
                all_mismatch_counts.append(mismatch_count)
                all_mismatch_info.append(mismatch_info)
                
                samples_collected += 1
                if samples_collected >= max_examples:
                    break
    
    predictions = np.array(all_predictions)
    samples = np.array(all_samples)
    ground_truth = np.array(all_ground_truth)
    edited_percentages = np.array(all_edited_percentages)
    cv_2d_values = np.array(all_cv_2d) if len(all_cv_2d) > 0 else np.array([])
    mismatch_counts = np.array(all_mismatch_counts)
    
    return (predictions, samples, ground_truth, edited_percentages, 
            cv_2d_values, mismatch_counts, all_mismatch_info)


def plot_editing_activity_vs_uncertainty(edited_percentages, cv_2d_values, output_dir):
    """Plot A: Box plot of Editing Activity vs 2D Coefficient of Variation Intervals."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create CV intervals similar to the reference figure (0-10, 10-20, 20-30, 30-40)
    # Since CV is typically a percentage, we'll multiply by 100 and create intervals
    cv_percent = cv_2d_values * 100
    
    # Create 4 intervals based on percentiles to ensure roughly equal distribution
    percentiles = [0, 25, 50, 75, 100]
    cv_quantiles = np.percentile(cv_percent, percentiles)
    
    # Create intervals
    intervals = []
    interval_labels = []
    for i in range(len(cv_quantiles) - 1):
        if i == len(cv_quantiles) - 2:  # Last interval includes the upper bound
            mask = (cv_percent >= cv_quantiles[i]) & (cv_percent <= cv_quantiles[i+1])
        else:
            mask = (cv_percent >= cv_quantiles[i]) & (cv_percent < cv_quantiles[i+1])
        
        if np.sum(mask) > 0:  # Only add if there are samples in this interval
            intervals.append(edited_percentages[mask])
            # Create labels with rounded values (as percentages)
            interval_labels.append(f'{cv_quantiles[i]:.1f}-{cv_quantiles[i+1]:.1f}')
    
    if len(intervals) == 0:
        print("Warning: No data for box plot")
        return
    
    # Create figure
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create box plot
    bp = ax.boxplot(intervals, labels=interval_labels, patch_artist=True)
    
    # Color the boxes
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
    
    ax.set_xlabel('2D Coefficient of Variation Intervals (%)', fontsize=12)
    ax.set_ylabel('Editing Activity (Edited %)', fontsize=12)
    ax.set_title('Editing Activity vs 2D Coefficient of Variation Intervals', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'uncertainty_analysis_A.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'uncertainty_analysis_A.pdf'))
    plt.close()
    
    print(f"Plot A saved to {output_dir}")


def plot_mismatch_vs_uncertainty(mismatch_counts, cv_2d_values, output_dir):
    """Plot B: Scatter plot of Mismatch Count vs 2D Coefficient of Variation."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Convert CV to percentage for display
    cv_percent = cv_2d_values * 100
    
    # Create figure
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create scatter plot with color coding by mismatch count
    scatter = ax.scatter(cv_percent, mismatch_counts, 
                        c=mismatch_counts, cmap='viridis', 
                        alpha=0.6, s=20, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('2D Coefficient of Variation (%)', fontsize=12)
    ax.set_ylabel('Mismatch Count', fontsize=12)
    ax.set_title('Mismatch Count vs 2D Coefficient of Variation', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Mismatch Count', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'uncertainty_analysis_B.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'uncertainty_analysis_B.pdf'))
    plt.close()
    
    print(f"Plot B saved to {output_dir}")


def plot_uncertainty_heatmap(mismatch_info, cv_2d_values, output_dir):
    """Plot C: Heatmap of 2D Coefficient of Variation by Mismatch Type and Position."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Collect all mismatch types and positions
    mismatch_data = []
    for i, mismatches in enumerate(mismatch_info):
        for pos, init_nuc, mut_nuc, mismatch_type in mismatches:
            mismatch_data.append({
                'position': pos,
                'mismatch_type': mismatch_type,
                'cv_2d': cv_2d_values[i]
            })
    
    if len(mismatch_data) == 0:
        print("Warning: No mismatch data found for heatmap")
        return
    
    df = pd.DataFrame(mismatch_data)
    
    # Get all unique mismatch types and positions
    all_mismatch_types = sorted(df['mismatch_type'].unique())
    max_position = int(df['position'].max())
    
    # Create a matrix for the heatmap
    # Group by mismatch type and position, compute mean uncertainty
    heatmap_data = []
    heatmap_positions = []
    heatmap_types = []
    
    for mismatch_type in all_mismatch_types:
        for pos in range(max_position + 1):
            subset = df[(df['mismatch_type'] == mismatch_type) & (df['position'] == pos)]
            if len(subset) > 0:
                mean_cv = subset['cv_2d'].mean()
                heatmap_data.append(mean_cv)
                heatmap_positions.append(pos)
                heatmap_types.append(mismatch_type)
    
    if len(heatmap_data) == 0:
        print("Warning: No data for heatmap after grouping")
        return
    
    # Create pivot table directly from df (simpler approach)
    pivot_table = df.pivot_table(
        values='cv_2d',
        index='mismatch_type',
        columns='position',
        aggfunc='mean'
    )
    
    # Fill NaN with NaN (will be shown as white/empty in heatmap)
    
    # Create figure
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create heatmap (convert CV to percentage for display)
    pivot_table_percent = pivot_table * 100
    sns.heatmap(pivot_table_percent, annot=False, fmt='.2f', cmap='Reds', 
                cbar_kws={'label': '2D Coefficient of Variation (%)'}, ax=ax,
                linewidths=0.5, linecolor='gray')
    
    ax.set_xlabel('Mismatch Position', fontsize=12)
    ax.set_ylabel('Mismatch Type', fontsize=12)
    ax.set_title('2D Coefficient of Variation by Mismatch Type and Position', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'uncertainty_analysis_C.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'uncertainty_analysis_C.pdf'))
    plt.close()
    
    print(f"Plot C saved to {output_dir}")


def create_combined_figure(edited_percentages, cv_2d_values, mismatch_counts, 
                          mismatch_info, output_dir):
    """Create a combined figure with all three plots (A, B, C)."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create figure with three subplots
    fig = plt.figure(figsize=(18, 6))
    
    # Plot A: Box plot
    ax1 = plt.subplot(1, 3, 1)
    
    cv_percent = cv_2d_values * 100
    cv_quantiles = np.percentile(cv_percent, [0, 25, 50, 75, 100])
    intervals = []
    interval_labels = []
    for i in range(len(cv_quantiles) - 1):
        if i == len(cv_quantiles) - 2:
            mask = (cv_percent >= cv_quantiles[i]) & (cv_percent <= cv_quantiles[i+1])
        else:
            mask = (cv_percent >= cv_quantiles[i]) & (cv_percent < cv_quantiles[i+1])
        
        if np.sum(mask) > 0:
            intervals.append(edited_percentages[mask])
            interval_labels.append(f'{cv_quantiles[i]:.1f}-{cv_quantiles[i+1]:.1f}')
    
    if len(intervals) > 0:
        bp = ax1.boxplot(intervals, labels=interval_labels, patch_artist=True)
        colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
    
    ax1.set_xlabel('2D CV Intervals (%)', fontsize=11)
    ax1.set_ylabel('Editing Activity (Edited %)', fontsize=11)
    ax1.set_title('A', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot B: Scatter plot
    ax2 = plt.subplot(1, 3, 2)
    scatter = ax2.scatter(cv_percent, mismatch_counts, 
                        c=mismatch_counts, cmap='viridis', 
                        alpha=0.6, s=20, edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('2D Coefficient of Variation (%)', fontsize=11)
    ax2.set_ylabel('Mismatch Count', fontsize=11)
    ax2.set_title('B', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='Mismatch Count')
    
    # Plot C: Heatmap
    ax3 = plt.subplot(1, 3, 3)
    
    # Collect mismatch data
    mismatch_data = []
    for i, mismatches in enumerate(mismatch_info):
        for pos, init_nuc, mut_nuc, mismatch_type in mismatches:
            mismatch_data.append({
                'position': pos,
                'mismatch_type': mismatch_type,
                'cv_2d': cv_2d_values[i]
            })
    
    if len(mismatch_data) > 0:
        df = pd.DataFrame(mismatch_data)
        all_mismatch_types = sorted(df['mismatch_type'].unique())
        max_position = int(df['position'].max())
        
        heatmap_data = []
        heatmap_positions = []
        heatmap_types = []
        
        for mismatch_type in all_mismatch_types:
            for pos in range(max_position + 1):
                subset = df[(df['mismatch_type'] == mismatch_type) & (df['position'] == pos)]
                if len(subset) > 0:
                    mean_cv = subset['cv_2d'].mean()
                    heatmap_data.append(mean_cv)
                    heatmap_positions.append(pos)
                    heatmap_types.append(mismatch_type)
        
        if len(heatmap_data) > 0:
            heatmap_df = pd.DataFrame({
                'position': heatmap_positions,
                'mismatch_type': heatmap_types,
                'cv_2d': heatmap_data
            })
            
            pivot_table = heatmap_df.pivot_table(
                values='cv_2d',
                index='mismatch_type',
                columns='position',
                aggfunc='mean'
            )
            
            # Convert CV to percentage for display
            pivot_table_percent = pivot_table * 100
            sns.heatmap(pivot_table_percent, annot=False, fmt='.2f', cmap='Reds', 
                       cbar_kws={'label': '2D CV (%)'}, ax=ax3,
                       linewidths=0.5, linecolor='gray')
    
    ax3.set_xlabel('Mismatch Position', fontsize=11)
    ax3.set_ylabel('Mismatch Type', fontsize=11)
    ax3.set_title('C', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'uncertainty_analysis_combined.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'uncertainty_analysis_combined.pdf'))
    plt.close()
    
    print(f"Combined figure saved to {output_dir}")


def main():
    args = parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("Loading model and data...")
    model, data_module = load_model_and_data(args.config, args.checkpoint)
    
    print("Generating predictions and computing uncertainty...")
    (predictions, samples, ground_truth, edited_percentages, 
     cv_2d_values, mismatch_counts, mismatch_info) = generate_predictions_and_uncertainty(
        model, data_module, n_samples=args.n_samples, 
        batch_size=args.batch_size, max_examples=args.max_examples
    )
    
    print(f"Dataset size: {len(predictions)} samples")
    print(f"2D CV range: {cv_2d_values.min():.4f} - {cv_2d_values.max():.4f} ({cv_2d_values.min()*100:.2f}% - {cv_2d_values.max()*100:.2f}%)")
    print(f"Mismatch count range: {mismatch_counts.min()} - {mismatch_counts.max()}")
    
    print("Creating plots...")
    plot_editing_activity_vs_uncertainty(edited_percentages, cv_2d_values, args.output_dir)
    plot_mismatch_vs_uncertainty(mismatch_counts, cv_2d_values, args.output_dir)
    plot_uncertainty_heatmap(mismatch_info, cv_2d_values, args.output_dir)
    
    print("Creating combined figure...")
    create_combined_figure(edited_percentages, cv_2d_values, mismatch_counts, 
                         mismatch_info, args.output_dir)
    
    print(f"\nDone! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()

