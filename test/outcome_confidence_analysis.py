"""
Confidence region and uncertainty analysis figure with novel analyses.
Creates a multi-row, multi-panel figure showing:
- Row 1: Confidence region size distribution, Region size vs prediction accuracy
- Row 2: Prediction error distribution by uncertainty quartiles, Error vs uncertainty scatter
- Row 3: Uncertainty vs sequence features, Coverage by outcome type

example cmd:
python test/outcome_confidence_analysis.py --config pe_uncert_models/configs/crispAIPE_train_test_split_conf.json --checkpoint pe_uncert_models/logs/crispAIPE_train_test_split_conf/2025-06-08-15-59-36/best_model-epoch=41-val_loss_val_loss=-3.0687.ckpt --output_dir test/figures --n_samples 5000 --batch_size 64 --max_examples 5000
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from scipy.stats import dirichlet, chi2
from sklearn.decomposition import PCA
from tqdm import tqdm
import json

# Add parent directory to path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pe_uncert_models.models.crispAIPE import crispAIPE
from pe_uncert_models.data_utils.data import PE_Dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Generate hybrid outcome and confidence region analysis figure')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./figures', help='Output directory for figures')
    parser.add_argument('--n_samples', type=int, default=5000, help='Number of samples to draw from distributions')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for inference')
    parser.add_argument('--max_examples', type=int, default=5000, help='Maximum number of examples to process')
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


def compute_pca_confidence_region(samples_2d, confidence_level=0.95):
    """
    Compute PCA-based confidence region around the mean of 2D samples.
    
    Returns:
        mean: center of confidence region
        principal_axes: eigenvectors (columns are principal components)
        eigenvalues: eigenvalues of covariance matrix
        confidence_scale: scaling factor for confidence ellipse
        area: area of the confidence ellipse
    """
    if len(samples_2d) < 2:
        return None, None, None, None, 0.0
    
    # Center the data
    mean = np.mean(samples_2d, axis=0)
    centered_samples = samples_2d - mean
    
    # Perform PCA
    pca = PCA(n_components=2)
    pca.fit(centered_samples)
    
    # Get principal components and explained variance
    principal_axes = pca.components_.T  # columns are principal components
    eigenvalues = pca.explained_variance_
    
    # Calculate confidence scale based on chi-square distribution
    confidence_scale = np.sqrt(chi2.ppf(confidence_level, df=2))
    
    # Calculate area of confidence ellipse
    area = np.pi * np.sqrt(eigenvalues[0]) * np.sqrt(eigenvalues[1]) * confidence_scale
    
    return mean, principal_axes, eigenvalues, confidence_scale, area


def point_in_confidence_region(point, mean, principal_axes, eigenvalues, confidence_scale):
    """Check if a point is within the PCA-based confidence region."""
    if mean is None:
        return False
    
    # Transform point to PCA space
    centered_point = point - mean
    pca_coords = principal_axes.T @ centered_point
    
    # Check if point is within confidence ellipse
    ellipse_coords = pca_coords / np.sqrt(eigenvalues)
    distance_squared = np.sum(ellipse_coords**2)
    
    return distance_squared <= confidence_scale**2


def extract_pegRNA_features(batch):
    """
    Extract pegRNA features from batch.
    
    Returns:
        rtt_lengths: (batch_size,) array of RTT lengths
        pbs_lengths: (batch_size,) array of PBS lengths
        edit_positions: (batch_size,) array of edit positions relative to nick site
        edit_types: List of edit type strings (e.g., 'A-G', 'G-T')
        edit_categories: List of edit categories ('Substitution', 'Insertion', 'Deletion')
        nick_sites: (batch_size,) array of nick site positions
    """
    # Extract location masks
    protospacer_location = batch[6].cpu().numpy()  # Shape: (batch_size, seq_len)
    pbs_location = batch[7].cpu().numpy()  # Shape: (batch_size, seq_len)
    rt_mutated_location = batch[9].cpu().numpy()  # Shape: (batch_size, seq_len)
    
    # Extract sequences
    initial_seq = batch[0].cpu().numpy()  # Shape: (batch_size, seq_len, input_dim)
    mutated_seq = batch[1].cpu().numpy()
    
    # Convert one-hot to indices
    initial_indices = np.argmax(initial_seq, axis=-1)
    mutated_indices = np.argmax(mutated_seq, axis=-1)
    
    # Convert indices to nucleotides
    index_to_nuc = {0: 'A', 1: 'G', 2: 'C', 3: 'T', 4: 'N'}
    
    rtt_lengths = []
    pbs_lengths = []
    edit_positions = []
    edit_types = []
    edit_categories = []
    nick_sites = []
    
    for i in range(len(initial_indices)):
        # Compute PBS length (sum of PBS location mask)
        pbs_length = np.sum(pbs_location[i])
        pbs_lengths.append(pbs_length)
        
        # Compute RTT length (sum of RT mutated location mask)
        rtt_length = np.sum(rt_mutated_location[i])
        rtt_lengths.append(rtt_length)
        
        # Find RTT region (RT mutated location)
        rtt_mask = rt_mutated_location[i] > 0.5
        if np.any(rtt_mask):
            rtt_positions = np.where(rtt_mask)[0]
            rtt_start = rtt_positions[0] if len(rtt_positions) > 0 else -1
            rtt_end = rtt_positions[-1] if len(rtt_positions) > 0 else -1
        else:
            rtt_start = -1
            rtt_end = -1
        
        # Find nick site (end of protospacer, typically the 3' end) - still needed for other analyses
        protospacer_mask = protospacer_location[i] > 0.5
        if np.any(protospacer_mask):
            nick_site = np.where(protospacer_mask)[0]
            if len(nick_site) > 0:
                nick_site_pos = nick_site[-1]  # Last position (3' end)
            else:
                nick_site_pos = -1
        else:
            nick_site_pos = -1
        nick_sites.append(nick_site_pos)
        
        # Find ALL edit positions and classify (not just the first one)
        all_edit_positions_for_seq = []
        all_edit_types_for_seq = []
        all_edit_categories_for_seq = []
        
        # Find all mismatches in the sequence
        for pos in range(len(initial_indices[i])):
            init_nuc = index_to_nuc.get(initial_indices[i, pos], 'N')
            mut_nuc = index_to_nuc.get(mutated_indices[i, pos], 'N')
            
            if init_nuc != mut_nuc:
                edit_pos = pos
                edit_type = None
                edit_category = None
                
                # Only process substitutions (transitions and transversions)
                # Ignore insertions and deletions
                if init_nuc != 'N' and mut_nuc != 'N':
                    # Classify as transition or transversion
                    # Transitions: A↔G, C↔T
                    # Transversions: A↔C, A↔T, G↔C, G↔T
                    transition_pairs = {('A', 'G'), ('G', 'A'), ('C', 'T'), ('T', 'C')}
                    
                    if (init_nuc, mut_nuc) in transition_pairs:
                        edit_category = 'Transition'
                        edit_type = f"{init_nuc}-{mut_nuc}"
                    else:
                        edit_category = 'Transversion'
                        edit_type = f"{init_nuc}-{mut_nuc}"
                    
                    if rtt_start >= 0 and rtt_end >= 0:
                        # Only include edits within RTT region
                        if rtt_start <= edit_pos <= rtt_end:
                            # Calculate position relative to RTT start (0-based within RTT)
                            rtt_relative_pos = edit_pos - rtt_start
                            all_edit_positions_for_seq.append(rtt_relative_pos)
                            all_edit_types_for_seq.append(edit_type)
                            all_edit_categories_for_seq.append(edit_category)
        
        # Store all edits for this sequence (we'll expand the lists later)
        if len(all_edit_positions_for_seq) > 0:
            edit_positions.append(all_edit_positions_for_seq)
            edit_types.append(all_edit_types_for_seq)
            edit_categories.append(all_edit_categories_for_seq)
        else:
            # No edits found within RTT
            edit_positions.append([-999])
            edit_types.append(['N-N'])
            edit_categories.append(['Unknown'])
    
    return (np.array(rtt_lengths), np.array(pbs_lengths), edit_positions, 
            edit_types, edit_categories, np.array(nick_sites))


def generate_predictions_and_data(model, data_module, n_samples=5000, batch_size=64, max_examples=None):
    """
    Generate predictions, samples, and compute all necessary metrics.
    
    Returns:
        predictions: (N, 3) array of predicted means
        ground_truth: (N, 3) array of ground truth proportions
        alpha_0_values: (N,) array of Total Concentration (α₀) values
        region_areas: (N,) array of 95% confidence region areas
        samples: (N, n_samples, 3) array of samples
        rtt_lengths: (N,) array of RTT lengths
        pbs_lengths: (N,) array of PBS lengths
        edit_positions: (N,) array of edit positions
        edit_types: List of edit type strings
    """
    device = next(model.parameters()).device
    test_loader = data_module.test_dataloader()
    
    all_predictions = []
    all_ground_truth = []
    all_alpha_0 = []
    all_region_areas = []
    all_samples = []
    all_rtt_lengths = []
    all_pbs_lengths = []
    all_edit_positions = []
    all_edit_types = []
    all_edit_categories = []
    all_nick_sites = []
    
    samples_collected = 0
    
    print("Generating predictions and computing metrics...")
    for batch in tqdm(test_loader):
        if max_examples is not None and samples_collected >= max_examples:
            break
        
        batch = [b.to(device) for b in batch]
        
        # Extract pegRNA features
        rtt_lengths, pbs_lengths, edit_positions, edit_types, edit_categories, nick_sites = extract_pegRNA_features(batch)
        
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
            
            # Process each sample in the batch
            for i in range(len(alpha_np)):
                # Convert samples to 2D
                samples_2d = simplex_to_2d(samples[i])
                
                # Compute Total Concentration α₀ (sum of alpha parameters)
                alpha_0 = alpha_sum[i, 0].cpu().item()
                
                # Compute confidence region area
                _, _, _, _, area = compute_pca_confidence_region(samples_2d, confidence_level=0.95)
                
                # Expand multiple edits per sequence into separate entries
                # Each edit gets its own row with the same prediction/alpha_0
                n_edits = len(edit_positions[i])
                for edit_idx in range(n_edits):
                    all_predictions.append(predictions[i].cpu().numpy())
                    all_ground_truth.append(ground_truth[i].cpu().numpy())
                    all_alpha_0.append(alpha_0)
                    all_region_areas.append(area)
                    all_samples.append(samples[i])
                    all_rtt_lengths.append(rtt_lengths[i])
                    all_pbs_lengths.append(pbs_lengths[i])
                    all_edit_positions.append(edit_positions[i][edit_idx])
                    all_edit_types.append(edit_types[i][edit_idx])
                    all_edit_categories.append(edit_categories[i][edit_idx])
                    all_nick_sites.append(nick_sites[i])
                
                samples_collected += 1
                if max_examples is not None and samples_collected >= max_examples:
                    break
    
    predictions = np.array(all_predictions)
    ground_truth = np.array(all_ground_truth)
    alpha_0_values = np.array(all_alpha_0)
    region_areas = np.array(all_region_areas)
    samples = np.array(all_samples)
    rtt_lengths = np.array(all_rtt_lengths)
    pbs_lengths = np.array(all_pbs_lengths)
    edit_positions = np.array(all_edit_positions)
    edit_categories = np.array(all_edit_categories)
    nick_sites = np.array(all_nick_sites)
    
    return (predictions, ground_truth, alpha_0_values, region_areas, samples,
            rtt_lengths, pbs_lengths, edit_positions, all_edit_types, edit_categories, nick_sites)


def compute_coverage_rates(predictions, samples, ground_truth, confidence_levels):
    """Compute coverage rates for different confidence levels."""
    n_samples = len(predictions)
    observed_coverage = []
    
    print("Computing coverage rates...")
    for conf_level in tqdm(confidence_levels):
        coverage_count = 0
        
        for i in range(n_samples):
            # Convert samples to 2D
            samples_2d = simplex_to_2d(samples[i])
            ground_truth_2d = simplex_to_2d(ground_truth[i:i+1])[0]
            
            # Compute PCA confidence region
            mean, principal_axes, eigenvalues, confidence_scale, _ = compute_pca_confidence_region(
                samples_2d, confidence_level=conf_level
            )
            
            # Check if ground truth is within confidence region
            if point_in_confidence_region(ground_truth_2d, mean, principal_axes, eigenvalues, confidence_scale):
                coverage_count += 1
        
        observed_coverage.append(coverage_count / n_samples)
    
    return np.array(observed_coverage)


def compute_coverage_by_outcome(predictions, samples, ground_truth, confidence_level=0.95):
    """Compute coverage rates separately for each outcome type."""
    n_samples = len(predictions)
    coverage_by_outcome = {'edited': [], 'unedited': [], 'indel': []}
    
    print("Computing coverage by outcome type...")
    for i in tqdm(range(n_samples)):
        samples_2d = simplex_to_2d(samples[i])
        mean, principal_axes, eigenvalues, confidence_scale, _ = compute_pca_confidence_region(
            samples_2d, confidence_level=confidence_level
        )
        
        if mean is not None:
            # Check coverage for each outcome dimension
            for outcome_idx, outcome_name in enumerate(['edited', 'unedited', 'indel']):
                # Project to 1D for this outcome
                pred_1d = predictions[i, outcome_idx]
                gt_1d = ground_truth[i, outcome_idx]
                
                # For 1D, we need to check if gt is within the confidence interval
                # We'll use the marginal distribution
                outcome_samples = samples[i][:, outcome_idx]
                sorted_samples = np.sort(outcome_samples)
                lower_bound = sorted_samples[int((1 - confidence_level) / 2 * len(sorted_samples))]
                upper_bound = sorted_samples[int((1 + confidence_level) / 2 * len(sorted_samples))]
                
                is_covered = (gt_1d >= lower_bound) and (gt_1d <= upper_bound)
                coverage_by_outcome[outcome_name].append(1 if is_covered else 0)
        else:
            for outcome_name in coverage_by_outcome:
                coverage_by_outcome[outcome_name].append(0)
    
    # Compute mean coverage for each outcome
    return {name: np.mean(vals) for name, vals in coverage_by_outcome.items()}


def create_hybrid_figure(predictions, ground_truth, alpha_0_values, region_areas, samples, 
                         rtt_lengths, pbs_lengths, edit_positions, edit_types, edit_categories, nick_sites, output_dir):
    """Create the hybrid multi-row, multi-panel figure with novel analyses."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Compute prediction errors
    errors = np.abs(predictions - ground_truth)
    overall_error = np.mean(errors, axis=1)
    edited_error = errors[:, 0]
    unedited_error = errors[:, 1]
    indel_error = errors[:, 2]
    
    # Get editing efficiency (ground truth edited %)
    editing_efficiency = ground_truth[:, 0]
    
    # Create figure with 3 rows and 2 columns
    plt.style.use('default')
    fig = plt.figure(figsize=(18, 18))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3, top=0.98, bottom=0.05, left=0.05, right=0.95)
    
    # ========== ROW 1: Confidence Region Analysis (2 columns) ==========
    
    # Panel A: Region size distribution (row 1, col 0)
    ax = fig.add_subplot(gs[0, 0])
    valid_areas = region_areas[region_areas > 0]
    ax.hist(valid_areas, bins=50, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('95% Confidence Region Area', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    median_area = np.median(valid_areas)
    mean_area = np.mean(valid_areas)
    ax.axvline(median_area, color='red', linestyle='--', linewidth=2, label=f'Median: {median_area:.4f}')
    ax.axvline(mean_area, color='orange', linestyle='--', linewidth=2, label=f'Mean: {mean_area:.4f}')
    ax.legend(fontsize=9)
    
    # Panel C: Coverage by Outcome Type (row 1, col 1)
    ax = fig.add_subplot(gs[0, 1])
    coverage_by_outcome = compute_coverage_by_outcome(predictions, samples, ground_truth, confidence_level=0.95)
    
    outcome_names = ['Edited', 'Unedited', 'Indel']
    outcome_colors = ['#2E86AB', '#A23B72', '#F18F01']
    coverage_values = [coverage_by_outcome['edited'], coverage_by_outcome['unedited'], coverage_by_outcome['indel']]
    
    bars = ax.bar(outcome_names, coverage_values, color=outcome_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.axhline(0.95, color='red', linestyle='--', linewidth=2, label='Expected (95%)')
    ax.set_ylabel('Coverage Rate', fontsize=11, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=9)
    
    # Add value labels on bars
    for bar, val in zip(bars, coverage_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # ========== ROW 2: Novel Analyses (2 columns) ==========
    
    # Panel D: Region Area vs Total Concentration (row 2, col 0)
    ax = fig.add_subplot(gs[1, 0])
    scatter = ax.scatter(alpha_0_values, region_areas, c=overall_error,
                        cmap='Reds', alpha=0.6, s=20, edgecolors='black', linewidth=0.3)
    ax.set_xlabel('Total Concentration (α₀)', fontsize=11, fontweight='bold')
    ax.set_ylabel('95% Confidence Region Area', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Prediction Error', fontsize=9)
    
    # Panel E: Prediction Accuracy by Total Concentration Bins, Stratified by Editing Efficiency (row 2, col 1)
    ax = fig.add_subplot(gs[1, 1])
    
    # Stratify by editing efficiency quartiles
    efficiency_quartiles = np.percentile(editing_efficiency, [0, 25, 50, 75, 100])
    efficiency_labels = ['Low (0-25%)', 'Med-Low (25-50%)', 'Med-High (50-75%)', 'High (75-100%)']
    colors_stratified = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Create bins based on α₀ (lower α₀ = higher uncertainty)
    n_bins = 10
    alpha_0_bins = np.linspace(alpha_0_values.min(), alpha_0_values.max(), n_bins + 1)
    bin_centers = (alpha_0_bins[:-1] + alpha_0_bins[1:]) / 2
    
    for q_idx in range(len(efficiency_quartiles) - 1):
        if q_idx == len(efficiency_quartiles) - 2:
            eff_mask = (editing_efficiency >= efficiency_quartiles[q_idx]) & (editing_efficiency <= efficiency_quartiles[q_idx+1])
        else:
            eff_mask = (editing_efficiency >= efficiency_quartiles[q_idx]) & (editing_efficiency < efficiency_quartiles[q_idx+1])
        
        bin_errors = []
        bin_stds = []
        
        for i in range(len(alpha_0_bins) - 1):
            if i == len(alpha_0_bins) - 2:
                alpha_mask = (alpha_0_values >= alpha_0_bins[i]) & (alpha_0_values <= alpha_0_bins[i+1])
            else:
                alpha_mask = (alpha_0_values >= alpha_0_bins[i]) & (alpha_0_values < alpha_0_bins[i+1])
            
            combined_mask = eff_mask & alpha_mask
            if np.sum(combined_mask) > 10:  # Need enough samples
                bin_errors.append(np.mean(overall_error[combined_mask]))
                bin_stds.append(np.std(overall_error[combined_mask]))
            else:
                bin_errors.append(np.nan)
                bin_stds.append(np.nan)
        
        # Plot only non-NaN values
        valid_mask = ~np.isnan(bin_errors)
        if np.sum(valid_mask) > 0:
            ax.plot(np.array(bin_centers)[valid_mask], np.array(bin_errors)[valid_mask], 
                   'o-', linewidth=2, markersize=6, color=colors_stratified[q_idx], 
                   label=efficiency_labels[q_idx], alpha=0.8)
    
    ax.set_xlabel('Total Concentration (α₀) (Binned)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Mean Absolute Error', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='best')
    
    # ========== ROW 3: Edit Type × Position Line Plot (spans 2 columns) ==========
    
    # Panel G: Edit Type × Position Line Plot (average α₀) - spans both columns of row 3
    ax = fig.add_subplot(gs[2, :])
    
    # Filter valid edits (relative positions, not invalid)
    valid_edit_mask = (edit_positions != -999) & (edit_positions != -1) & (np.array(edit_types) != 'N-N')
    valid_positions = edit_positions[valid_edit_mask]
    valid_edit_categories = edit_categories[valid_edit_mask]
    valid_alpha_0 = alpha_0_values[valid_edit_mask]
    
    # Filter to only Transition and Transversion categories
    category_mask = np.isin(valid_edit_categories, ['Transition', 'Transversion'])
    valid_positions = valid_positions[category_mask]
    valid_edit_categories = valid_edit_categories[category_mask]
    valid_alpha_0 = valid_alpha_0[category_mask]
    
    # Use exact positions (no binning)
    if len(valid_positions) > 0:
        # Create DataFrame
        plot_data = pd.DataFrame({
            'category': valid_edit_categories,
            'position': valid_positions,
            'alpha_0': valid_alpha_0
        })
        
        # Group by category and exact position, compute mean α₀
        grouped = plot_data.groupby(['category', 'position'])['alpha_0'].mean().reset_index()
        
        # Create line plot for each category
        for category in ['Transition', 'Transversion']:
            category_data = grouped[grouped['category'] == category]
            if len(category_data) > 0:
                # Sort by position
                category_data = category_data.sort_values('position')
                ax.plot(category_data['position'], category_data['alpha_0'], 
                       marker='o', label=category, linewidth=2, markersize=4)
        
        ax.set_xlabel('Position Within RTT (bp from RTT start)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Total Concentration (α₀)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
    
    # No title - keep figure compact
    
    plt.savefig(os.path.join(output_dir, 'outcome_confidence_hybrid_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'outcome_confidence_hybrid_analysis.pdf'), 
                bbox_inches='tight')
    plt.close()
    
    print(f"Hybrid analysis figure saved to {output_dir}")


def main():
    args = parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("Loading model and data...")
    model, data_module = load_model_and_data(args.config, args.checkpoint)
    
    print("Generating predictions and computing metrics...")
    (predictions, ground_truth, alpha_0_values, region_areas, samples,
     rtt_lengths, pbs_lengths, edit_positions, edit_types, edit_categories, nick_sites) = generate_predictions_and_data(
        model, data_module, n_samples=args.n_samples, 
        batch_size=args.batch_size, max_examples=None  # Use all test data
    )
    
    print(f"Dataset size: {len(predictions)} samples")
    print(f"Total Concentration (α₀) range: {alpha_0_values.min():.4f} - {alpha_0_values.max():.4f}")
    print(f"Region area range: {region_areas[region_areas > 0].min():.4f} - {region_areas.max():.4f}")
    print(f"RTT length range: {rtt_lengths[rtt_lengths > 0].min():.0f} - {rtt_lengths.max():.0f}")
    print(f"PBS length range: {pbs_lengths[pbs_lengths > 0].min():.0f} - {pbs_lengths.max():.0f}")
    
    print("Creating hybrid figure...")
    create_hybrid_figure(predictions, ground_truth, alpha_0_values, region_areas, samples,
                        rtt_lengths, pbs_lengths, edit_positions, edit_types, edit_categories, nick_sites, args.output_dir)
    
    print(f"\nDone! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()

