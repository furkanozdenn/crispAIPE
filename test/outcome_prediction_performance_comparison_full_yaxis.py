"""
Generate performance comparison bar plot (Fig 2d) with full y-axis (0 to 1).
Reads pre-computed results from performance_results.json.

Usage:
    python test/outcome_prediction_performance_comparison_full_yaxis.py \
        --results_dir ./test/figures_revs/performance_comparison
"""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def parse_args():
    parser = argparse.ArgumentParser(description='Generate performance comparison plot with full y-axis')
    parser.add_argument('--results_dir', type=str, 
                       default='./test/figures_revs/performance_comparison',
                       help='Directory containing performance_results.json')
    return parser.parse_args()


def create_performance_comparison_plot_full_yaxis(results, output_dir):
    """Create a bar plot comparing performance across all models with y-axis from 0 to 1."""

    crispAIPE_results = results['crispAIPE']
    competing_models = {
        'OPED': results['OPED'],
        'PRIDICT': results['PRIDICT']
    }

    regression_competing_models = {
        'DeepPrime': {'spearman_correlation': 0.74},
        'EasyPrime': {'spearman_correlation': 0.67}
    }
    crispAIPE_regression_results = {'spearman_correlation': 0.814}

    crispAIPE_outcome_avg = crispAIPE_results['average']
    oped_avg = competing_models['OPED']['average']
    pridict_avg = competing_models['PRIDICT']['average']

    outcome_metrics = ['Average\nPerformance', 'Unedited\n(Spearman)', 'Unedited\n(Pearson)',
                       'Intended Edits\n(Spearman)', 'Intended Edits\n(Pearson)']
    regression_metrics = ['Spearman\nCorrelation']

    outcome_data = [
        [crispAIPE_outcome_avg, oped_avg, pridict_avg],
        [crispAIPE_results['unintended_spearman'],
         competing_models['OPED']['unintended_spearman'],
         competing_models['PRIDICT']['unintended_spearman']],
        [crispAIPE_results['unintended_pearson'],
         competing_models['OPED']['unintended_pearson'],
         competing_models['PRIDICT']['unintended_pearson']],
        [crispAIPE_results['intended_spearman'],
         competing_models['OPED']['intended_spearman'],
         competing_models['PRIDICT']['intended_spearman']],
        [crispAIPE_results['intended_pearson'],
         competing_models['OPED']['intended_pearson'],
         competing_models['PRIDICT']['intended_pearson']],
    ]

    regression_data = [
        [crispAIPE_regression_results['spearman_correlation'],
         regression_competing_models['DeepPrime']['spearman_correlation'],
         regression_competing_models['EasyPrime']['spearman_correlation']]
    ]

    all_metrics = outcome_metrics + regression_metrics
    all_data = outcome_data + regression_data

    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(18, 8))

    x = np.arange(len(all_metrics))
    width = 0.25

    bar1_colors = []
    for i in range(len(all_metrics)):
        if i < len(outcome_metrics):
            bar1_colors.append('#2E86AB')
        else:
            bar1_colors.append('#FF6B35')

    bars1 = ax.bar(x - width, [row[0] for row in all_data], width, label='crispAIPE',
                   color=bar1_colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    bar2_colors = []
    bar3_colors = []
    for i in range(len(all_metrics)):
        if i < len(outcome_metrics):
            bar2_colors.append('#A23B72')
            bar3_colors.append('#F18F01')
        else:
            bar2_colors.append('#8B4513')
            bar3_colors.append('#32CD32')

    bars2 = ax.bar(x, [row[1] for row in all_data], width, label='OPED/DeepPrime',
                   color=bar2_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x + width, [row[2] for row in all_data], width, label='PRIDICT/EasyPrime',
                   color=bar3_colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    for i in range(len(all_data)):
        add_value_labels([bars1[i], bars2[i], bars3[i]])

    ax.set_ylabel('Correlation Coefficient', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(all_metrics, fontsize=12)

    legend_elements = [
        Patch(facecolor='#2E86AB', alpha=0.8, label='crispAIPE'),
        Patch(facecolor='#FF6B35', alpha=0.8, label='crispAIPE-reg'),
        Patch(facecolor='#A23B72', alpha=0.8, label='OPED'),
        Patch(facecolor='#F18F01', alpha=0.8, label='PRIDICT'),
        Patch(facecolor='#8B4513', alpha=0.8, label='DeepPrime'),
        Patch(facecolor='#32CD32', alpha=0.8, label='EasyPrime')
    ]
    ax.legend(handles=legend_elements, fontsize=12, loc='upper right', ncol=2)

    ax.grid(True, alpha=0.3, axis='y')

    ax.set_ylim(0, 1.0)

    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))

    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    plt.tight_layout()

    plot_path = os.path.join(output_dir, 'performance_comparison_full_yaxis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(output_dir, 'performance_comparison_full_yaxis.pdf'),
                bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Full y-axis plot saved to {plot_path}")


def main():
    args = parse_args()

    results_path = os.path.join(args.results_dir, 'performance_results.json')
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found. Run outcome_prediction_performance_comparison.py first.")
        sys.exit(1)

    with open(results_path, 'r') as f:
        results = json.load(f)

    create_performance_comparison_plot_full_yaxis(results, args.results_dir)


if __name__ == "__main__":
    main()
