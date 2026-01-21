"""
Plot visualizations for simulated candidates data
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import argparse

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'


def load_data(csv_path):
    """Load the simulated candidates CSV file"""
    df = pd.read_csv(csv_path)
    return df


def plot_distributions(df, output_dir=None):
    """Plot distributions of key variables"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Distribution of N_Papers by True_Fit
    ax1 = axes[0, 0]
    for fit in ['Good', 'Bad']:
        data = df[df['True_Fit'] == fit]['N_Papers']
        ax1.hist(data, alpha=0.6, label=fit, bins=range(1, 12))
    ax1.set_xlabel('Number of Papers (N)', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Distribution of Papers by True Fit', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Distribution of Posterior_Good
    ax2 = axes[0, 1]
    ax2.hist(df['Posterior_Good'], bins=30, alpha=0.7, edgecolor='black')
    ax2.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold (0.5)')
    ax2.set_xlabel('Posterior Probability (Good)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Distribution of Posterior Probability', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. Distribution of Expected_Payoff
    ax3 = axes[1, 0]
    ax3.hist(df['Expected_Payoff'], bins=30, alpha=0.7, edgecolor='black', color='green')
    ax3.axvline(0.0, color='red', linestyle='--', linewidth=2, label='Payoff Threshold (0.0)')
    ax3.set_xlabel('Expected Payoff', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Distribution of Expected Payoff', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 4. Network Effect distribution
    ax4 = axes[1, 1]
    network_counts = df['Network_Effect'].value_counts()
    ax4.bar(network_counts.index, network_counts.values, alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Network Effect', fontsize=11)
    ax4.set_ylabel('Count', fontsize=11)
    ax4.set_title('Distribution of Network Effect', fontsize=12, fontweight='bold')
    for i, v in enumerate(network_counts.values):
        ax4.text(i, v, str(v), ha='center', va='bottom', fontweight='bold')
    ax4.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'distributions.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: {os.path.join(output_dir, 'distributions.png')}")
    plt.show()


def plot_relationships(df, output_dir=None):
    """Plot relationships between variables"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Posterior_Good vs N_Papers by Network_Effect
    ax1 = axes[0, 0]
    for network in ['Yes', 'No']:
        data = df[df['Network_Effect'] == network]
        ax1.scatter(data['N_Papers'], data['Posterior_Good'], 
                   alpha=0.5, label=f'Network: {network}', s=30)
    ax1.axhline(0.5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Decision Threshold')
    ax1.set_xlabel('Number of Papers (N)', fontsize=11)
    ax1.set_ylabel('Posterior Probability (Good)', fontsize=11)
    ax1.set_title('Posterior Probability vs Papers by Network Effect', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Expected_Payoff vs N_Papers by Network_Effect
    ax2 = axes[0, 1]
    for network in ['Yes', 'No']:
        data = df[df['Network_Effect'] == network]
        ax2.scatter(data['N_Papers'], data['Expected_Payoff'], 
                   alpha=0.5, label=f'Network: {network}', s=30)
    ax2.axhline(0.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Payoff Threshold')
    ax2.set_xlabel('Number of Papers (N)', fontsize=11)
    ax2.set_ylabel('Expected Payoff', fontsize=11)
    ax2.set_title('Expected Payoff vs Papers by Network Effect', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. Box plot: Posterior_Good by Network_Effect and True_Fit
    ax3 = axes[1, 0]
    df_plot = df.copy()
    df_plot['Group'] = df_plot['Network_Effect'] + ' / ' + df_plot['True_Fit']
    sns.boxplot(data=df_plot, x='Group', y='Posterior_Good', ax=ax3)
    ax3.axhline(0.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax3.set_xlabel('Network Effect / True Fit', fontsize=11)
    ax3.set_ylabel('Posterior Probability (Good)', fontsize=11)
    ax3.set_title('Posterior Probability by Network Effect and True Fit', fontsize=12, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(alpha=0.3, axis='y')
    
    # 4. Average Posterior_Good by N_Papers
    ax4 = axes[1, 1]
    avg_posterior = df.groupby(['N_Papers', 'Network_Effect'])['Posterior_Good'].mean().reset_index()
    for network in ['Yes', 'No']:
        data = avg_posterior[avg_posterior['Network_Effect'] == network]
        ax4.plot(data['N_Papers'], data['Posterior_Good'], 
                marker='o', label=f'Network: {network}', linewidth=2, markersize=8)
    ax4.axhline(0.5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Decision Threshold')
    ax4.set_xlabel('Number of Papers (N)', fontsize=11)
    ax4.set_ylabel('Average Posterior Probability (Good)', fontsize=11)
    ax4.set_title('Average Posterior Probability by Papers', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    ax4.set_xlim(0.5, 10.5)
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'relationships.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: {os.path.join(output_dir, 'relationships.png')}")
    plt.show()


def plot_decisions(df, output_dir=None):
    """Plot decision analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Decision comparison: Prob-based vs Payoff-based
    ax1 = axes[0, 0]
    decision_counts = pd.crosstab(df['Decision_Prob_Based'], df['Decision_Payoff_Based'])
    sns.heatmap(decision_counts, annot=True, fmt='d', cmap='Blues', ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_xlabel('Payoff-Based Decision', fontsize=11)
    ax1.set_ylabel('Probability-Based Decision', fontsize=11)
    ax1.set_title('Decision Agreement Matrix', fontsize=12, fontweight='bold')
    
    # 2. Accuracy by decision method
    ax2 = axes[0, 1]
    # Probability-based accuracy
    pred_prob = df['Decision_Prob_Based'].map({'Hire': 'Good', 'Reject': 'Bad'})
    accuracy_prob = (df['True_Fit'] == pred_prob).mean()
    
    # Payoff-based accuracy
    pred_payoff = df['Decision_Payoff_Based'].map({'Hire': 'Good', 'Reject': 'Bad'})
    accuracy_payoff = (df['True_Fit'] == pred_payoff).mean()
    
    methods = ['Probability-Based', 'Payoff-Based']
    accuracies = [accuracy_prob, accuracy_payoff]
    colors = ['#3498db', '#2ecc71']
    bars = ax2.bar(methods, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Accuracy', fontsize=11)
    ax2.set_title('Prediction Accuracy by Decision Method', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.grid(alpha=0.3, axis='y')
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{acc:.1%}', ha='center', fontsize=12, fontweight='bold')
    
    # 3. Confusion matrix for probability-based decisions
    ax3 = axes[1, 0]
    cm_prob = pd.crosstab(df['True_Fit'], pred_prob, margins=True)
    sns.heatmap(cm_prob.iloc[:-1, :-1], annot=True, fmt='d', cmap='RdYlGn', ax=ax3, 
                cbar_kws={'label': 'Count'}, vmin=0)
    ax3.set_xlabel('Predicted Fit', fontsize=11)
    ax3.set_ylabel('True Fit', fontsize=11)
    ax3.set_title('Confusion Matrix: Probability-Based', fontsize=12, fontweight='bold')
    
    # 4. Confusion matrix for payoff-based decisions
    ax4 = axes[1, 1]
    cm_payoff = pd.crosstab(df['True_Fit'], pred_payoff, margins=True)
    sns.heatmap(cm_payoff.iloc[:-1, :-1], annot=True, fmt='d', cmap='RdYlGn', ax=ax4,
                cbar_kws={'label': 'Count'}, vmin=0)
    ax4.set_xlabel('Predicted Fit', fontsize=11)
    ax4.set_ylabel('True Fit', fontsize=11)
    ax4.set_title('Confusion Matrix: Payoff-Based', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'decisions.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: {os.path.join(output_dir, 'decisions.png')}")
    plt.show()


def plot_summary_statistics(df, output_dir=None):
    """Plot summary statistics"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Average metrics by True_Fit and Network_Effect
    ax1 = axes[0, 0]
    summary = df.groupby(['True_Fit', 'Network_Effect']).agg({
        'Posterior_Good': 'mean',
        'Expected_Payoff': 'mean',
        'N_Papers': 'mean'
    }).reset_index()
    
    x = np.arange(len(summary))
    width = 0.25
    ax1.bar(x - width, summary['Posterior_Good'], width, label='Posterior Good', alpha=0.7)
    ax1.bar(x, summary['Expected_Payoff']/10, width, label='Expected Payoff (scaled)', alpha=0.7)
    ax1.bar(x + width, summary['N_Papers']/10, width, label='N Papers (scaled)', alpha=0.7)
    ax1.set_xlabel('True Fit / Network Effect', fontsize=11)
    ax1.set_ylabel('Normalized Value', fontsize=11)
    ax1.set_title('Average Metrics by Group', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{row['True_Fit']}/{row['Network_Effect']}" 
                         for _, row in summary.iterrows()], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(alpha=0.3, axis='y')
    
    # 2. Hire rate by N_Papers
    ax2 = axes[0, 1]
    hire_rate = df.groupby('N_Papers')['Decision_Prob_Based'].apply(
        lambda x: (x == 'Hire').mean()
    ).reset_index()
    ax2.plot(hire_rate['N_Papers'], hire_rate['Decision_Prob_Based'], 
            marker='o', linewidth=2, markersize=8, color='blue')
    ax2.set_xlabel('Number of Papers (N)', fontsize=11)
    ax2.set_ylabel('Hire Rate', fontsize=11)
    ax2.set_title('Hire Rate by Number of Papers', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.grid(alpha=0.3)
    
    # 3. Average Expected Payoff by N_Papers and Network_Effect
    ax3 = axes[1, 0]
    payoff_by_group = df.groupby(['N_Papers', 'Network_Effect'])['Expected_Payoff'].mean().reset_index()
    for network in ['Yes', 'No']:
        data = payoff_by_group[payoff_by_group['Network_Effect'] == network]
        ax3.plot(data['N_Papers'], data['Expected_Payoff'], 
                marker='o', label=f'Network: {network}', linewidth=2, markersize=8)
    ax3.axhline(0.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Payoff Threshold')
    ax3.set_xlabel('Number of Papers (N)', fontsize=11)
    ax3.set_ylabel('Average Expected Payoff', fontsize=11)
    ax3.set_title('Average Expected Payoff by Papers and Network', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)
    ax3.set_xlim(0.5, 10.5)
    
    # 4. Decision distribution
    ax4 = axes[1, 1]
    decision_counts = df['Decision_Prob_Based'].value_counts()
    colors = ['#2ecc71' if d == 'Hire' else '#e74c3c' for d in decision_counts.index]
    bars = ax4.bar(decision_counts.index, decision_counts.values, 
                   color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax4.set_xlabel('Decision', fontsize=11)
    ax4.set_ylabel('Count', fontsize=11)
    ax4.set_title('Decision Distribution (Probability-Based)', fontsize=12, fontweight='bold')
    for bar, count in zip(bars, decision_counts.values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                str(count), ha='center', va='bottom', fontweight='bold')
    ax4.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'summary_statistics.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: {os.path.join(output_dir, 'summary_statistics.png')}")
    plt.show()


def main(csv_path, output_dir=None, show_plots=True):
    """
    Main function to generate all plots
    
    Parameters:
    -----------
    csv_path : str
        Path to the simulated candidates CSV file
    output_dir : str, optional
        Directory to save plots. If None, plots are only displayed.
    show_plots : bool
        Whether to display plots (default: True)
    """
    # Load data
    print(f"Loading data from {csv_path}...")
    df = load_data(csv_path)
    print(f"Loaded {len(df)} candidates")
    
    # Generate all plots
    print("\nGenerating plots...")
    
    if not show_plots:
        plt.ioff()  # Turn off interactive mode
    
    plot_distributions(df, output_dir)
    plot_relationships(df, output_dir)
    plot_decisions(df, output_dir)
    plot_summary_statistics(df, output_dir)
    
    print("\nAll plots generated!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Plot visualizations for simulated candidates data'
    )
    parser.add_argument(
        '--csv',
        type=str,
        default='results/simulated_candidates.csv',
        help='Path to simulated candidates CSV file (default: results/simulated_candidates.csv)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/plots',
        help='Directory to save plots (default: results/plots)'
    )
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Do not display plots (only save them)'
    )
    
    args = parser.parse_args()
    
    main(
        csv_path=args.csv,
        output_dir=args.output_dir,
        show_plots=not args.no_show
    )
