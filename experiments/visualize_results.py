"""
Visualization and Analysis for CDN Cache Policy Experiments
Generates plots and statistical analysis for thesis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import os
from typing import Dict, List, Tuple, Any

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


class ResultsAnalyzer:
    """Analyze and visualize cache policy experiment results"""
    
    def __init__(self, results_dir: str = "experiments/results"):
        """
        Initialize analyzer
        
        Args:
            results_dir: Directory containing experiment results
        """
        self.results_dir = results_dir
        self.figures_dir = os.path.join(results_dir, "figures")
        os.makedirs(self.figures_dir, exist_ok=True)
    
    def load_simulation_results(self, filename: str = "cache_simulation_results.csv") -> pd.DataFrame:
        """Load simulation results from CSV"""
        filepath = os.path.join(self.results_dir, filename)
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        else:
            print(f"File not found: {filepath}")
            return pd.DataFrame()
    
    def plot_hit_ratio_comparison(self, df: pd.DataFrame, save: bool = True):
        """Plot hit ratio comparison across policies and workloads"""
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Group by cache size
        cache_sizes = df['cache_size_mb'].unique()
        
        for idx, cache_size in enumerate(sorted(cache_sizes)):
            ax = axes[idx]
            data = df[df['cache_size_mb'] == cache_size]
            
            # Pivot for heatmap
            pivot = data.pivot_table(
                values='hit_ratio',
                index='policy',
                columns='workload',
                aggfunc='mean'
            )
            
            sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', 
                       ax=ax, cbar_kws={'label': 'Hit Ratio'})
            ax.set_title(f'Cache Size: {cache_size} MB')
            ax.set_xlabel('Workload')
            ax.set_ylabel('Policy')
        
        plt.suptitle('Cache Hit Ratio Comparison', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.figures_dir, 'hit_ratio_heatmap.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_byte_hit_ratio(self, df: pd.DataFrame, save: bool = True):
        """Plot byte hit ratio comparison"""
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calculate mean and std for each policy-workload combination
        summary = df.groupby(['policy', 'workload'])['byte_hit_ratio'].agg(['mean', 'std']).reset_index()
        
        # Bar plot with error bars
        workloads = summary['workload'].unique()
        policies = summary['policy'].unique()
        
        x = np.arange(len(workloads))
        width = 0.2
        
        for i, policy in enumerate(policies):
            policy_data = summary[summary['policy'] == policy]
            means = [policy_data[policy_data['workload'] == w]['mean'].values[0] 
                    if len(policy_data[policy_data['workload'] == w]) > 0 else 0 
                    for w in workloads]
            stds = [policy_data[policy_data['workload'] == w]['std'].values[0] 
                   if len(policy_data[policy_data['workload'] == w]) > 0 else 0 
                   for w in workloads]
            
            ax.bar(x + i * width, means, width, label=policy.upper(), yerr=stds, capsize=5)
        
        ax.set_xlabel('Workload Type', fontsize=12)
        ax.set_ylabel('Byte Hit Ratio', fontsize=12)
        ax.set_title('Byte Hit Ratio Comparison Across Policies and Workloads', fontsize=14)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(workloads)
        ax.legend(title='Policy')
        ax.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(os.path.join(self.figures_dir, 'byte_hit_ratio_bars.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_cache_size_sensitivity(self, df: pd.DataFrame, save: bool = True):
        """Plot sensitivity analysis for cache size"""
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Hit ratio vs cache size
        ax1 = axes[0]
        for policy in df['policy'].unique():
            policy_data = df[df['policy'] == policy]
            summary = policy_data.groupby('cache_size_mb')['hit_ratio'].agg(['mean', 'std']).reset_index()
            
            ax1.errorbar(summary['cache_size_mb'], summary['mean'], 
                        yerr=summary['std'], marker='o', label=policy.upper(), 
                        capsize=5, linewidth=2, markersize=8)
        
        ax1.set_xlabel('Cache Size (MB)', fontsize=12)
        ax1.set_ylabel('Hit Ratio', fontsize=12)
        ax1.set_title('Hit Ratio vs Cache Size', fontsize=14)
        ax1.legend(title='Policy')
        ax1.grid(True, alpha=0.3)
        
        # Byte hit ratio vs cache size
        ax2 = axes[1]
        for policy in df['policy'].unique():
            policy_data = df[df['policy'] == policy]
            summary = policy_data.groupby('cache_size_mb')['byte_hit_ratio'].agg(['mean', 'std']).reset_index()
            
            ax2.errorbar(summary['cache_size_mb'], summary['mean'], 
                        yerr=summary['std'], marker='s', label=policy.upper(), 
                        capsize=5, linewidth=2, markersize=8)
        
        ax2.set_xlabel('Cache Size (MB)', fontsize=12)
        ax2.set_ylabel('Byte Hit Ratio', fontsize=12)
        ax2.set_title('Byte Hit Ratio vs Cache Size', fontsize=14)
        ax2.legend(title='Policy')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Cache Size Sensitivity Analysis', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.figures_dir, 'cache_size_sensitivity.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_eviction_rate_analysis(self, df: pd.DataFrame, save: bool = True):
        """Plot eviction rate analysis"""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Box plot of eviction rates
        data_to_plot = []
        labels = []
        
        for policy in df['policy'].unique():
            policy_data = df[df['policy'] == policy]['eviction_rate']
            data_to_plot.append(policy_data)
            labels.append(policy.upper())
        
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        
        # Color the boxes
        colors = sns.color_palette("husl", len(labels))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xlabel('Policy', fontsize=12)
        ax.set_ylabel('Eviction Rate', fontsize=12)
        ax.set_title('Eviction Rate Distribution by Policy', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        
        if save:
            plt.savefig(os.path.join(self.figures_dir, 'eviction_rate_boxplot.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_workload_comparison(self, df: pd.DataFrame, metric: str = 'hit_ratio', save: bool = True):
        """Plot comparison across different workload types"""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Violin plot for distribution comparison
        sns.violinplot(data=df, x='workload', y=metric, hue='policy', ax=ax)
        
        ax.set_xlabel('Workload Type', fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'{metric.replace("_", " ").title()} Distribution by Workload', fontsize=14)
        ax.legend(title='Policy', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.figures_dir, f'{metric}_violin.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
    
    def calculate_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive statistics for all experiments"""
        
        # Group by policy and calculate statistics
        stats_df = df.groupby('policy').agg({
            'hit_ratio': ['mean', 'std', 'min', 'max'],
            'byte_hit_ratio': ['mean', 'std', 'min', 'max'],
            'eviction_rate': ['mean', 'std'],
            'bytes_from_origin_mb': ['mean', 'sum'],
            'bytes_from_cache_mb': ['mean', 'sum']
        }).round(4)
        
        # Flatten column names
        stats_df.columns = ['_'.join(col).strip() for col in stats_df.columns.values]
        
        # Calculate origin traffic reduction
        total_traffic = df.groupby('policy')[['bytes_from_origin_mb', 'bytes_from_cache_mb']].sum().sum(axis=1)
        origin_traffic = df.groupby('policy')['bytes_from_origin_mb'].sum()
        stats_df['origin_reduction_pct'] = ((1 - origin_traffic / total_traffic) * 100).round(2)
        
        return stats_df
    
    def perform_statistical_tests(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform statistical significance tests between policies"""
        
        results = {}
        policies = df['policy'].unique()
        
        # Pairwise t-tests for hit ratio
        hit_ratio_tests = {}
        for i, policy1 in enumerate(policies):
            for policy2 in policies[i+1:]:
                data1 = df[df['policy'] == policy1]['hit_ratio']
                data2 = df[df['policy'] == policy2]['hit_ratio']
                
                t_stat, p_value = stats.ttest_ind(data1, data2)
                hit_ratio_tests[f'{policy1}_vs_{policy2}'] = {
                    't_statistic': round(t_stat, 4),
                    'p_value': round(p_value, 4),
                    'significant': p_value < 0.05
                }
        
        results['hit_ratio_tests'] = hit_ratio_tests
        
        # ANOVA test across all policies
        policy_groups = [df[df['policy'] == p]['hit_ratio'].values for p in policies]
        f_stat, p_value = stats.f_oneway(*policy_groups)
        
        results['anova'] = {
            'f_statistic': round(f_stat, 4),
            'p_value': round(p_value, 4),
            'significant': p_value < 0.05
        }
        
        return results
    
    def generate_latex_tables(self, df: pd.DataFrame):
        """Generate LaTeX tables for thesis"""
        
        # Summary statistics table
        stats_df = self.calculate_statistics(df)
        
        latex_stats = stats_df.to_latex(
            caption="Cache Policy Performance Statistics",
            label="tab:cache_stats",
            column_format='l' + 'r' * len(stats_df.columns),
            float_format="%.4f"
        )
        
        # Save to file
        with open(os.path.join(self.results_dir, 'stats_table.tex'), 'w') as f:
            f.write(latex_stats)
        
        # Comparison table by workload
        comparison = df.pivot_table(
            values=['hit_ratio', 'byte_hit_ratio'],
            index='policy',
            columns='workload',
            aggfunc='mean'
        ).round(4)
        
        latex_comparison = comparison.to_latex(
            caption="Policy Performance by Workload Type",
            label="tab:workload_comparison",
            multicolumn=True,
            multicolumn_format='c',
            float_format="%.4f"
        )
        
        with open(os.path.join(self.results_dir, 'comparison_table.tex'), 'w') as f:
            f.write(latex_comparison)
        
        print("LaTeX tables saved to results directory")
    
    def generate_summary_report(self, df: pd.DataFrame):
        """Generate a comprehensive summary report"""
        
        report = []
        report.append("=" * 80)
        report.append("CDN CACHE POLICY EVALUATION SUMMARY REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Overall statistics
        report.append("1. OVERALL PERFORMANCE METRICS")
        report.append("-" * 40)
        stats_df = self.calculate_statistics(df)
        report.append(stats_df.to_string())
        report.append("")
        
        # Best performing policy per workload
        report.append("2. BEST POLICY PER WORKLOAD")
        report.append("-" * 40)
        for workload in df['workload'].unique():
            workload_data = df[df['workload'] == workload]
            best_hit = workload_data.loc[workload_data['hit_ratio'].idxmax()]
            best_byte = workload_data.loc[workload_data['byte_hit_ratio'].idxmax()]
            
            report.append(f"\nWorkload: {workload}")
            report.append(f"  Best Hit Ratio: {best_hit['policy'].upper()} ({best_hit['hit_ratio']:.4f})")
            report.append(f"  Best Byte Hit Ratio: {best_byte['policy'].upper()} ({best_byte['byte_hit_ratio']:.4f})")
        
        # Statistical significance
        report.append("\n3. STATISTICAL SIGNIFICANCE TESTS")
        report.append("-" * 40)
        test_results = self.perform_statistical_tests(df)
        
        report.append(f"\nANOVA Test: F={test_results['anova']['f_statistic']}, p={test_results['anova']['p_value']}")
        report.append(f"Significant difference between policies: {test_results['anova']['significant']}")
        
        report.append("\nPairwise t-tests (Hit Ratio):")
        for comparison, results in test_results['hit_ratio_tests'].items():
            report.append(f"  {comparison}: t={results['t_statistic']}, p={results['p_value']}, significant={results['significant']}")
        
        # Key findings
        report.append("\n4. KEY FINDINGS")
        report.append("-" * 40)
        
        # Find best overall policy
        best_overall = stats_df['hit_ratio_mean'].idxmax()
        report.append(f"• Best overall policy (hit ratio): {best_overall.upper()}")
        
        best_byte = stats_df['byte_hit_ratio_mean'].idxmax()
        report.append(f"• Best overall policy (byte hit ratio): {best_byte.upper()}")
        
        # Cache size impact
        size_impact = df.groupby(['cache_size_mb', 'policy'])['hit_ratio'].mean().unstack()
        report.append(f"\n• Cache size impact:")
        for size in sorted(df['cache_size_mb'].unique()):
            best_at_size = size_impact.loc[size].idxmax()
            report.append(f"  At {size}MB: {best_at_size.upper()} performs best")
        
        # Save report
        report_text = "\n".join(report)
        with open(os.path.join(self.results_dir, 'summary_report.txt'), 'w') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\nReport saved to {os.path.join(self.results_dir, 'summary_report.txt')}")


def main():
    """Main function to run all analyses"""
    
    analyzer = ResultsAnalyzer()
    
    # Load simulation results
    df = analyzer.load_simulation_results()
    
    if df.empty:
        print("No simulation results found. Please run cache_simulator.py first.")
        return
    
    print("Generating visualizations and analysis...")
    
    # Generate all plots
    analyzer.plot_hit_ratio_comparison(df)
    analyzer.plot_byte_hit_ratio(df)
    analyzer.plot_cache_size_sensitivity(df)
    analyzer.plot_eviction_rate_analysis(df)
    analyzer.plot_workload_comparison(df, 'hit_ratio')
    analyzer.plot_workload_comparison(df, 'byte_hit_ratio')
    
    # Generate tables and report
    analyzer.generate_latex_tables(df)
    analyzer.generate_summary_report(df)
    
    print("\nAnalysis complete! Check the experiments/results/figures directory for plots.")


if __name__ == "__main__":
    main()
