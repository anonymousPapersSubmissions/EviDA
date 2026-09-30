"""
Comprehensive Results Analysis
Analyzes evaluation results and generates paper-ready tables/figures
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set plot style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


def load_results(results_dir='evaluation_results'):
    """Load all evaluation results"""
    results_dir = Path(results_dir)
    
    results = {}
    
    # Load JSON files
    for json_file in ['overall_metrics.json', 'domain_metrics.json', 'uncertainty_analysis.json']:
        path = results_dir / json_file
        if path.exists():
            with open(path) as f:
                results[json_file.replace('.json', '')] = json.load(f)
    
    # Load CSV files
    for csv_file in ['predictions.csv', 'errors.csv']:
        path = results_dir / csv_file
        if path.exists():
            results[csv_file.replace('.csv', '')] = pd.read_csv(path)
    
    return results


def print_overall_metrics(results):
    """Print overall performance metrics"""
    logger.info("\n" + "="*80)
    logger.info("OVERALL PERFORMANCE")
    logger.info("="*80)
    
    metrics = results['overall_metrics']
    
    # Main metrics
    main_metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    for metric in main_metrics:
        if metric in metrics:
            logger.info(f"{metric.upper():15s}: {metrics[metric]:.4f}")
    
    # Per-class metrics
    logger.info("\nPer-Class Performance:")
    for class_name in ['real', 'fake']:
        logger.info(f"\n{class_name.upper()}:")
        for metric_type in ['precision', 'recall', 'f1']:
            key = f'{class_name}_{metric_type}'
            if key in metrics:
                logger.info(f"  {metric_type:10s}: {metrics[key]:.4f}")


def print_domain_metrics(results):
    """Print per-domain performance"""
    logger.info("\n" + "="*80)
    logger.info("PER-DOMAIN PERFORMANCE")
    logger.info("="*80)
    
    domain_metrics = results['domain_metrics']
    
    # Create comparison table
    domains = list(domain_metrics.keys())
    metrics_list = ['accuracy', 'precision', 'recall', 'f1']
    
    # Print table header
    header = f"{'Domain':<15s} | " + " | ".join([f"{m.capitalize():>10s}" for m in metrics_list])
    logger.info("\n" + header)
    logger.info("-" * len(header))
    
    # Print each domain
    for domain in sorted(domains):
        metrics = domain_metrics[domain]
        row = f"{domain:<15s} | "
        row += " | ".join([f"{metrics.get(m, 0):>10.4f}" for m in metrics_list])
        logger.info(row)
    
    # Calculate and print average
    avg_metrics = {}
    for metric in metrics_list:
        values = [domain_metrics[d].get(metric, 0) for d in domains]
        avg_metrics[metric] = np.mean(values)
    
    logger.info("-" * len(header))
    row = f"{'AVERAGE':<15s} | "
    row += " | ".join([f"{avg_metrics[m]:>10.4f}" for m in metrics_list])
    logger.info(row)


def print_uncertainty_analysis(results):
    """Print uncertainty analysis"""
    logger.info("\n" + "="*80)
    logger.info("UNCERTAINTY ANALYSIS")
    logger.info("="*80)
    
    unc_analysis = results['uncertainty_analysis']
    
    for domain, metrics in sorted(unc_analysis.items()):
        logger.info(f"\n{domain.upper()}:")
        logger.info(f"  Overall Mean:     {metrics['overall_mean']:.4f}")
        logger.info(f"  Overall Std:      {metrics['overall_std']:.4f}")
        logger.info(f"  Correct Mean:     {metrics['correct_mean']:.4f}")
        logger.info(f"  Incorrect Mean:   {metrics['incorrect_mean']:.4f}")
        
        # Calibration quality (lower is better)
        calibration_gap = metrics['incorrect_mean'] - metrics['correct_mean']
        logger.info(f"  Calibration Gap:  {calibration_gap:.4f}")
        
        if calibration_gap > 0:
            logger.info(f"  ✓ Well calibrated (uncertain about errors)")
        else:
            logger.info(f"  ⚠ Poor calibration (overconfident on errors)")


def analyze_errors(results):
    """Analyze error patterns"""
    logger.info("\n" + "="*80)
    logger.info("ERROR ANALYSIS")
    logger.info("="*80)
    
    errors = results['errors']
    
    logger.info(f"\nTotal Errors: {len(errors)}")
    
    # Errors by source
    logger.info("\nErrors by Source:")
    error_counts = errors['source'].value_counts()
    for source, count in error_counts.items():
        total = len(results['predictions'][results['predictions']['source'] == source])
        error_rate = count / total * 100
        logger.info(f"  {source:15s}: {count:4d} / {total:5d} ({error_rate:5.2f}%)")
    
    # High uncertainty errors
    if 'uncertainty' in errors.columns:
        high_unc_errors = errors[errors['uncertainty'] > 0.5]
        logger.info(f"\nHigh Uncertainty Errors (unc > 0.5): {len(high_unc_errors)} ({len(high_unc_errors)/len(errors)*100:.1f}%)")
        
        low_unc_errors = errors[errors['uncertainty'] <= 0.5]
        logger.info(f"Low Uncertainty Errors (unc ≤ 0.5): {len(low_unc_errors)} ({len(low_unc_errors)/len(errors)*100:.1f}%)")


def create_paper_table(results, output_dir='evaluation_results'):
    """Create LaTeX table for paper"""
    logger.info("\n" + "="*80)
    logger.info("GENERATING LATEX TABLE")
    logger.info("="*80)
    
    domain_metrics = results['domain_metrics']
    
    # Create DataFrame
    table_data = []
    for domain in sorted(domain_metrics.keys()):
        metrics = domain_metrics[domain]
        table_data.append({
            'Domain': domain.capitalize(),
            'Accuracy': metrics.get('accuracy', 0) * 100,
            'Precision': metrics.get('precision', 0) * 100,
            'Recall': metrics.get('recall', 0) * 100,
            'F1': metrics.get('f1', 0) * 100,
        })
    
    df = pd.DataFrame(table_data)
    
    # Generate LaTeX
    latex = df.to_latex(
        index=False,
        float_format="%.2f",
        caption="Per-domain performance comparison",
        label="tab:domain_performance"
    )
    
    # Save
    output_path = Path(output_dir) / 'paper_table.tex'
    with open(output_path, 'w') as f:
        f.write(latex)
    
    logger.info(f"\n✓ LaTeX table saved to: {output_path}")
    logger.info("\nPreview:")
    logger.info(latex)


def create_comparison_figure(results, output_dir='evaluation_results'):
    """Create comparison figure for paper"""
    logger.info("\n" + "="*80)
    logger.info("GENERATING COMPARISON FIGURE")
    logger.info("="*80)
    
    domain_metrics = results['domain_metrics']
    
    # Prepare data
    domains = sorted(domain_metrics.keys())
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1']
    metrics_keys = ['accuracy', 'precision', 'recall', 'f1']
    
    data = {metric: [] for metric in metrics_names}
    for domain in domains:
        for metric_name, metric_key in zip(metrics_names, metrics_keys):
            data[metric_name].append(domain_metrics[domain].get(metric_key, 0) * 100)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(domains))
    width = 0.2
    
    for i, metric in enumerate(metrics_names):
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, data[metric], width, label=metric, alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Domain', fontsize=14)
    ax.set_ylabel('Performance (%)', fontsize=14)
    ax.set_title('Per-Domain Performance Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in domains])
    ax.legend(loc='lower right', fontsize=11)
    ax.set_ylim([0, 105])
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'performance_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Figure saved to: {output_path}")


def create_uncertainty_correlation_plot(results, output_dir='evaluation_results'):
    """Plot uncertainty vs accuracy correlation"""
    logger.info("\n" + "="*80)
    logger.info("GENERATING UNCERTAINTY CORRELATION PLOT")
    logger.info("="*80)
    
    predictions = results['predictions']
    
    if 'uncertainty' not in predictions.columns:
        logger.warning("Uncertainty data not available")
        return
    
    # Create bins based on uncertainty
    predictions['unc_bin'] = pd.cut(predictions['uncertainty'], bins=10)
    
    # Calculate accuracy per bin
    bin_stats = predictions.groupby('unc_bin').agg({
        'correct': ['mean', 'count'],
        'uncertainty': 'mean'
    }).reset_index()
    
    bin_stats.columns = ['unc_bin', 'accuracy', 'count', 'avg_uncertainty']
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Uncertainty vs Accuracy
    ax1.scatter(bin_stats['avg_uncertainty'], bin_stats['accuracy'] * 100, 
               s=bin_stats['count']*2, alpha=0.6, c='steelblue')
    ax1.set_xlabel('Average Uncertainty', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Uncertainty vs Accuracy\n(size = sample count)', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    # Add correlation coefficient
    corr = np.corrcoef(bin_stats['avg_uncertainty'], bin_stats['accuracy'])[0, 1]
    ax1.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
            transform=ax1.transAxes, fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Distribution comparison
    correct_unc = predictions[predictions['correct'] == True]['uncertainty']
    incorrect_unc = predictions[predictions['correct'] == False]['uncertainty']
    
    ax2.hist(correct_unc, bins=30, alpha=0.6, label='Correct', color='green', density=True)
    ax2.hist(incorrect_unc, bins=30, alpha=0.6, label='Incorrect', color='red', density=True)
    ax2.set_xlabel('Uncertainty', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Uncertainty Distribution\n(Correct vs Incorrect)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'uncertainty_correlation.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Figure saved to: {output_path}")
    logger.info(f"Correlation coefficient: {corr:.4f}")


def generate_summary_report(results, output_dir='evaluation_results'):
    """Generate comprehensive summary report"""
    logger.info("\n" + "="*80)
    logger.info("GENERATING SUMMARY REPORT")
    logger.info("="*80)
    
    output_path = Path(output_dir) / 'summary_report.txt'
    
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("EVALUATION SUMMARY REPORT\n")
        f.write("Cross-Domain Fake News Detection with Uncertainty-Weighted Domain Adaptation\n")
        f.write("="*80 + "\n\n")
        
        # Overall metrics
        f.write("OVERALL PERFORMANCE\n")
        f.write("-"*80 + "\n")
        metrics = results['overall_metrics']
        for key in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
            if key in metrics:
                f.write(f"{key.upper():15s}: {metrics[key]:.4f}\n")
        
        # Domain breakdown
        f.write("\n" + "="*80 + "\n")
        f.write("PER-DOMAIN PERFORMANCE\n")
        f.write("-"*80 + "\n")
        
        domain_metrics = results['domain_metrics']
        for domain in sorted(domain_metrics.keys()):
            f.write(f"\n{domain.upper()}:\n")
            dm = domain_metrics[domain]
            for metric in ['accuracy', 'precision', 'recall', 'f1']:
                if metric in dm:
                    f.write(f"  {metric:10s}: {dm[metric]:.4f}\n")
        
        # Uncertainty analysis
        f.write("\n" + "="*80 + "\n")
        f.write("UNCERTAINTY ANALYSIS\n")
        f.write("-"*80 + "\n")
        
        unc_analysis = results['uncertainty_analysis']
        for domain, metrics in sorted(unc_analysis.items()):
            f.write(f"\n{domain.upper()}:\n")
            f.write(f"  Overall Mean:   {metrics['overall_mean']:.4f}\n")
            f.write(f"  Correct Mean:   {metrics['correct_mean']:.4f}\n")
            f.write(f"  Incorrect Mean: {metrics['incorrect_mean']:.4f}\n")
        
        # Error analysis
        f.write("\n" + "="*80 + "\n")
        f.write("ERROR ANALYSIS\n")
        f.write("-"*80 + "\n")
        
        errors = results['errors']
        f.write(f"\nTotal Errors: {len(errors)}\n")
        f.write("\nErrors by Source:\n")
        for source in sorted(errors['source'].unique()):
            count = (errors['source'] == source).sum()
            total = (results['predictions']['source'] == source).sum()
            f.write(f"  {source:15s}: {count:4d} / {total:5d} ({count/total*100:5.2f}%)\n")
    
    logger.info(f"✓ Summary report saved to: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze evaluation results')
    parser.add_argument('--results_dir', type=str, default='evaluation_results',
                       help='Directory containing evaluation results')
    args = parser.parse_args()
    
    # Load results
    logger.info(f"Loading results from {args.results_dir}...")
    results = load_results(args.results_dir)
    
    # Print analyses
    print_overall_metrics(results)
    print_domain_metrics(results)
    print_uncertainty_analysis(results)
    analyze_errors(results)
    
    # Generate outputs
    create_paper_table(results, args.results_dir)
    create_comparison_figure(results, args.results_dir)
    create_uncertainty_correlation_plot(results, args.results_dir)
    generate_summary_report(results, args.results_dir)
    
    logger.info("\n" + "="*80)
    logger.info("ANALYSIS COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nGenerated files in {args.results_dir}/:")
    logger.info("  - paper_table.tex")
    logger.info("  - performance_comparison.png")
    logger.info("  - uncertainty_correlation.png")
    logger.info("  - summary_report.txt")
    logger.info("="*80 + "\n")


if __name__ == '__main__':
    main()