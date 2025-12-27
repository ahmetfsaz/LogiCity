#!/usr/bin/env python3
"""
Analyze and visualize agent_region experiment results.
This script reads the summary file and generates plots and statistics.
"""

import sys
import argparse
from pathlib import Path
import numpy as np


def parse_summary_file(summary_file):
    """Parse the summary.txt file and return data."""
    data = {
        'agent_region': [],
        'fully': [],
        'partially': [],
        'unobserved': []
    }
    
    with open(summary_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                region, fully, partially, unobserved = parts
                data['agent_region'].append(int(region))
                data['fully'].append(float(fully))
                data['partially'].append(float(partially))
                data['unobserved'].append(float(unobserved))
    
    return data


def print_statistics(data):
    """Print statistical summary of the experiment."""
    print("\n" + "="*60)
    print("STATISTICAL SUMMARY")
    print("="*60)
    
    metrics = ['fully', 'partially', 'unobserved']
    
    for metric in metrics:
        values = np.array(data[metric])
        print(f"\n{metric.upper()}:")
        print(f"  Mean:   {np.mean(values):.2f}")
        print(f"  Median: {np.median(values):.2f}")
        print(f"  Std:    {np.std(values):.2f}")
        print(f"  Min:    {np.min(values):.2f} (agent_region={data['agent_region'][np.argmin(values)]})")
        print(f"  Max:    {np.max(values):.2f} (agent_region={data['agent_region'][np.argmax(values)]})")
    
    print("\n" + "="*60)


def print_table(data):
    """Print a nicely formatted table."""
    print("\n" + "="*70)
    print("DETAILED RESULTS TABLE")
    print("="*70)
    print(f"{'Region':>8} | {'Fully':>8} | {'Partially':>10} | {'Unobserved':>10} | {'Total':>8}")
    print("-"*70)
    
    for i in range(len(data['agent_region'])):
        region = data['agent_region'][i]
        fully = data['fully'][i]
        partially = data['partially'][i]
        unobserved = data['unobserved'][i]
        total = fully + partially + unobserved
        print(f"{region:8d} | {fully:8.2f} | {partially:10.2f} | {unobserved:10.2f} | {total:8.2f}")
    
    print("="*70)


def plot_results(data, output_dir):
    """Create plots of the results."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
    except ImportError:
        print("\nWarning: matplotlib not available. Skipping plots.")
        return
    
    regions = data['agent_region']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Agent Region Experiment Results', fontsize=16, fontweight='bold')
    
    # Plot 1: All metrics together
    ax = axes[0, 0]
    ax.plot(regions, data['fully'], 'o-', label='Fully Observed', linewidth=2, markersize=8)
    ax.plot(regions, data['partially'], 's-', label='Partially Observed', linewidth=2, markersize=8)
    ax.plot(regions, data['unobserved'], '^-', label='Unobserved', linewidth=2, markersize=8)
    ax.set_xlabel('Agent Region', fontsize=12)
    ax.set_ylabel('Average Count', fontsize=12)
    ax.set_title('Sub-rule Observability vs Agent Region', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Stacked area chart
    ax = axes[0, 1]
    ax.stackplot(regions, data['fully'], data['partially'], data['unobserved'],
                 labels=['Fully Observed', 'Partially Observed', 'Unobserved'],
                 alpha=0.7)
    ax.set_xlabel('Agent Region', fontsize=12)
    ax.set_ylabel('Average Count', fontsize=12)
    ax.set_title('Sub-rule Distribution (Stacked)', fontsize=13)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Percentage distribution
    ax = axes[1, 0]
    total = np.array(data['fully']) + np.array(data['partially']) + np.array(data['unobserved'])
    fully_pct = np.array(data['fully']) / total * 100
    partially_pct = np.array(data['partially']) / total * 100
    unobserved_pct = np.array(data['unobserved']) / total * 100
    
    ax.plot(regions, fully_pct, 'o-', label='Fully Observed %', linewidth=2, markersize=8)
    ax.plot(regions, partially_pct, 's-', label='Partially Observed %', linewidth=2, markersize=8)
    ax.plot(regions, unobserved_pct, '^-', label='Unobserved %', linewidth=2, markersize=8)
    ax.set_xlabel('Agent Region', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Sub-rule Observability (Percentage)', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Fully observed trend with error estimate
    ax = axes[1, 1]
    ax.plot(regions, data['fully'], 'o-', linewidth=2, markersize=8, color='green', label='Fully Observed')
    ax.fill_between(regions, 
                     np.array(data['fully']) * 0.95,  # Estimate ±5% error bands
                     np.array(data['fully']) * 1.05,
                     alpha=0.3, color='green')
    ax.set_xlabel('Agent Region', fontsize=12)
    ax.set_ylabel('Average Count', fontsize=12)
    ax.set_title('Fully Observed Sub-rules Trend', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = output_dir / 'experiment_plots.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\nPlots saved to: {plot_file}")
    
    plt.close()


def export_csv(data, output_dir):
    """Export results to CSV format."""
    csv_file = output_dir / 'results.csv'
    
    with open(csv_file, 'w') as f:
        # Write header
        f.write("agent_region,fully_observed,partially_observed,unobserved,total\n")
        
        # Write data
        for i in range(len(data['agent_region'])):
            region = data['agent_region'][i]
            fully = data['fully'][i]
            partially = data['partially'][i]
            unobserved = data['unobserved'][i]
            total = fully + partially + unobserved
            f.write(f"{region},{fully:.2f},{partially:.2f},{unobserved:.2f},{total:.2f}\n")
    
    print(f"\nCSV results saved to: {csv_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze agent_region experiment results')
    parser.add_argument('results_dir', type=str, help='Path to experiment results directory')
    parser.add_argument('--plot', action='store_true', help='Generate plots (requires matplotlib)')
    parser.add_argument('--csv', action='store_true', help='Export to CSV')
    
    args = parser.parse_args()
    
    # Parse results
    results_dir = Path(args.results_dir)
    summary_file = results_dir / 'summary.txt'
    
    if not summary_file.exists():
        print(f"Error: Summary file not found: {summary_file}")
        sys.exit(1)
    
    print(f"Reading results from: {summary_file}")
    data = parse_summary_file(summary_file)
    
    if not data['agent_region']:
        print("Error: No data found in summary file")
        sys.exit(1)
    
    print(f"Found {len(data['agent_region'])} agent_region configurations")
    
    # Print results
    print_table(data)
    print_statistics(data)
    
    # Generate plots if requested
    if args.plot:
        plot_results(data, results_dir)
    
    # Export CSV if requested
    if args.csv:
        export_csv(data, results_dir)
    
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()

