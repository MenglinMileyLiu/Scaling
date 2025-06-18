#!/usr/bin/env python3
# create_visualizations.py - Standalone visualization script for existing pipeline results

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import os

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_visualizations_from_files():
    """Create comprehensive visualizations from existing CSV files"""
    
    # Define the latest timestamp based on the files we found
    latest_timestamp = "20250618_093621"
    
    print(f"📂 Loading results from timestamp: {latest_timestamp}")
    
    # Load multidimensional scores
    scores_df = pd.read_csv(f"multidimensional_scores_{latest_timestamp}.csv")
    scores = {}
    for dimension in scores_df['dimension'].unique():
        dim_data = scores_df[scores_df['dimension'] == dimension]
        scores[dimension] = dict(zip(dim_data['legislator_id'], dim_data['score']))
    
    # Load overall scores
    overall_df = pd.read_csv(f"overall_ideology_scores_{latest_timestamp}.csv")
    overall_scores = dict(zip(overall_df['legislator_id'], overall_df['overall_ideology_score']))
    
    # Load dimension weights
    dimension_weights = {}
    weights_file = f"dimension_weights_{latest_timestamp}.csv"
    if os.path.exists(weights_file):
        weights_df = pd.read_csv(weights_file)
        dimension_weights = dict(zip(weights_df['dimension'], weights_df['weight']))
    
    # Load week-to-week changes
    changes = None
    changes_file = f"week_to_week_changes_{latest_timestamp}.csv"
    if os.path.exists(changes_file):
        changes_df = pd.read_csv(changes_file)
        changes = changes_df.to_dict('records')
    
    print(f"✅ Loaded:")
    print(f"   - {len(scores)} policy dimensions")
    print(f"   - {len(overall_scores)} legislators")
    print(f"   - {len(dimension_weights)} dimension weights")
    if changes:
        print(f"   - {len(changes)} week-to-week comparisons")
    
    # Create comprehensive visualization
    if changes:
        create_two_week_visualization(scores, dimension_weights, overall_scores, changes, latest_timestamp)
    else:
        create_single_week_visualization(scores, dimension_weights, overall_scores, latest_timestamp)

def create_two_week_visualization(scores, dimension_weights, overall_scores, changes, timestamp):
    """Create two-week comparison visualization"""
    
    print(f"\n📊 CREATING TWO-WEEK VISUALIZATIONS...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Overall Ideology Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    if overall_scores:
        scores_list = list(overall_scores.values())
        ax1.hist(scores_list, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(np.mean(scores_list), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(scores_list):.3f}')
        ax1.set_xlabel('Ideology Score (Lower=Liberal, Higher=Conservative)')
        ax1.set_ylabel('Number of Legislators')
        ax1.set_title('Overall Ideology Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. Dimension Weights Bar Chart
    ax2 = fig.add_subplot(gs[0, 1])
    if dimension_weights:
        dimensions = list(dimension_weights.keys())
        weights = list(dimension_weights.values())
        colors = sns.color_palette("viridis", len(dimensions))
        
        bars = ax2.bar(range(len(dimensions)), weights, color=colors)
        ax2.set_xlabel('Policy Dimensions')
        ax2.set_ylabel('Importance Weight')
        ax2.set_title('Policy Dimension Importance')
        ax2.set_xticks(range(len(dimensions)))
        ax2.set_xticklabels(dimensions, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, weight) in enumerate(zip(bars, weights)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{weight:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Liberal vs Conservative Ranking
    ax3 = fig.add_subplot(gs[0, 2])
    if overall_scores:
        sorted_scores = sorted(overall_scores.items(), key=lambda x: x[1])
        legislators = [item[0] for item in sorted_scores]
        scores_values = [item[1] for item in sorted_scores]
        
        # Color code: blue for liberal, red for conservative, gray for moderate
        colors = ['blue' if score < -0.3 else 'red' if score > 0.3 else 'gray' for score in scores_values]
        
        y_pos = np.arange(len(legislators))
        ax3.barh(y_pos, scores_values, color=colors, alpha=0.7)
        ax3.set_yticks(y_pos[::max(1, len(legislators)//10)])
        ax3.set_yticklabels([legislators[i] for i in range(0, len(legislators), max(1, len(legislators)//10))], 
                           fontsize=8)
        ax3.set_xlabel('Ideology Score')
        ax3.set_title('Liberal ← → Conservative Ranking')
        ax3.axvline(0, color='black', linestyle='-', alpha=0.5)
        ax3.grid(True, alpha=0.3)
    
    # 4. Multi-dimensional Heatmap
    ax4 = fig.add_subplot(gs[1, :2])
    if scores:
        # Create matrix for heatmap
        all_legislators = set()
        for dim_scores in scores.values():
            all_legislators.update(dim_scores.keys())
        
        all_legislators = sorted(list(all_legislators))
        dimensions = list(scores.keys())
        
        matrix = np.full((len(all_legislators), len(dimensions)), np.nan)
        
        for j, dimension in enumerate(dimensions):
            for i, legislator in enumerate(all_legislators):
                if legislator in scores[dimension]:
                    matrix[i, j] = scores[dimension][legislator]
        
        im = ax4.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)
        ax4.set_xticks(range(len(dimensions)))
        ax4.set_xticklabels(dimensions, rotation=45, ha='right')
        ax4.set_yticks(range(0, len(all_legislators), max(1, len(all_legislators)//15)))
        ax4.set_yticklabels([all_legislators[i] for i in range(0, len(all_legislators), 
                           max(1, len(all_legislators)//15))], fontsize=8)
        ax4.set_title('Multi-dimensional Ideology Heatmap\\n(Blue=Liberal, Red=Conservative)')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
        cbar.set_label('Ideology Score')
    
    # 5. Dimension Correlation
    ax5 = fig.add_subplot(gs[1, 2])
    if len(scores) > 1:
        # Create correlation matrix between dimensions
        dimension_data = {}
        shared_legislators = set.intersection(*[set(dim_scores.keys()) for dim_scores in scores.values()])
        
        for dimension, dim_scores in scores.items():
            dimension_data[dimension] = [dim_scores[leg] for leg in shared_legislators if leg in dim_scores]
        
        if len(dimension_data) > 1 and all(len(data) > 1 for data in dimension_data.values()):
            corr_df = pd.DataFrame(dimension_data)
            correlation_matrix = corr_df.corr()
            
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, ax=ax5, cbar_kws={'shrink': 0.8})
            ax5.set_title('Dimension Correlations')
        else:
            ax5.text(0.5, 0.5, 'Insufficient data\\nfor correlation analysis', 
                    ha='center', va='center', transform=ax5.transAxes, fontsize=12)
            ax5.set_title('Dimension Correlations')
    
    # 6. Week-to-Week Changes
    if changes:
        ax6 = fig.add_subplot(gs[2, :])
        
        legislators = [c['legislator_id'] for c in changes]
        week1_scores = [c['week1_score'] for c in changes]
        week2_scores = [c['week2_score'] for c in changes]
        score_changes = [c['change'] for c in changes]
        
        # Create connected scatter plot
        x_pos = np.arange(len(legislators))
        
        # Plot week 1 and week 2 scores
        ax6.scatter(x_pos, week1_scores, color='blue', alpha=0.7, s=50, label='Week 1')
        ax6.scatter(x_pos, week2_scores, color='red', alpha=0.7, s=50, label='Week 2')
        
        # Draw lines connecting the scores
        for i in range(len(legislators)):
            color = 'green' if score_changes[i] < -0.1 else 'orange' if score_changes[i] > 0.1 else 'gray'
            ax6.plot([x_pos[i], x_pos[i]], [week1_scores[i], week2_scores[i]], 
                    color=color, alpha=0.6, linewidth=1)
        
        ax6.set_xlabel('Legislators')
        ax6.set_ylabel('Ideology Score')
        ax6.set_title('Week-to-Week Ideology Changes\\n(Green=More Liberal, Orange=More Conservative, Gray=Stable)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # Only show some x-labels to avoid crowding
        step = max(1, len(legislators) // 15)
        ax6.set_xticks(x_pos[::step])
        ax6.set_xticklabels([legislators[i] for i in range(0, len(legislators), step)], 
                           rotation=45, ha='right', fontsize=8)
    
    # Save the visualization
    viz_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"ideology_analysis_plots_{viz_timestamp}.png"
    plt.suptitle(f"Political Ideology Analysis Results\\nData from: {timestamp}", fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"💾 Main visualization saved to: {plot_filename}")
    
    # Create individual detailed plots
    create_individual_plots(scores, dimension_weights, overall_scores, changes, viz_timestamp)
    
    return plot_filename

def create_individual_plots(scores, dimension_weights, overall_scores, changes, timestamp):
    """Create individual detailed plots"""
    
    # Individual plot 1: Ideology Distribution
    if overall_scores:
        plt.figure(figsize=(10, 6))
        scores_list = list(overall_scores.values())
        plt.hist(scores_list, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
        plt.axvline(np.mean(scores_list), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(scores_list):.3f}')
        plt.axvline(np.median(scores_list), color='green', linestyle='--', 
                   label=f'Median: {np.median(scores_list):.3f}')
        plt.xlabel('Ideology Score (Lower=Liberal, Higher=Conservative)', fontsize=12)
        plt.ylabel('Number of Legislators', fontsize=12)
        plt.title('Distribution of Overall Ideology Scores', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"ideology_distribution_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Individual plot 2: Week-to-week changes detail
    if changes:
        plt.figure(figsize=(12, 8))
        
        # Sort by magnitude of change
        changes_sorted = sorted(changes, key=lambda x: abs(x['change']), reverse=True)
        
        legislators = [c['legislator_id'] for c in changes_sorted[:20]]  # Top 20 changes
        score_changes = [c['change'] for c in changes_sorted[:20]]
        
        colors = ['red' if change > 0.1 else 'blue' if change < -0.1 else 'gray' for change in score_changes]
        
        bars = plt.bar(range(len(legislators)), score_changes, color=colors, alpha=0.7)
        plt.xlabel('Legislators', fontsize=12)
        plt.ylabel('Ideology Change (Week 2 - Week 1)', fontsize=12)
        plt.title('Biggest Week-to-Week Ideology Changes\\n(Red=More Conservative, Blue=More Liberal)', fontsize=14)
        plt.xticks(range(len(legislators)), legislators, rotation=45, ha='right')
        plt.axhline(0, color='black', linestyle='-', alpha=0.5)
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, change in zip(bars, score_changes):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + (0.01 if height >= 0 else -0.01), 
                    f'{change:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f"week_to_week_changes_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"💾 Individual plots saved: ideology_distribution_{timestamp}.png" + 
          (f", week_to_week_changes_{timestamp}.png" if changes else ""))

if __name__ == "__main__":
    print("🎨 CREATING VISUALIZATIONS FROM EXISTING RESULTS")
    print("=" * 60)
    
    try:
        create_visualizations_from_files()
        print("\n✅ ALL VISUALIZATIONS CREATED SUCCESSFULLY!")
        print("📊 Check the generated PNG files for detailed analysis")
        
    except Exception as e:
        print(f"\n❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc()