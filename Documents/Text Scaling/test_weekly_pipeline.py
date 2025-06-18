# test_weekly_pipeline.py - Test the weekly multi-dimensional pipeline

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from weekly_multidimensional_pipeline import (
    MultiDimensionalLLMClient, 
    run_weekly_multidimensional_pipeline
)

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_visualizations(scores, dimension_weights, overall_scores, timestamp, TWO_WEEK_MODE=False, changes=None):
    """Create comprehensive visualizations of the pipeline results"""
    
    print(f"\n📊 CREATING VISUALIZATIONS...")
    
    # Create figure with subplots
    if TWO_WEEK_MODE and changes:
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    else:
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Overall Ideology Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    if overall_scores:
        scores_list = list(overall_scores.values())
        ax1.hist(scores_list, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(np.mean(scores_list), color='red', linestyle='--', label=f'Mean: {np.mean(scores_list):.3f}')
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
        ax3.set_yticks(y_pos[::max(1, len(legislators)//10)])  # Show every nth label to avoid crowding
        ax3.set_yticklabels([legislators[i] for i in range(0, len(legislators), max(1, len(legislators)//10))], fontsize=8)
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
        ax4.set_yticklabels([all_legislators[i] for i in range(0, len(all_legislators), max(1, len(all_legislators)//15))], fontsize=8)
        ax4.set_title('Multi-dimensional Ideology Heatmap\n(Blue=Liberal, Red=Conservative)')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
        cbar.set_label('Ideology Score')
    
    # 5. Dimension Correlation (if multiple dimensions)
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
            ax5.text(0.5, 0.5, 'Insufficient data\nfor correlation analysis', 
                    ha='center', va='center', transform=ax5.transAxes, fontsize=12)
            ax5.set_title('Dimension Correlations')
    
    # 6. Week-to-Week Changes (only for two-week mode)
    if TWO_WEEK_MODE and changes:
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
        ax6.set_title('Week-to-Week Ideology Changes\n(Green=More Liberal, Orange=More Conservative, Gray=Stable)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # Only show some x-labels to avoid crowding
        step = max(1, len(legislators) // 15)
        ax6.set_xticks(x_pos[::step])
        ax6.set_xticklabels([legislators[i] for i in range(0, len(legislators), step)], 
                           rotation=45, ha='right', fontsize=8)
    
    # Save the visualization
    plot_filename = f"ideology_analysis_plots_{timestamp}.png"
    plt.suptitle(f"Political Ideology Analysis Results\n{timestamp}", fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"💾 Visualizations saved to: {plot_filename}")
    
    # Also save individual plots for detailed viewing
    save_individual_plots(scores, dimension_weights, overall_scores, timestamp, TWO_WEEK_MODE, changes)
    
    return plot_filename

def save_individual_plots(scores, dimension_weights, overall_scores, timestamp, TWO_WEEK_MODE, changes):
    """Save individual plots for detailed analysis"""
    
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
    
    # Individual plot 2: Week-to-week changes detail (if applicable)
    if TWO_WEEK_MODE and changes:
        plt.figure(figsize=(12, 8))
        
        # Sort by magnitude of change
        changes_sorted = sorted(changes, key=lambda x: abs(x['change']), reverse=True)
        
        legislators = [c['legislator_id'] for c in changes_sorted[:20]]  # Top 20 changes
        score_changes = [c['change'] for c in changes_sorted[:20]]
        
        colors = ['red' if change > 0.1 else 'blue' if change < -0.1 else 'gray' for change in score_changes]
        
        bars = plt.bar(range(len(legislators)), score_changes, color=colors, alpha=0.7)
        plt.xlabel('Legislators', fontsize=12)
        plt.ylabel('Ideology Change (Week 2 - Week 1)', fontsize=12)
        plt.title('Biggest Week-to-Week Ideology Changes\n(Red=More Conservative, Blue=More Liberal)', fontsize=14)
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
          (f", week_to_week_changes_{timestamp}.png" if TWO_WEEK_MODE and changes else ""))

# === Load API Key ===
load_dotenv("/Users/menglinliu/Documents/JoshuaClinton/emotion_pipeline/.env")
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found in environment.")

# === Load Dataset ===
print("📂 Loading congressional speech dataset...")
df = pd.read_csv("/Users/menglinliu/Documents/Text Scaling/congress_demo.csv")
print(f"📊 Total speeches in dataset: {len(df)}")

# === Load CF Scores for Validation ===
print("📂 Loading Bonica CF scores for validation...")
cf_df = pd.read_csv("/Users/menglinliu/Documents/Text Scaling/cfscore_federal_demo.csv")
print(f"📊 Total CF score records: {len(cf_df)}")

# Create a mapping of legislator to CF score (use most recent score per legislator)
cf_scores = {}
cf_parties = {}
for _, row in cf_df.iterrows():
    legislator_id = row['bonica.rid']
    cf_score = row['recipient.cfscore']
    party = row['party']
    if not pd.isna(cf_score) and legislator_id not in cf_scores:
        cf_scores[legislator_id] = cf_score
        cf_parties[legislator_id] = party

print(f"📊 Unique legislators with CF scores: {len(cf_scores)}")

# === Quick Data Verification ===
print("\n🔬 DATASET VERIFICATION")
print("-" * 30)
df['date'] = pd.to_datetime(df['date'])
min_date = df['date'].min()
max_date = df['date'].max()
print(f"📅 Date range: {min_date.date()} to {max_date.date()}")
print(f"👥 Unique legislators: {df['bonica.rid'].nunique()}")

# Test week selection
import random
random.seed(42)  # Reproducible results
random_date = min_date + pd.Timedelta(days=random.randint(0, (max_date - min_date).days))
week_start = random_date - pd.Timedelta(days=random_date.weekday())
week_end = week_start + pd.Timedelta(days=6)

print(f"🎯 Sample week: {week_start.date()} to {week_end.date()}")

# Check week data
week_data = df[(df['date'] >= week_start) & (df['date'] <= week_end)]
print(f"📊 Speeches in sample week: {len(week_data)}")
print(f"👥 Legislators in sample week: {week_data['bonica.rid'].nunique()}")

if len(week_data) == 0:
    print("⚠️ No speeches found in selected week, trying different approach...")
    # Use first week with data
    df_sorted = df.sort_values('date')
    first_week_data = df_sorted.head(50)  # Use first 50 speeches as "week"
    print(f"📊 Using first 50 speeches as sample week")
    print(f"👥 Legislators in sample: {first_week_data['bonica.rid'].nunique()}")
else:
    print("✅ Sample week has sufficient data")

# === Initialize LLM Client ===
client = MultiDimensionalLLMClient(provider="openai", api_key=api_key, model="gpt-4")

# === Test Mode Selection ===
print(f"\n🎛️ PIPELINE OPTIONS:")
print("1. Quick test mode (no API calls)")
print("2. Full pipeline with LLM analysis")
print("3. Visualize existing results (no API calls)")

# For this demo, let's use existing results for CF score comparison
TEST_MODE = "full"  # Run complete pipeline with LLM calls
TWO_WEEK_MODE = True  # Analyze same legislators across two weeks
VISUALIZE_EXISTING = True  # Use existing results instead of running new analysis

def find_consecutive_weeks_with_shared_legislators(df, min_legislators=10):
    """Find two consecutive weeks with shared legislators"""
    df['date'] = pd.to_datetime(df['date'])
    df_sorted = df.sort_values('date')
    
    # Create week boundaries
    df_sorted['week_start'] = df_sorted['date'] - pd.to_timedelta(df_sorted['date'].dt.dayofweek, unit='D')
    weeks = df_sorted['week_start'].unique()
    
    best_overlap = 0
    best_weeks = None
    
    for i in range(len(weeks) - 1):
        week1_start = weeks[i]
        week2_start = weeks[i + 1]
        
        if (week2_start - week1_start).days == 7:  # Consecutive weeks
            week1_end = week1_start + pd.Timedelta(days=6)
            week2_end = week2_start + pd.Timedelta(days=6)
            
            week1_data = df[(df['date'] >= week1_start) & (df['date'] <= week1_end)]
            week2_data = df[(df['date'] >= week2_start) & (df['date'] <= week2_end)]
            
            week1_legislators = set(week1_data['bonica.rid'].unique())
            week2_legislators = set(week2_data['bonica.rid'].unique())
            shared_legislators = week1_legislators & week2_legislators
            
            if len(shared_legislators) > best_overlap and len(shared_legislators) >= min_legislators:
                best_overlap = len(shared_legislators)
                best_weeks = {
                    'week1_start': week1_start.strftime('%Y-%m-%d'),
                    'week1_end': week1_end.strftime('%Y-%m-%d'),
                    'week2_start': week2_start.strftime('%Y-%m-%d'),
                    'week2_end': week2_end.strftime('%Y-%m-%d'),
                    'shared_legislators': list(shared_legislators),
                    'overlap_count': len(shared_legislators)
                }
    
    return best_weeks

def load_existing_results():
    """Load results from existing CSV files"""
    import glob
    import os
    
    # First try to find the specific full results you mentioned
    target_timestamp = "20250618_151143"
    
    # Check if your specific files exist
    specific_overall_file = f"overall_ideology_scores_full_{target_timestamp}.csv"
    specific_scores_file = f"multidimensional_scores_full_{target_timestamp}.csv"
    
    if os.path.exists(specific_overall_file):
        latest_timestamp = target_timestamp
        print(f"📂 Using your specific results from: {latest_timestamp}")
    else:
        # Fall back to finding the most recent full results
        score_files = glob.glob("multidimensional_scores_full_*.csv") + glob.glob("multidimensional_scores_*.csv")
        overall_files = glob.glob("overall_ideology_scores_full_*.csv") + glob.glob("overall_ideology_scores_*.csv")
        weight_files = glob.glob("dimension_weights_full_*.csv") + glob.glob("dimension_weights_*.csv")
        change_files = glob.glob("week_to_week_changes_full_*.csv") + glob.glob("week_to_week_changes_*.csv")
        
        if not score_files or not overall_files:
            print("❌ No result files found. Please run the pipeline first.")
            return None
        
        # Get the most recent timestamp
        timestamps = []
        for file in score_files:
            if "full_" in file:
                timestamp = file.split('_full_')[-1].replace('.csv', '')
            else:
                timestamp = file.split('_')[-1].replace('.csv', '')
            if len(timestamp) == 15:  # YYYYMMDD_HHMMSS format
                timestamps.append(timestamp)
        
        if not timestamps:
            print("❌ No valid timestamp found in result files.")
            return None
        
        latest_timestamp = max(timestamps)
        print(f"📂 Found most recent results from: {latest_timestamp}")
    
    # Try to load files with "full" prefix first, then fall back to regular naming
    def try_load_file(base_name, timestamp):
        full_name = f"{base_name}_full_{timestamp}.csv"
        regular_name = f"{base_name}_{timestamp}.csv"
        
        if os.path.exists(full_name):
            return pd.read_csv(full_name), full_name
        elif os.path.exists(regular_name):
            return pd.read_csv(regular_name), regular_name
        else:
            return None, None
    
    # Load multidimensional scores
    scores_df, scores_filename = try_load_file("multidimensional_scores", latest_timestamp)
    if scores_df is None:
        print("❌ Could not find multidimensional scores file")
        return None
    
    scores = {}
    for dimension in scores_df['dimension'].unique():
        dim_data = scores_df[scores_df['dimension'] == dimension]
        scores[dimension] = dict(zip(dim_data['legislator_id'], dim_data['score']))
    
    # Load overall scores
    overall_df, overall_filename = try_load_file("overall_ideology_scores", latest_timestamp)
    if overall_df is None:
        print("❌ Could not find overall ideology scores file")
        return None
    
    overall_scores = dict(zip(overall_df['legislator_id'], overall_df['overall_ideology_score']))
    
    # Load dimension weights
    dimension_weights = {}
    weights_df, weights_filename = try_load_file("dimension_weights", latest_timestamp)
    if weights_df is not None:
        dimension_weights = dict(zip(weights_df['dimension'], weights_df['weight']))
    
    # Load week-to-week changes if available
    changes = None
    changes_df, changes_filename = try_load_file("week_to_week_changes", latest_timestamp)
    if changes_df is not None:
        changes = changes_df.to_dict('records')
    
    print(f"✅ Loaded:")
    print(f"   - {len(scores)} policy dimensions")
    print(f"   - {len(overall_scores)} legislators")
    print(f"   - {len(dimension_weights)} dimension weights")
    if changes:
        print(f"   - {len(changes)} week-to-week comparisons")
    
    return scores, overall_scores, dimension_weights, changes, latest_timestamp

def compare_with_cf_scores(overall_scores, cf_scores, cf_parties, timestamp):
    """Compare pipeline results with Bonica CF scores"""
    print(f"\n🔍 VALIDATION: Comparing with Bonica CF Scores")
    print("=" * 60)
    
    # Find legislators with both scores
    common_legislators = set(overall_scores.keys()) & set(cf_scores.keys())
    
    if len(common_legislators) < 3:
        print(f"❌ Only {len(common_legislators)} legislators have both scores - insufficient for analysis")
        return
    
    print(f"✅ Found {len(common_legislators)} legislators with both pipeline and CF scores")
    
    # Extract matching scores
    pipeline_scores = [overall_scores[leg_id] for leg_id in common_legislators]
    bonica_scores = [cf_scores[leg_id] for leg_id in common_legislators]
    parties = [cf_parties.get(leg_id, 'Unknown') for leg_id in common_legislators]
    
    # Calculate correlation
    import numpy as np
    correlation = np.corrcoef(pipeline_scores, bonica_scores)[0, 1]
    
    print(f"📊 Correlation between pipeline and CF scores: {correlation:.3f}")
    
    # Analyze by party
    party_analysis = {}
    for leg_id in common_legislators:
        party = cf_parties.get(leg_id, 'Unknown')
        if party not in party_analysis:
            party_analysis[party] = {'pipeline': [], 'cf': [], 'legislators': []}
        party_analysis[party]['pipeline'].append(overall_scores[leg_id])
        party_analysis[party]['cf'].append(cf_scores[leg_id])
        party_analysis[party]['legislators'].append(leg_id)
    
    print(f"\n📈 Analysis by Party:")
    for party, data in party_analysis.items():
        if len(data['pipeline']) > 2:  # Only analyze parties with 3+ members
            party_corr = np.corrcoef(data['pipeline'], data['cf'])[0, 1]
            avg_pipeline = np.mean(data['pipeline'])
            avg_cf = np.mean(data['cf'])
            print(f"  {party}: {len(data['legislators'])} legislators, correlation = {party_corr:.3f}")
            print(f"      Avg Pipeline Score: {avg_pipeline:.3f}, Avg CF Score: {avg_cf:.3f}")
    
    # Show detailed comparison for top/bottom legislators
    print(f"\n📋 Detailed Comparison (sorted by pipeline score):")
    comparison_data = []
    for leg_id in common_legislators:
        comparison_data.append({
            'legislator_id': leg_id,
            'pipeline_score': overall_scores[leg_id],
            'cf_score': cf_scores[leg_id],
            'party': cf_parties.get(leg_id, 'Unknown'),
            'pipeline_rank': 0,  # Will be filled after sorting
            'cf_rank': 0
        })
    
    # Sort by pipeline score and assign ranks
    comparison_data.sort(key=lambda x: x['pipeline_score'])
    for i, item in enumerate(comparison_data):
        item['pipeline_rank'] = i + 1
    
    # Sort by CF score and assign ranks
    comparison_data.sort(key=lambda x: x['cf_score'])
    for i, item in enumerate(comparison_data):
        item['cf_rank'] = i + 1
    
    # Sort back by pipeline score for display
    comparison_data.sort(key=lambda x: x['pipeline_score'])
    
    print("Legislator | Pipeline Score | CF Score | Party | Pipeline Rank | CF Rank | Difference")
    print("-" * 85)
    
    for i, data in enumerate(comparison_data):
        rank_diff = abs(data['pipeline_rank'] - data['cf_rank'])
        print(f"{data['legislator_id']:>10} | {data['pipeline_score']:>13.3f} | {data['cf_score']:>8.3f} | {data['party']:>5} | {data['pipeline_rank']:>13} | {data['cf_rank']:>7} | {rank_diff:>10}")
    
    # Calculate average rank difference
    avg_rank_diff = np.mean([abs(data['pipeline_rank'] - data['cf_rank']) for data in comparison_data])
    max_possible_diff = len(comparison_data) / 2
    rank_agreement = 1 - (avg_rank_diff / max_possible_diff)
    
    print(f"\n📊 Ranking Agreement:")
    print(f"Average rank difference: {avg_rank_diff:.1f} positions")
    print(f"Ranking agreement score: {rank_agreement:.3f} (1.0 = perfect agreement)")
    
    # Save comparison results
    comparison_df = pd.DataFrame(comparison_data)
    filename = f"cf_score_comparison_full_{timestamp}.csv"
    comparison_df.to_csv(filename, index=False)
    print(f"💾 CF score comparison saved to: {filename}")
    
    # Interpretation
    print(f"\n🔍 VALIDATION INTERPRETATION:")
    if correlation > 0.5:
        print(f"✅ Strong positive correlation ({correlation:.3f}) - Pipeline captures ideological dimension well")
    elif correlation > 0.3:
        print(f"✅ Moderate positive correlation ({correlation:.3f}) - Pipeline captures some ideological signal")
    elif correlation > 0.1:
        print(f"⚠️ Weak positive correlation ({correlation:.3f}) - Pipeline captures limited ideological signal")
    else:
        print(f"❌ Poor correlation ({correlation:.3f}) - Pipeline may not capture traditional ideology")
    
    if rank_agreement > 0.7:
        print(f"✅ High ranking agreement ({rank_agreement:.3f}) - Similar relative ordering")
    elif rank_agreement > 0.5:
        print(f"⚠️ Moderate ranking agreement ({rank_agreement:.3f}) - Some ordering differences")
    else:
        print(f"❌ Low ranking agreement ({rank_agreement:.3f}) - Different ordering of legislators")
    
    return correlation, rank_agreement, comparison_data

if VISUALIZE_EXISTING:
    print("\n🎨 VISUALIZING EXISTING RESULTS")
    print("=" * 60)
    
    # Load existing results
    result = load_existing_results()
    if result:
        scores, overall_scores, dimension_weights, changes, data_timestamp = result
        
        # Create visualizations
        viz_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if changes:
            create_visualizations(scores, dimension_weights, overall_scores, viz_timestamp, 
                                TWO_WEEK_MODE=True, changes=changes)
            print(f"\n✅ TWO-WEEK VISUALIZATIONS CREATED!")
            print(f"📊 Analyzed {len(changes)} legislators across two weeks")
        else:
            create_visualizations(scores, dimension_weights, overall_scores, viz_timestamp, 
                                TWO_WEEK_MODE=False, changes=None)
            print(f"\n✅ SINGLE-WEEK VISUALIZATIONS CREATED!")
            print(f"📊 Analyzed {len(overall_scores)} legislators")
        
        # Also run CF score comparison on existing results
        correlation, rank_agreement, comparison_data = compare_with_cf_scores(overall_scores, cf_scores, cf_parties, viz_timestamp)
        
        print(f"📈 Data source: Results from {data_timestamp}")
        print(f"🎨 Visualizations timestamp: {viz_timestamp}")

elif TEST_MODE == "quick":
    if TWO_WEEK_MODE:
        print("\n🔬 RUNNING TWO-WEEK STRUCTURAL TEST")
        print("=" * 60)
        
        # Find consecutive weeks with shared legislators
        week_info = find_consecutive_weeks_with_shared_legislators(df, min_legislators=30)
        
        if week_info:
            print(f"✅ Found consecutive weeks with {week_info['overlap_count']} shared legislators")
            print(f"📅 Week 1: {week_info['week1_start']} to {week_info['week1_end']}")
            print(f"📅 Week 2: {week_info['week2_start']} to {week_info['week2_end']}")
            
            # Test both weeks
            from weekly_multidimensional_pipeline import aggregate_weekly_speeches
            
            week1_speeches = aggregate_weekly_speeches(df, week_info['week1_start'], week_info['week1_end'])
            week2_speeches = aggregate_weekly_speeches(df, week_info['week2_start'], week_info['week2_end'])
            
            # Filter to shared legislators only
            shared_legislators = set(week_info['shared_legislators'][:30])
            week1_filtered = [s for s in week1_speeches if s.legislator_id in shared_legislators]
            week2_filtered = [s for s in week2_speeches if s.legislator_id in shared_legislators]
            
            print(f"\n📊 TWO-WEEK COMPARISON RESULTS:")
            print(f"Week 1: {len(week1_filtered)} legislators with speeches")
            print(f"Week 2: {len(week2_filtered)} legislators with speeches")
            print(f"Shared: {len(shared_legislators)} legislators")
            
            print(f"\n📋 SAMPLE WEEKLY AGGREGATIONS (Week 1):")
            for i, speech in enumerate(week1_filtered[:5], 1):
                print(f"  {i}. {speech.legislator_id}: {speech.total_speeches} speeches, {len(speech.concatenated_text):,} chars")
            
            print(f"\n📋 SAMPLE WEEKLY AGGREGATIONS (Week 2):")
            for i, speech in enumerate(week2_filtered[:5], 1):
                print(f"  {i}. {speech.legislator_id}: {speech.total_speeches} speeches, {len(speech.concatenated_text):,} chars")
            
            print(f"\n🎯 To run full two-week pipeline, change TEST_MODE to 'full'")
            print(f"📝 This will analyze {len(shared_legislators)} legislators across both weeks")
            
            # Save two-week test results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            two_week_results = {
                "test_type": "two_week_structural_test",
                "timestamp": timestamp,
                "week_info": week_info,
                "week1_legislators": len(week1_filtered),
                "week2_legislators": len(week2_filtered),
                "shared_legislators_count": len(shared_legislators),
                "shared_legislators": list(shared_legislators)
            }
            
            with open(f"two_week_test_results_{timestamp}.json", "w") as f:
                json.dump(two_week_results, f, indent=2, default=str)
            
            print(f"\n💾 Two-week test results saved to: two_week_test_results_{timestamp}.json")
            
        else:
            print("❌ Could not find consecutive weeks with 30+ shared legislators")
            print("🔄 Falling back to single week analysis...")
            TWO_WEEK_MODE = False
    
    if not TWO_WEEK_MODE:
        print("\n🔬 RUNNING QUICK STRUCTURAL TEST")
        print("=" * 60)
        
        # Test the data processing components without API calls
        from weekly_multidimensional_pipeline import select_random_week, aggregate_weekly_speeches
        
        # Test week selection
        week_start, week_end = select_random_week(df)
        weekly_speeches = aggregate_weekly_speeches(df, week_start, week_end)
        
        print(f"\n📊 STRUCTURAL TEST RESULTS:")
        print(f"✅ Week selection: {week_start} to {week_end}")
        print(f"✅ Speech aggregation: {len(weekly_speeches)} legislators")
    
        if len(weekly_speeches) > 0:
            sample = weekly_speeches[0]
            print(f"✅ Sample aggregation:")
            print(f"   - Legislator: {sample.legislator_id}")
            print(f"   - Speeches: {sample.total_speeches}")
            print(f"   - Text length: {len(sample.concatenated_text):,} characters")
            print(f"   - Preview: {sample.concatenated_text[:200]}...")
        
        print(f"\n🎯 To run full pipeline with LLM calls, change TEST_MODE to 'full'")
        print(f"📝 This will analyze {min(len(weekly_speeches), 30)} legislators across multiple dimensions")
        
        # Show more details about the weekly aggregation
        print(f"\n📋 SAMPLE WEEKLY AGGREGATIONS:")
        for i, speech in enumerate(weekly_speeches[:5], 1):
            print(f"  {i}. {speech.legislator_id}: {speech.total_speeches} speeches, {len(speech.concatenated_text):,} chars")
        
        if len(weekly_speeches) > 5:
            print(f"  ... and {len(weekly_speeches) - 5} more legislators")
        
        # Save quick test results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        quick_results = {
            "test_type": "quick_structural_test",
            "timestamp": timestamp,
            "dataset_info": {
                "total_speeches": len(df),
                "date_range": f"{min_date.date()} to {max_date.date()}",
                "unique_legislators": df['bonica.rid'].nunique()
            },
            "selected_week": {
                "week_start": week_start,
                "week_end": week_end,
                "speeches_in_week": len(week_data),
                "legislators_in_week": week_data['bonica.rid'].nunique() if len(week_data) > 0 else 0
            },
            "weekly_aggregations": [
                {
                    "legislator_id": speech.legislator_id,
                    "total_speeches": speech.total_speeches,
                    "text_length": len(speech.concatenated_text),
                    "preview": speech.concatenated_text[:200] + "..."
                }
                for speech in weekly_speeches[:10]  # Save first 10
            ]
        }
        
        with open(f"quick_test_results_{timestamp}.json", "w") as f:
            json.dump(quick_results, f, indent=2, default=str)
        
        print(f"\n💾 Quick test results saved to: quick_test_results_{timestamp}.json")

else:
    if TWO_WEEK_MODE:
        # === Run Two-Week Comparison Pipeline ===
        print("\n🚀 RUNNING TWO-WEEK COMPARISON PIPELINE")
        print("=" * 60)
        
        # Find consecutive weeks
        week_info = find_consecutive_weeks_with_shared_legislators(df, min_legislators=10)  # Lower minimum threshold
        if not week_info:
            print("❌ Could not find consecutive weeks with 10+ shared legislators")
            print("🔄 Falling back to single week analysis...")
            TWO_WEEK_MODE = False
        else:
            shared_legislators = set(week_info['shared_legislators'])  # Use ALL shared legislators
            print(f"📊 FULL ANALYSIS: {len(shared_legislators)} legislators across two weeks")
            print(f"📅 Week 1: {week_info['week1_start']} to {week_info['week1_end']}")
            print(f"📅 Week 2: {week_info['week2_start']} to {week_info['week2_end']}")
    
    if not TWO_WEEK_MODE:
        # === Run Weekly Multi-Dimensional Pipeline ===
        print("\n🚀 RUNNING WEEKLY MULTI-DIMENSIONAL PIPELINE")
        print("=" * 60)

try:
    if TWO_WEEK_MODE:
        # Run pipeline for both weeks
        week_results = {}
        
        for week_num, (week_key, start_key, end_key) in enumerate([
            ('week1', 'week1_start', 'week1_end'),
            ('week2', 'week2_start', 'week2_end')
        ], 1):
            print(f"\n🗓️ PROCESSING WEEK {week_num}")
            print("-" * 40)
            
            # Get week boundaries
            week_start = week_info[start_key]
            week_end = week_info[end_key]
            
            # Run pipeline for this week (limited to shared legislators)
            from weekly_multidimensional_pipeline import aggregate_weekly_speeches
            weekly_speeches = aggregate_weekly_speeches(df, week_start, week_end)
            filtered_speeches = [s for s in weekly_speeches if s.legislator_id in shared_legislators]
            
            # Create temporary dataframe for this week's shared legislators
            week_df = df[
                (df['date'] >= week_start) & 
                (df['date'] <= week_end) & 
                (df['bonica.rid'].isin(shared_legislators))
            ].copy()
            
            # Override the pipeline's week selection to use our specific week
            original_select_week = __import__('weekly_multidimensional_pipeline').select_random_week
            def fixed_week_selection(df_param):
                return week_start, week_end
            __import__('weekly_multidimensional_pipeline').select_random_week = fixed_week_selection
            
            # Run pipeline with ALL legislators (no limit)
            scores, summaries, dimension_comparisons, dimension_weights, overall_scores = run_weekly_multidimensional_pipeline(
                week_df, client, max_legislators=len(shared_legislators)  # Use all shared legislators
            )
            
            # Restore original function
            __import__('weekly_multidimensional_pipeline').select_random_week = original_select_week
            
            # Store results
            week_results[week_key] = {
                'scores': scores,
                'summaries': summaries,
                'dimension_comparisons': dimension_comparisons,
                'dimension_weights': dimension_weights,
                'overall_scores': overall_scores,
                'week_period': f"{week_start} to {week_end}"
            }
        
        # Analyze changes between weeks
        print(f"\n🔄 WEEK-TO-WEEK IDEOLOGY CHANGES")
        print("=" * 60)
        
        week1_overall = week_results['week1']['overall_scores']
        week2_overall = week_results['week2']['overall_scores']
        
        changes = []
        for legislator in shared_legislators:
            if legislator in week1_overall and legislator in week2_overall:
                week1_score = week1_overall[legislator]
                week2_score = week2_overall[legislator]
                change = week2_score - week1_score
                
                changes.append({
                    'legislator_id': legislator,
                    'week1_score': week1_score,
                    'week2_score': week2_score,
                    'change': change,
                    'direction': 'More Conservative' if change > 0.1 else 'More Liberal' if change < -0.1 else 'Stable'
                })
        
        # Sort by magnitude of change
        changes.sort(key=lambda x: abs(x['change']), reverse=True)
        
        print("Biggest Ideological Shifts:")
        print("Legislator | Week 1 | Week 2 | Change | Direction")
        print("-" * 55)
        
        for change in changes[:15]:  # Top 15 biggest changes
            direction_symbol = "→ R" if change['change'] > 0.1 else "→ L" if change['change'] < -0.1 else "→ S"
            print(f"{change['legislator_id']:>10} | {change['week1_score']:>6.3f} | {change['week2_score']:>6.3f} | {change['change']:>6.3f} | {direction_symbol}")
        
        # Summary statistics
        avg_change = sum(abs(c['change']) for c in changes) / len(changes)
        stable_count = sum(1 for c in changes if abs(c['change']) <= 0.1)
        conservative_shift = sum(1 for c in changes if c['change'] > 0.1)
        liberal_shift = sum(1 for c in changes if c['change'] < -0.1)
        
        print(f"\n📈 CHANGE SUMMARY:")
        print(f"Average absolute change: {avg_change:.3f}")
        print(f"Stable positions (±0.1): {stable_count}/{len(changes)} ({stable_count/len(changes)*100:.1f}%)")
        print(f"Conservative shifts: {conservative_shift} legislators")
        print(f"Liberal shifts: {liberal_shift} legislators")
        
        # Use week 2 results for display (most recent)
        scores = week_results['week2']['scores']
        summaries = week_results['week2']['summaries']
        dimension_comparisons = week_results['week2']['dimension_comparisons'] 
        dimension_weights = week_results['week2']['dimension_weights']
        overall_scores = week_results['week2']['overall_scores']
        
    else:
        scores, summaries, dimension_comparisons, dimension_weights, overall_scores = run_weekly_multidimensional_pipeline(
            df, client, max_legislators=1000  # Allow many legislators for full analysis
        )
    
    # === Display Results ===
    print("\n" + "=" * 60)
    print("📊 MULTI-DIMENSIONAL IDEOLOGY SCORES")
    print("=" * 60)
    
    if scores:
        for dimension, dimension_scores in scores.items():
            print(f"\n🏷️ {dimension.upper()} DIMENSION:")
            print("-" * 40)
            
            # Sort legislators by score (high to low = more liberal/progressive)
            sorted_legislators = sorted(dimension_scores.items(), key=lambda x: x[1], reverse=True)
            
            for rank, (legislator_id, score) in enumerate(sorted_legislators, 1):
                if rank == 1:
                    label = "(Most Liberal/Progressive)"
                elif rank == len(sorted_legislators):
                    label = "(Most Conservative)"
                else:
                    label = ""
                    
                print(f"  {rank}. {legislator_id:>12} | {score:>7.3f} {label}")
    else:
        print("❌ No valid dimension scores calculated")
    
    # === Display Overall Weighted Scores ===
    print("\n🎯 OVERALL LIBERAL-CONSERVATIVE SCORES:")
    print("(Lower = More Liberal, Higher = More Conservative)")
    print("-" * 50)
    
    if overall_scores:
        sorted_overall = sorted(overall_scores.items(), key=lambda x: x[1])
        print("Rank | Legislator | Score | Ideology")
        print("-" * 45)
        for rank, (legislator_id, score) in enumerate(sorted_overall, 1):
            if rank == 1:
                ideology = "Most Liberal"
            elif rank == len(sorted_overall):
                ideology = "Most Conservative"
            elif score < -0.5:
                ideology = "Liberal"
            elif score > 0.5:
                ideology = "Conservative"
            else:
                ideology = "Moderate"
                
            print(f"  {rank:>2}.  {legislator_id:>10} | {score:>6.3f} | {ideology}")
    
    # === Display Dimension Weights ===
    print(f"\n⚖️ DIMENSION WEIGHTS:")
    if dimension_weights:
        print("Dimension | Weight | Interpretation")
        print("-" * 40)
        for dimension, weight in sorted(dimension_weights.items(), key=lambda x: x[1], reverse=True):
            if weight > 0.7:
                interp = "High importance"
            elif weight > 0.4:
                interp = "Medium importance"  
            else:
                interp = "Low importance"
            print(f"{dimension:>15} | {weight:>6.3f} | {interp}")
    
    # === Display Policy Summaries ===
    print("\n" + "=" * 60)
    print("📝 WEEKLY POLICY SUMMARIES")
    print("=" * 60)
    
    for i, summary in enumerate(summaries, 1):
        print(f"\n--- {i}. {summary.legislator_id} ---")
        print(f"Week: {summary.week_period}")
        print(f"Activity Level: {summary.overall_activity_level}")
        print("Policy Positions:")
        for dimension, stance in summary.policy_dimensions.items():
            print(f"  • {dimension}: {stance}")
    
    # === Display Sample Comparisons ===
    print("\n" + "=" * 60)
    print("⚖️ SAMPLE PAIRWISE COMPARISONS")
    print("=" * 60)
    
    for dimension, comparisons in dimension_comparisons.items():
        valid_comparisons = [c for c in comparisons if c.winner != "Skip"]
        if valid_comparisons:
            print(f"\n🏷️ {dimension} (showing first 3):")
            for comp in valid_comparisons[:3]:
                print(f"  {comp.legislator_a_id} vs {comp.legislator_b_id} → Winner: {comp.winner}")
                print(f"  Reasoning: {comp.reasoning[:100]}...")
                print()
    
    print("\n✅ FULL PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"📈 Dimensions analyzed: {len(scores)}")
    print(f"👥 Total legislators processed: {len(summaries)}")
    if TWO_WEEK_MODE and 'changes' in locals():
        print(f"🔄 Week-to-week changes tracked: {len(changes)}")
    
    # === VALIDATION: Compare with Bonica CF Scores ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    correlation, rank_agreement, comparison_data = compare_with_cf_scores(overall_scores, cf_scores, cf_parties, timestamp)
    
    # Create visualizations
    
    if TWO_WEEK_MODE and 'changes' in locals():
        create_visualizations(scores, dimension_weights, overall_scores, timestamp, 
                            TWO_WEEK_MODE=True, changes=changes)
    else:
        create_visualizations(scores, dimension_weights, overall_scores, timestamp, 
                            TWO_WEEK_MODE=False, changes=None)
    
    # Save full results to files
    
    # Save multi-dimensional scores
    scores_df_list = []
    for dimension, dimension_scores in scores.items():
        for legislator_id, score in dimension_scores.items():
            scores_df_list.append({
                "dimension": dimension,
                "legislator_id": legislator_id,
                "score": score
            })
    
    if scores_df_list:
        scores_df = pd.DataFrame(scores_df_list)
        filename = f"multidimensional_scores_full_{timestamp}.csv"
        scores_df.to_csv(filename, index=False)
        print(f"💾 Full analysis scores saved to: {filename}")
    
    # Save overall weighted scores
    if overall_scores:
        overall_df = pd.DataFrame([
            {
                "legislator_id": legislator_id,
                "overall_ideology_score": score,
                "ideology_label": (
                    "Most Liberal" if score == min(overall_scores.values()) else
                    "Most Conservative" if score == max(overall_scores.values()) else
                    "Liberal" if score < -0.5 else
                    "Conservative" if score > 0.5 else
                    "Moderate"
                )
            }
            for legislator_id, score in overall_scores.items()
        ])
        overall_df = overall_df.sort_values('overall_ideology_score')
        filename = f"overall_ideology_scores_full_{timestamp}.csv"
        overall_df.to_csv(filename, index=False)
        print(f"💾 Full analysis overall scores saved to: {filename}")
    
    # Save dimension weights
    if dimension_weights:
        weights_df = pd.DataFrame([
            {
                "dimension": dimension,
                "weight": weight,
                "importance": (
                    "High importance" if weight > 0.7 else
                    "Medium importance" if weight > 0.4 else
                    "Low importance"
                )
            }
            for dimension, weight in dimension_weights.items()
        ])
        weights_df = weights_df.sort_values('weight', ascending=False)
        filename = f"dimension_weights_full_{timestamp}.csv"
        weights_df.to_csv(filename, index=False)
        print(f"💾 Full analysis weights saved to: {filename}")
    
    # Save detailed summaries
    summaries_data = []
    for summary in summaries:
        summary_dict = {
            "legislator_id": summary.legislator_id,
            "legislator_name": summary.legislator_name,
            "week_period": summary.week_period,
            "activity_level": summary.overall_activity_level,
            "raw_analysis": summary.raw_analysis
        }
        # Add policy dimensions as separate columns
        for dimension, stance in summary.policy_dimensions.items():
            summary_dict[f"stance_{dimension}"] = stance
        summaries_data.append(summary_dict)
    
    summaries_df = pd.DataFrame(summaries_data)
    filename = f"policy_summaries_full_{timestamp}.csv"
    summaries_df.to_csv(filename, index=False)
    print(f"💾 Full analysis summaries saved to: {filename}")
    
    # Save comparisons for analysis
    comparisons_data = []
    for dimension, comps in dimension_comparisons.items():
        for comp in comps:
            comparisons_data.append({
                "dimension": dimension,
                "legislator_a": comp.legislator_a_id,
                "legislator_b": comp.legislator_b_id,
                "winner": comp.winner,
                "reasoning": comp.reasoning
            })
    
    comparisons_df = pd.DataFrame(comparisons_data)
    filename = f"pairwise_comparisons_full_{timestamp}.csv"
    comparisons_df.to_csv(filename, index=False)
    print(f"💾 Full analysis comparisons saved to: {filename}")
    
    # Save two-week comparison results if applicable
    if TWO_WEEK_MODE and 'changes' in locals():
        changes_df = pd.DataFrame(changes)
        filename = f"week_to_week_changes_full_{timestamp}.csv"
        changes_df.to_csv(filename, index=False)
        print(f"💾 Full analysis week-to-week changes saved to: {filename}")
        
        # Save both weeks' results for comparison
        week1_overall_df = pd.DataFrame([
            {"legislator_id": leg_id, "week": "Week 1", "ideology_score": score}
            for leg_id, score in week_results['week1']['overall_scores'].items()
        ])
        week2_overall_df = pd.DataFrame([
            {"legislator_id": leg_id, "week": "Week 2", "ideology_score": score}
            for leg_id, score in week_results['week2']['overall_scores'].items()
        ])
        
        both_weeks_df = pd.concat([week1_overall_df, week2_overall_df])
        filename = f"two_week_comparison_full_{timestamp}.csv"
        both_weeks_df.to_csv(filename, index=False)
        print(f"💾 Full analysis two-week comparison saved to: {filename}")
    
except Exception as e:
    print(f"\n❌ PIPELINE FAILED: {e}")
    import traceback
    traceback.print_exc()