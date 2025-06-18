#!/usr/bin/env python3
# validate_with_cf_scores.py - Quick CF score validation using existing results

import pandas as pd
import numpy as np
from datetime import datetime

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
    
    # Calculate correlation
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
    
    # Show detailed comparison
    print(f"\n📋 Detailed Comparison (sorted by pipeline score):")
    comparison_data = []
    for leg_id in common_legislators:
        comparison_data.append({
            'legislator_id': leg_id,
            'pipeline_score': overall_scores[leg_id],
            'cf_score': cf_scores[leg_id],
            'party': cf_parties.get(leg_id, 'Unknown'),
            'pipeline_rank': 0,
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
    
    for data in comparison_data:
        rank_diff = abs(data['pipeline_rank'] - data['cf_rank'])
        print(f"{data['legislator_id']:>10} | {data['pipeline_score']:>13.3f} | {data['cf_score']:>8.3f} | {data['party']:>5} | {data['pipeline_rank']:>13} | {data['cf_rank']:>7} | {rank_diff:>10}")
    
    # Calculate ranking metrics
    avg_rank_diff = np.mean([abs(data['pipeline_rank'] - data['cf_rank']) for data in comparison_data])
    max_possible_diff = len(comparison_data) / 2
    rank_agreement = 1 - (avg_rank_diff / max_possible_diff)
    
    print(f"\n📊 Ranking Agreement:")
    print(f"Average rank difference: {avg_rank_diff:.1f} positions")
    print(f"Ranking agreement score: {rank_agreement:.3f} (1.0 = perfect agreement)")
    
    # Save comparison results
    comparison_df = pd.DataFrame(comparison_data)
    filename = f"cf_score_validation_{timestamp}.csv"
    comparison_df.to_csv(filename, index=False)
    print(f"💾 CF score validation saved to: {filename}")
    
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

def main():
    print("🔍 CF SCORE VALIDATION - Using Existing Results")
    print("=" * 60)
    
    # Load your existing pipeline results
    target_timestamp = "20250618_151143"
    pipeline_file = f"overall_ideology_scores_full_{target_timestamp}.csv"
    
    try:
        pipeline_df = pd.read_csv(pipeline_file)
        print(f"✅ Loaded pipeline results: {len(pipeline_df)} legislators")
        
        # Create mapping of legislator to pipeline score
        overall_scores = dict(zip(pipeline_df['legislator_id'], pipeline_df['overall_ideology_score']))
        
    except FileNotFoundError:
        print(f"❌ Could not find file: {pipeline_file}")
        return
    
    # Load CF scores
    print("📂 Loading Bonica CF scores...")
    cf_df = pd.read_csv("cfscore_federal_demo.csv")
    print(f"📊 Total CF score records: {len(cf_df)}")
    
    # Create mapping of legislator to CF score (use most recent score per legislator)
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
    
    # Run comparison
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    correlation, rank_agreement, comparison_data = compare_with_cf_scores(overall_scores, cf_scores, cf_parties, timestamp)
    
    print(f"\n✅ VALIDATION COMPLETED!")
    print(f"📊 Final Results: Correlation = {correlation:.3f}, Rank Agreement = {rank_agreement:.3f}")

if __name__ == "__main__":
    main()