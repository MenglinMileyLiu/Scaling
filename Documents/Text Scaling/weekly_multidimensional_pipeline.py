# weekly_multidimensional_pipeline.py - Multi-dimensional weekly speech aggregation pipeline

import os
import json
import random
import itertools
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from datetime import datetime, timedelta
import pandas as pd

import openai
import choix
from dotenv import load_dotenv

# === Data Classes ===
@dataclass
class WeeklyLegislatorSpeech:
    legislator_id: str
    legislator_name: str
    week_start: str
    week_end: str
    total_speeches: int
    concatenated_text: str
    party: Optional[str] = None
    state: Optional[str] = None

@dataclass
class MultiDimensionalSummary:
    legislator_id: str
    legislator_name: str
    week_period: str
    policy_dimensions: Dict[str, str]  # dimension -> stance/summary
    overall_activity_level: str
    raw_analysis: str

@dataclass
class DimensionComparison:
    legislator_a_id: str
    legislator_b_id: str
    dimension: str
    winner: str  # 'A', 'B', or 'Skip'
    confidence: float
    reasoning: str

# === LLM Client ===
class MultiDimensionalLLMClient:
    def __init__(self, provider: str, api_key: str, model: str = "gpt-4"):
        if provider != "openai":
            raise ValueError("Only 'openai' is supported in this version.")
        self.model = model
        self.client = openai.OpenAI(api_key=api_key)

    def extract_policy_dimensions(self, speech: WeeklyLegislatorSpeech) -> MultiDimensionalSummary:
        """Extract multiple policy dimensions from concatenated weekly speeches"""
        prompt = (
            f"Analyze this legislator's complete speeches from one week. "
            f"Identify the main policy areas they addressed and their positions.\n\n"
            f"Legislator: {speech.legislator_name}\n"
            f"Week: {speech.week_start} to {speech.week_end}\n"
            f"Total speeches: {speech.total_speeches}\n\n"
            f"Concatenated speeches:\n\"\"\"{speech.concatenated_text[:3000]}...\"\"\"\n\n"
            f"Please provide:\n"
            f"1. POLICY DIMENSIONS: List 3-5 main policy areas they discussed (e.g., Healthcare, Defense, Environment)\n"
            f"2. POSITIONS: For each dimension, summarize their stance in 1-2 sentences\n"
            f"3. ACTIVITY LEVEL: High/Medium/Low based on engagement depth\n\n"
            f"Format as:\n"
            f"DIMENSIONS:\n"
            f"- Healthcare: [stance]\n"
            f"- Environment: [stance]\n"
            f"ACTIVITY: [level]\n"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        content = response.choices[0].message.content.strip()

        # Parse dimensions from response
        policy_dimensions = self._parse_dimensions(content)
        activity_level = self._parse_activity_level(content)

        return MultiDimensionalSummary(
            legislator_id=speech.legislator_id,
            legislator_name=speech.legislator_name,
            week_period=f"{speech.week_start} to {speech.week_end}",
            policy_dimensions=policy_dimensions,
            overall_activity_level=activity_level,
            raw_analysis=content
        )

    def _parse_dimensions(self, content: str) -> Dict[str, str]:
        """Parse policy dimensions from LLM response"""
        dimensions = {}
        lines = content.split('\n')
        in_dimensions = False
        
        for line in lines:
            if 'DIMENSIONS:' in line.upper():
                in_dimensions = True
                continue
            elif 'ACTIVITY:' in line.upper():
                in_dimensions = False
                continue
            elif in_dimensions and line.strip().startswith('-'):
                # Parse "- Policy Area: stance description"
                parts = line.strip()[1:].split(':', 1)
                if len(parts) == 2:
                    dimension = parts[0].strip()
                    stance = parts[1].strip()
                    dimensions[dimension] = stance
        
        return dimensions

    def _parse_activity_level(self, content: str) -> str:
        """Parse activity level from LLM response"""
        lines = content.split('\n')
        for line in lines:
            if 'ACTIVITY:' in line.upper():
                return line.split(':', 1)[1].strip() if ':' in line else "Medium"
        return "Medium"

    def assess_dimension_weight(self, summaries: List[MultiDimensionalSummary], dimension: str) -> float:
        """Assess how much attention this dimension received overall (for weighting)"""
        total_legislators = len(summaries)
        legislators_addressing = sum(1 for s in summaries if dimension in s.policy_dimensions)
        
        if legislators_addressing == 0:
            return 0.0
        
        # Weight based on coverage and text length discussing this dimension
        coverage_weight = legislators_addressing / total_legislators
        
        # Simple heuristic: more legislators addressing = more important
        importance_weight = min(coverage_weight * 2, 1.0)  # Cap at 1.0
        
        return importance_weight

    def compare_on_dimension(self, a: MultiDimensionalSummary, b: MultiDimensionalSummary, 
                           dimension: str) -> DimensionComparison:
        """Compare two legislators on a specific policy dimension"""
        
        # Check if both legislators addressed this dimension
        stance_a = a.policy_dimensions.get(dimension, "")
        stance_b = b.policy_dimensions.get(dimension, "")
        
        if not stance_a or not stance_b:
            return DimensionComparison(
                legislator_a_id=a.legislator_id,
                legislator_b_id=b.legislator_id,
                dimension=dimension,
                winner="Skip",
                confidence=0.0,
                reasoning=f"One or both legislators did not address {dimension} significantly"
            )

        prompt = (
            f"Compare these two legislators' positions on {dimension}:\n\n"
            f"Legislator A ({a.legislator_name}):\n{stance_a}\n\n"
            f"Legislator B ({b.legislator_name}):\n{stance_b}\n\n"
            f"Question: Who takes a more CONSERVATIVE/Republican stance on {dimension}?\n"
            f"(Conservative = traditional values, limited government, free market)\n"
            f"Respond with only: Legislator A or Legislator B\n"
            f"Then briefly explain why."
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        content = response.choices[0].message.content.strip()
        lines = content.split("\n")
        winner_line = lines[0].strip()
        reasoning = "\n".join(lines[1:]).strip()

        if "Legislator A" in winner_line:
            winner = "A"
        elif "Legislator B" in winner_line:
            winner = "B"
        else:
            winner = "Skip"

        return DimensionComparison(
            legislator_a_id=a.legislator_id,
            legislator_b_id=b.legislator_id,
            dimension=dimension,
            winner=winner,
            confidence=1.0,
            reasoning=reasoning
        )

# === Data Processing ===
def select_random_week(df: pd.DataFrame) -> Tuple[str, str]:
    """Select a random week from the dataset"""
    df['date'] = pd.to_datetime(df['date'])
    min_date = df['date'].min()
    max_date = df['date'].max()
    
    # Generate random date within range
    random_date = min_date + timedelta(days=random.randint(0, (max_date - min_date).days))
    
    # Get week boundaries (Monday to Sunday)
    week_start = random_date - timedelta(days=random_date.weekday())
    week_end = week_start + timedelta(days=6)
    
    return week_start.strftime('%Y-%m-%d'), week_end.strftime('%Y-%m-%d')

def aggregate_weekly_speeches(df: pd.DataFrame, week_start: str, week_end: str) -> List[WeeklyLegislatorSpeech]:
    """Aggregate speeches by legislator for the specified week"""
    df['date'] = pd.to_datetime(df['date'])
    week_start_dt = pd.to_datetime(week_start)
    week_end_dt = pd.to_datetime(week_end)
    
    # Filter to target week
    week_data = df[(df['date'] >= week_start_dt) & (df['date'] <= week_end_dt)]
    
    print(f"📅 Target week: {week_start} to {week_end}")
    print(f"📊 Total speeches in week: {len(week_data)}")
    
    # Group by legislator
    weekly_speeches = []
    for legislator_id, group in week_data.groupby('bonica.rid'):
        concatenated_text = " ".join(group['text'].astype(str))
        
        weekly_speech = WeeklyLegislatorSpeech(
            legislator_id=legislator_id,
            legislator_name=legislator_id,  # Could enhance with actual names
            week_start=week_start,
            week_end=week_end,
            total_speeches=len(group),
            concatenated_text=concatenated_text
        )
        weekly_speeches.append(weekly_speech)
    
    print(f"👥 Unique legislators in week: {len(weekly_speeches)}")
    return weekly_speeches

# === Multi-Dimensional Bradley-Terry ===
def run_multidimensional_bradley_terry(dimension_comparisons: Dict[str, List[DimensionComparison]], 
                                     legislators: List[str]) -> Dict[str, Dict[str, float]]:
    """Run Bradley-Terry for each dimension separately"""
    results = {}
    
    for dimension, comparisons in dimension_comparisons.items():
        print(f"\n🔍 Processing dimension: {dimension}")
        
        # Filter out 'Skip' comparisons
        valid_comparisons = [c for c in comparisons if c.winner != "Skip"]
        
        if len(valid_comparisons) < 3:
            print(f"⚠️ Not enough comparisons for {dimension} ({len(valid_comparisons)})")
            continue
            
        # Build index mapping
        index_map = {}
        current_index = 0
        comparisons_list = []
        
        for comp in valid_comparisons:
            if comp.legislator_a_id not in index_map:
                index_map[comp.legislator_a_id] = current_index
                current_index += 1
            if comp.legislator_b_id not in index_map:
                index_map[comp.legislator_b_id] = current_index
                current_index += 1

            i = index_map[comp.legislator_a_id]
            j = index_map[comp.legislator_b_id]

            if comp.winner == "A":
                comparisons_list.append((i, j))
            elif comp.winner == "B":
                comparisons_list.append((j, i))

        if len(comparisons_list) == 0:
            print(f"❌ No usable comparisons for {dimension}")
            continue

        try:
            scores = choix.ilsr_pairwise(len(index_map), comparisons_list, alpha=0.1, max_iter=1000)
            
            # Map scores back to legislator IDs
            dimension_scores = {}
            for leg_id, idx in index_map.items():
                dimension_scores[leg_id] = scores[idx]
                
            results[dimension] = dimension_scores
            print(f"✅ {dimension}: {len(dimension_scores)} legislators scored")
            
        except Exception as e:
            print(f"❌ Bradley-Terry failed for {dimension}: {e}")
            continue
    
    return results

# === Weighted Overall Scoring ===
def calculate_weighted_overall_scores(dimension_scores: Dict[str, Dict[str, float]], 
                                    dimension_weights: Dict[str, float]) -> Dict[str, float]:
    """Calculate weighted overall liberal-conservative scores"""
    
    # Get all legislators
    all_legislators = set()
    for dimension_scores_dict in dimension_scores.values():
        all_legislators.update(dimension_scores_dict.keys())
    
    overall_scores = {}
    
    for legislator in all_legislators:
        weighted_sum = 0.0
        total_weight = 0.0
        
        for dimension, scores_dict in dimension_scores.items():
            if legislator in scores_dict and dimension in dimension_weights:
                weight = dimension_weights[dimension]
                score = scores_dict[legislator]
                
                weighted_sum += score * weight
                total_weight += weight
        
        # Calculate weighted average (higher = more conservative)
        if total_weight > 0:
            overall_scores[legislator] = weighted_sum / total_weight
        else:
            overall_scores[legislator] = 0.0  # Neutral if no data
    
    return overall_scores

# === Main Pipeline ===
def run_weekly_multidimensional_pipeline(df: pd.DataFrame, client: MultiDimensionalLLMClient, 
                                        max_legislators: int = 8) -> Tuple[Dict[str, Dict[str, float]], 
                                                                         List[MultiDimensionalSummary], 
                                                                         Dict[str, List[DimensionComparison]],
                                                                         Dict[str, float],
                                                                         Dict[str, float]]:
    """Run the complete weekly multi-dimensional pipeline"""
    
    # Step 1: Select random week and aggregate speeches
    week_start, week_end = select_random_week(df)
    weekly_speeches = aggregate_weekly_speeches(df, week_start, week_end)
    
    # Limit for testing
    weekly_speeches = weekly_speeches[:max_legislators]
    print(f"🔬 Using {len(weekly_speeches)} legislators for analysis")
    
    # Step 2: Extract policy dimensions for each legislator
    print("\n📋 Stage 1: Extracting policy dimensions...")
    summaries = []
    for i, speech in enumerate(weekly_speeches, 1):
        print(f"  Processing {i}/{len(weekly_speeches)}: {speech.legislator_id}")
        summary = client.extract_policy_dimensions(speech)
        summaries.append(summary)
    
    # Step 3: Identify common dimensions
    all_dimensions = set()
    for summary in summaries:
        all_dimensions.update(summary.policy_dimensions.keys())
    
    print(f"\n🏷️ Discovered policy dimensions: {list(all_dimensions)}")
    
    # Step 4: Pairwise comparisons for each dimension
    print("\n⚖️ Stage 2: Pairwise comparisons by dimension...")
    dimension_comparisons = {dim: [] for dim in all_dimensions}
    
    for dim in all_dimensions:
        print(f"\n  Comparing on: {dim}")
        for i, j in itertools.combinations(range(len(summaries)), 2):
            comparison = client.compare_on_dimension(summaries[i], summaries[j], dim)
            dimension_comparisons[dim].append(comparison)
            print(f"    {comparison.legislator_a_id} vs {comparison.legislator_b_id} → {comparison.winner}")
    
    # Step 5: Multi-dimensional Bradley-Terry scoring
    print("\n📊 Stage 3: Multi-dimensional Bradley-Terry scoring...")
    legislator_ids = [s.legislator_id for s in summaries]
    scores = run_multidimensional_bradley_terry(dimension_comparisons, legislator_ids)
    
    # Step 6: Calculate dimension weights
    print("\n⚖️ Stage 4: Calculating dimension weights...")
    dimension_weights = {}
    for dimension in all_dimensions:
        if dimension in scores:  # Only weight dimensions that have scores
            weight = client.assess_dimension_weight(summaries, dimension)
            dimension_weights[dimension] = weight
            print(f"  {dimension}: weight = {weight:.3f}")
    
    # Step 7: Calculate weighted overall scores
    print("\n🎯 Stage 5: Computing weighted overall ideology scores...")
    overall_scores = calculate_weighted_overall_scores(scores, dimension_weights)
    
    # Sort overall scores (lower = more liberal, higher = more conservative)
    sorted_overall = sorted(overall_scores.items(), key=lambda x: x[1])
    print("📊 Overall Liberal-Conservative Ranking:")
    for i, (legislator, score) in enumerate(sorted_overall):
        if i == 0:
            print(f"  {i+1}. {legislator}: {score:.3f} (Most Liberal)")
        elif i == len(sorted_overall) - 1:
            print(f"  {i+1}. {legislator}: {score:.3f} (Most Conservative)")
        else:
            print(f"  {i+1}. {legislator}: {score:.3f}")
    
    return scores, summaries, dimension_comparisons, dimension_weights, overall_scores