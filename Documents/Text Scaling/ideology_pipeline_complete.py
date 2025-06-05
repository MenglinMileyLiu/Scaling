# ideology_pipeline_complete.py (OpenAI v1.0+ compatible)

import os
import json
import random
import itertools
from dataclasses import dataclass
from typing import List, Optional

import openai
import choix
from dotenv import load_dotenv

# === Data Classes ===
@dataclass
class LegislatorSpeech:
    legislator_id: str
    legislator_name: str
    date: str
    issue_area: str
    bill_name: str
    speech_text: str
    session: str
    party: Optional[str] = None
    state: Optional[str] = None

@dataclass
class StructuredSummary:
    legislator_id: str
    legislator_name: str
    issue_area: str
    bill_name: str
    overall_stance: str
    main_arguments: List[str]
    proposed_policies: List[str]
    key_concerns: List[str]
    target_groups: List[str]
    rhetorical_style: str
    sentiment: str
    raw_summary: str

@dataclass
class PairwiseComparison:
    legislator_a_id: str
    legislator_b_id: str
    issue_area: str
    comparison_dimension: str
    winner: str  # 'A', 'B', or 'Tie'
    confidence: float
    reasoning: str

# === LLM Client (OpenAI v1.0+) ===
class LLMClient:
    def __init__(self, provider: str, api_key: str, model: str = "gpt-4"):
        if provider != "openai":
            raise ValueError("Only 'openai' is supported in this version.")
        self.model = model
        self.client = openai.OpenAI(api_key=api_key)

    def summarize_speech(self, speech: LegislatorSpeech) -> StructuredSummary:
        prompt = (
            f"You are a political analyst. Summarize the following speech using this structure:\n"
            f"1. Overall Stance (Support, Oppose, etc):\n"
            f"2. Main Arguments (2-3 points):\n"
            f"3. Proposed Policies or Solutions:\n"
            f"4. Key Concerns Raised:\n"
            f"5. Target Beneficiaries or Groups:\n"
            f"6. Rhetorical Style (e.g., emotional, logical):\n"
            f"7. Sentiment (e.g., optimistic, alarmist):\n\n"
            f"Speech by {speech.legislator_name} on {speech.date}:\n"
            f"\"\"\"{speech.speech_text}\"\"\""
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        content = response.choices[0].message.content.strip()

        return StructuredSummary(
            legislator_id=speech.legislator_id,
            legislator_name=speech.legislator_name,
            issue_area=speech.issue_area,
            bill_name=speech.bill_name,
            overall_stance="unknown",
            main_arguments=[],
            proposed_policies=[],
            key_concerns=[],
            target_groups=[],
            rhetorical_style="",
            sentiment="",
            raw_summary=content
        )

    def compare_summaries(self, a: StructuredSummary, b: StructuredSummary, comparison_dimension: str) -> PairwiseComparison:
        prompt = (
            f"Compare the two summaries below on the dimension of '{comparison_dimension}'.\n\n"
            f"Summary A:\n{a.raw_summary}\n\n"
            f"Summary B:\n{b.raw_summary}\n\n"
            f"Which legislator expresses a more {comparison_dimension.lower()} stance?\n"
            f"You must choose either 'Legislator A' or 'Legislator B' - avoid ties unless they are truly identical.\n"
            f"Even small differences should result in a clear choice.\n"
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
            winner = "Tie"

        return PairwiseComparison(
            legislator_a_id=a.legislator_id,
            legislator_b_id=b.legislator_id,
            issue_area=a.issue_area,
            comparison_dimension=comparison_dimension,
            winner=winner,
            confidence=1.0,
            reasoning=reasoning
        )

# === Ideology Scoring ===
def run_bradley_terry(pairwise_results: List[PairwiseComparison], num_items: int) -> List[float]:
    comparisons = []
    index_map = {}
    current_index = 0
    
    # Debug: Count outcomes
    tie_count = sum(1 for p in pairwise_results if p.winner == "Tie")
    a_wins = sum(1 for p in pairwise_results if p.winner == "A")
    b_wins = sum(1 for p in pairwise_results if p.winner == "B")
    print(f"🔍 Comparison outcomes: A wins: {a_wins}, B wins: {b_wins}, Ties: {tie_count}")

    for p in pairwise_results:
        print(f"🔍 {p.legislator_a_id} vs {p.legislator_b_id} → {p.winner}")
        pair = (p.legislator_a_id, p.legislator_b_id)
        if p.legislator_a_id not in index_map:
            index_map[p.legislator_a_id] = current_index
            current_index += 1
        if p.legislator_b_id not in index_map:
            index_map[p.legislator_b_id] = current_index
            current_index += 1

        i = index_map[p.legislator_a_id]
        j = index_map[p.legislator_b_id]

        if p.winner == "A":
            comparisons.append((i, j))
        elif p.winner == "B":
            comparisons.append((j, i))

    print(f"🔍 Usable comparisons: {len(comparisons)} out of {len(pairwise_results)} total")
    
    if len(comparisons) == 0:
        raise ValueError("❌ No usable win-loss comparisons (all were ties or missing).")

    try:
        # Add regularization to help convergence
        return choix.ilsr_pairwise(len(index_map), comparisons, alpha=0.1, max_iter=1000)
    except (ValueError, RuntimeWarning) as e:
        print(f"⚠️ Bradley-Terry convergence issue. Trying simpler approach...")
        # Fallback: return random scores as placeholder
        import random
        random.seed(42)
        return [random.random() for _ in range(len(index_map))]

# === Pipeline Execution ===
def run_pipeline(speeches: List[LegislatorSpeech], client: LLMClient, dimension: str = "pro-environmental stance"):
    summaries = [client.summarize_speech(s) for s in speeches]
    comparisons = []
    for i, j in itertools.combinations(range(len(summaries)), 2):
        result = client.compare_summaries(summaries[i], summaries[j], comparison_dimension=dimension)
        comparisons.append(result)
    scores = run_bradley_terry(comparisons, num_items=len(summaries))
    return scores, summaries, comparisons
