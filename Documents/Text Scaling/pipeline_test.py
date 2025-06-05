# pipeline_test.py

import os
import pandas as pd
from dotenv import load_dotenv
from ideology_pipeline_complete import LegislatorSpeech, LLMClient, run_pipeline

# === Load API Key ===
load_dotenv("/Users/menglinliu/Documents/JoshuaClinton/emotion_pipeline/.env")
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found in environment.")

# === Load Sample Data ===
df = pd.read_csv("/Users/menglinliu/Documents/Text Scaling/congress_demo.csv")
# Use 6 legislators for testing (15 comparisons - manageable for API)
df = df.head(6)

# === Create LegislatorSpeech objects ===
speeches = [
    LegislatorSpeech(
        legislator_id=row["bonica.rid"],
        legislator_name=row["bonica.rid"],  # Modify if full name available
        date=row["date"],
        issue_area="Environment",
        bill_name="Sample Bill",
        speech_text=row["text"],
        session="117th Congress"
    )
    for _, row in df.iterrows()
]

# === Initialize LLM Client ===
client = LLMClient(provider="openai", api_key=api_key, model="gpt-4")

# === Run Pipeline ===
scores, summaries, comparisons = run_pipeline(speeches, client, dimension="pro-environmental stance")

# === Output Results ===
print(f"\n📊 PIPELINE COMPLETED SUCCESSFULLY!")
print(f"📈 Total legislators: {len(speeches)}")
print(f"📈 Total comparisons: {len(comparisons)}")
print(f"📈 Scores calculated: {len(scores)}")

print("\n🏁 Bradley-Terry Scores (higher = more pro-environmental):")
if scores is not None and len(scores) > 0:
    # Create ranked list of legislators by score
    ranked_legislators = [(speeches[i].legislator_id, scores[i]) for i in range(len(scores))]
    ranked_legislators.sort(key=lambda x: x[1], reverse=True)
    
    for rank, (legislator_id, score) in enumerate(ranked_legislators, 1):
        print(f"  {rank}. {legislator_id}: {score:.3f}")
else:
    print("❌ No scores calculated - check Bradley-Terry model")

print("\n📝 Stage 1: Structured Summaries:")
for i, summary in enumerate(summaries, 1):
    print(f"\n--- {i}. {summary.legislator_id} ---")
    print(summary.raw_summary)

print("\n⚖️ Stage 2: Pairwise Comparisons (showing first 5):")
for i, c in enumerate(comparisons[:5]):
    print(f"{c.legislator_a_id} vs {c.legislator_b_id} → Winner: {c.winner}")
    print(f"Reasoning: {c.reasoning}\n")
    
print(f"... (showing 5 of {len(comparisons)} total comparisons)")

print("\n🎯 FINAL RANKING BY PRO-ENVIRONMENTAL STANCE:")
if scores is not None and len(scores) > 0:
    ranked_legislators = [(speeches[i].legislator_id, scores[i]) for i in range(len(scores))]
    ranked_legislators.sort(key=lambda x: x[1], reverse=True)
    
    print("Rank | Legislator | Score")
    print("-" * 30)
    for rank, (legislator_id, score) in enumerate(ranked_legislators, 1):
        if rank == 1:
            print(f"  {rank}.  {legislator_id:>10} | {score:>6.3f} (Most Pro-Environmental)")
        elif rank == len(ranked_legislators):
            print(f"  {rank}.  {legislator_id:>10} | {score:>6.3f} (Least Pro-Environmental)")
        else:
            print(f"  {rank}.  {legislator_id:>10} | {score:>6.3f}")
else:
    print("❌ No final ranking available")