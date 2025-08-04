
#!/usr/bin/env python3
"""
Test script to verify political text scaling dataset upload
"""

from datasets import load_dataset
import pandas as pd

def test_dataset_access():
    print("Testing access to political text scaling dataset...")
    
    try:
        # Load the dataset (try without token first, then with token if needed)
        try:
            dataset = load_dataset("mliliu/political-text-scaling")
        except Exception:
            dataset = load_dataset("mliliu/political-text-scaling", token=True)
        
        print("\n✅ Dataset loaded successfully!")
        print(f"\nDataset structure: {dataset}")
        
        # Check each split
        for split_name in dataset.keys():
            split = dataset[split_name]
            print(f"\n{split_name.upper()} split:")
            print(f"  - Number of documents: {len(split):,}")
            print(f"  - Features: {list(split.features.keys())[:5]}... ({len(split.features)} total)")
        
        # Sample from training data
        print("\n" + "="*60)
        print("SAMPLE DOCUMENT FROM TRAINING SET:")
        print("="*60)
        
        sample = dataset['train'][0]
        
        # Display basic info
        print(f"\nDocument ID: {sample.get('doc.id', 'N/A')}")
        print(f"Date: {sample.get('date', 'N/A')}")
        print(f"Congress: {sample.get('congno', 'N/A')}")
        print(f"Legislative Body: {sample.get('legis.body', 'N/A')}")
        
        # Show text preview
        text = sample.get('text', '')
        print(f"\nText preview (first 300 chars):")
        print(f"{text[:300]}...")
        
        # Show top topic weights
        print("\nTop 5 topic weights:")
        topic_scores = {}
        for key, value in sample.items():
            if key.startswith('tw.') and isinstance(value, (int, float)):
                topic_scores[key] = value
        
        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        for topic, score in sorted_topics:
            topic_name = topic.replace('tw.', '').replace('.', ' ').title()
            print(f"  - {topic_name}: {score:.4f}")
        
        # Convert small sample to pandas for analysis
        print("\n" + "="*60)
        print("CONVERTING TO PANDAS DATAFRAME:")
        print("="*60)
        
        # Take small sample to avoid memory issues
        df_sample = dataset['train'].select(range(min(100, len(dataset['train'])))).to_pandas()
        
        print(f"\nDataFrame shape: {df_sample.shape}")
        print(f"Memory usage: {df_sample.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Show column types
        print("\nColumn types summary:")
        print(df_sample.dtypes.value_counts())
        
        # Find topic columns
        topic_cols = [col for col in df_sample.columns if col.startswith('tw.')]
        print(f"\nFound {len(topic_cols)} topic weight columns")
        
        # Analyze topic dominance
        print("\nDominant topics in sample:")
        df_sample['dominant_topic'] = df_sample[topic_cols].idxmax(axis=1)
        topic_counts = df_sample['dominant_topic'].value_counts().head(5)
        for topic, count in topic_counts.items():
            topic_name = topic.replace('tw.', '').replace('.', ' ').title()
            print(f"  - {topic_name}: {count} documents")
        
        print("\n✅ All tests passed! Dataset is accessible and properly formatted.")
        
    except Exception as e:
        print(f"\n❌ Error accessing dataset: {e}")
        print("\nTroubleshooting steps:")
        print("1. Make sure you're logged in with: huggingface-cli login")
        print("2. Check if the dataset is private and you have access")
        print("3. Try accessing directly at: https://huggingface.co/datasets/mliliu/political-text-scaling")

if __name__ == "__main__":
    test_dataset_access()
