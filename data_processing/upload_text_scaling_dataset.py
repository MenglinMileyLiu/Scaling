#!/usr/bin/env python3
"""
Script to upload Political Text Scaling dataset to Hugging Face
Handles large congressional speech dataset with topic weights
"""

from huggingface_hub import login, create_repo
from datasets import Dataset, DatasetDict
import pandas as pd
import os
import sys

def upload_text_scaling_dataset():
    # Configuration
    REPO_NAME = "political-text-scaling"  # Repository name for text scaling data
    USERNAME = input("Enter your Hugging Face username: ")  # Will prompt you for username
    
    # Using your scaling token
    TOKEN = "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"
    
    # Use the correct text scaling data file
    DATA_FILE = "/Users/menglinliu/Documents/Text Scaling/text_db.csv"
    
    print(f"\nWill upload text scaling dataset from: {DATA_FILE}")
    print("Dataset info:")
    print("- 307,793 congressional speeches")
    print("- 25 pre-computed topic weights")
    print("- 3.7GB file size")
    
    proceed = input("\nProceed with upload? (y/n): ")
    if proceed.lower() != 'y':
        print("Upload cancelled.")
        return
    
    # Login to Hugging Face first
    print("\nLogging in to Hugging Face...")
    try:
        login(token=TOKEN)
        print("Successfully logged in!")
    except Exception as e:
        print(f"Error logging in: {e}")
        print("Please check your token and try again.")
        return
    
    # Check file exists
    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file not found at {DATA_FILE}")
        return
    
    print(f"\nLoading text scaling data...")
    print("This is a large file (3.7GB), processing will take time...")
    
    # Read data in chunks due to large size
    chunk_size = 50000
    chunks = []
    
    try:
        for i, chunk in enumerate(pd.read_csv(DATA_FILE, chunksize=chunk_size)):
            if i == 0:
                print(f"\nColumns found: {len(chunk.columns)}")
                print(f"Topic columns: {sum(1 for col in chunk.columns if col.startswith('tw.'))}")
            
            # Clean data types
            for col in chunk.columns:
                if chunk[col].dtype == 'object':
                    chunk[col] = chunk[col].fillna('').astype(str)
            
            chunks.append(chunk)
            
            if (i + 1) * chunk_size % 100000 == 0:
                print(f"Processed {(i + 1) * chunk_size:,} rows...")
            
            # Optional: limit for testing
            # if i >= 5:  # Only process first 250k rows for testing
            #     print("\nLimiting to first 250k rows for testing")
            #     break
        
        # Combine chunks
        print("\nCombining all chunks...")
        df = pd.concat(chunks, ignore_index=True)
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    print(f"\nTotal documents loaded: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Show sample document
    print("\nSample document:")
    sample = df.iloc[0]
    print(f"Doc ID: {sample['doc.id']}")
    print(f"Date: {sample['date']}")
    print(f"Congress: {sample['congno']}")
    print(f"Text preview: {sample['text'][:150]}...")
    
    # Show topic distribution
    topic_cols = [col for col in df.columns if col.startswith('tw.')]
    print(f"\nTopic columns found: {len(topic_cols)}")
    
    # Create splits based on training.set column
    print("\nCreating dataset splits...")
    train_df = df[df['training.set'] == 1].copy()
    test_df = df[df['training.set'] == 0].copy()
    
    # Create validation set from training data
    val_size = int(0.1 * len(train_df))
    val_df = train_df.sample(n=val_size, random_state=42)
    train_df = train_df.drop(val_df.index)
    
    print(f"Train: {len(train_df):,} documents")
    print(f"Validation: {len(val_df):,} documents")
    print(f"Test: {len(test_df):,} documents")
    
    # Convert to Hugging Face DatasetDict
    print("\nConverting to Hugging Face format...")
    try:
        dataset_dict = DatasetDict({
            'train': Dataset.from_pandas(train_df, preserve_index=False),
            'validation': Dataset.from_pandas(val_df, preserve_index=False),
            'test': Dataset.from_pandas(test_df, preserve_index=False)
        })
    except Exception as e:
        print(f"Error during conversion: {e}")
        print("\nTrying to save smaller sample...")
        # If full dataset fails, try smaller sample
        sample_size = 10000
        dataset_dict = DatasetDict({
            'train': Dataset.from_pandas(train_df.head(sample_size), preserve_index=False),
            'validation': Dataset.from_pandas(val_df.head(1000), preserve_index=False),
            'test': Dataset.from_pandas(test_df.head(1000), preserve_index=False)
        })
        print(f"Using sample of {sample_size} documents due to size constraints")
    
    # Create repository if it doesn't exist
    repo_id = f"{USERNAME}/{REPO_NAME}"
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=True  # Start with private, can make public later
        )
        print(f"Created new repository: {repo_id}")
    except Exception as e:
        print(f"Repository might already exist or error occurred: {e}")
    
    # Upload the dataset
    print(f"\nUploading dataset to {repo_id}...")
    print("This may take a while due to the large size...")
    try:
        dataset_dict.push_to_hub(
            repo_id,
            private=True,
            commit_message="Upload political text scaling dataset with congressional speeches and topic weights"
        )
        print("\n✅ Upload successful!")
        print(f"Dataset available at: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"Error uploading dataset: {e}")
        print("\nTrying to save locally instead...")
        dataset_dict.save_to_disk("text_scaling_dataset")
        print("Dataset saved locally to 'text_scaling_dataset' folder")
        return
    
    # Create a simple test script
    print("\nCreating test script...")
    test_script = f"""
# Test script to verify dataset upload
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("{repo_id}", use_auth_token=True)

# Check the data
print(f"Dataset info: {{dataset}}")
print(f"Number of recipients: {{len(dataset['train'])}}")
print(f"Sample data: {{dataset['train'][:5]}}")
"""
    
    with open("test_dataset_access.py", "w") as f:
        f.write(test_script)
    
    print("\nTest script created: test_dataset_access.py")
    print("\nNext steps:")
    print("1. Share the repository with your coauthor by going to the dataset page")
    print("2. Click 'Settings' and add collaborators")
    print("3. Test access using the test_dataset_access.py script")

if __name__ == "__main__":
    upload_text_scaling_dataset()