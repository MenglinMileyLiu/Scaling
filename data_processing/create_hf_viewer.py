#!/usr/bin/env python3
"""
Create properly structured dataset for Hugging Face viewer
"""

from datasets import load_dataset, DatasetDict, Dataset
import pandas as pd
from huggingface_hub import login

def create_viewer_friendly_dataset():
    """Create a dataset structure that enables HF viewer"""
    
    # Login
    login(token="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")
    
    # Load the current dataset
    print("Loading current dataset...")
    dataset = load_dataset("mliliu/dime-recipients")
    df = dataset['train'].to_pandas()
    
    # Clean and optimize for viewer
    print("Preparing data for viewer...")
    
    # Convert problematic columns
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str)
    
    # Create multiple splits for better organization
    print("Creating organized splits...")
    
    # Main dataset
    main_dataset = Dataset.from_pandas(df)
    
    # Create meaningful subsets
    recent_df = df[df['cycle'] >= 2010].copy()
    federal_df = df[df['nimsp.office'].isin(['house', 'senate'])].copy()
    
    # Create DatasetDict with multiple splits
    dataset_dict = DatasetDict({
        'train': main_dataset,
        'recent': Dataset.from_pandas(recent_df),
        'federal': Dataset.from_pandas(federal_df)
    })
    
    # Upload with viewer-friendly format
    print("Uploading viewer-friendly version...")
    dataset_dict.push_to_hub(
        "mliliu/dime-recipients",
        private=True,
        commit_message="Add dataset viewer support with multiple splits"
    )
    
    print("✅ Dataset updated for viewer support!")
    print("Check: https://huggingface.co/datasets/mliliu/dime-recipients")
    
    # Show preview
    print("\nDataset preview:")
    print(f"Main dataset: {len(dataset_dict['train'])} records")
    print(f"Recent (2010+): {len(dataset_dict['recent'])} records") 
    print(f"Federal candidates: {len(dataset_dict['federal'])} records")
    
    return dataset_dict

if __name__ == "__main__":
    create_viewer_friendly_dataset()