#!/usr/bin/env python3
"""
Script to upload DIME recipients database to Hugging Face
Can work with downloaded file or direct URL if available
"""

from huggingface_hub import login, create_repo
from datasets import Dataset
import pandas as pd
import os
import sys
import requests
from io import StringIO

def upload_dime_recipients():
    # Configuration
    REPO_NAME = "dime-recipients"  # Change this to your preferred name
    USERNAME = input("Enter your Hugging Face username: ")  # Will prompt you for username
    
    # Using your scaling token
    TOKEN = "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"
    
    print("\nOptions for data source:")
    print("1. Local file (if you already downloaded it)")
    print("2. Sample data (recent years only, for testing)")
    print("3. Filter by specific years/states")
    
    choice = input("\nChoose option (1-3): ")
    
    # Login to Hugging Face first
    print("\nLogging in to Hugging Face...")
    try:
        login(token=TOKEN)
        print("Successfully logged in!")
    except Exception as e:
        print(f"Error logging in: {e}")
        print("Please check your token and try again.")
        return
    
    recipients_df = None
    
    if choice == "1":
        # Option 1: Local file
        DATA_FILE = input("Enter path to recipDB file: ").strip()
        # Remove escape characters if present
        DATA_FILE = DATA_FILE.replace('\\', '')
        
        # If file not found, try in current directory
        if not os.path.exists(DATA_FILE):
            # Check if file exists in current directory
            filename = os.path.basename(DATA_FILE)
            if os.path.exists(filename):
                DATA_FILE = filename
                print(f"Found file in current directory: {DATA_FILE}")
            else:
                print(f"Error: Data file '{DATA_FILE}' not found!")
                print(f"Current directory: {os.getcwd()}")
                print("\nFiles in current directory:")
                for f in os.listdir('.'):
                    if f.endswith('.csv'):
                        print(f"  - {f}")
                return
        
        print(f"Loading recipients data from {DATA_FILE}...")
        print("This may take a while for large files...")
        try:
            # Read in chunks if file is very large
            chunk_size = 100000
            chunks = []
            for chunk in pd.read_csv(DATA_FILE, chunksize=chunk_size):
                chunks.append(chunk)
                print(f"Loaded {len(chunks) * chunk_size} rows...")
            recipients_df = pd.concat(chunks, ignore_index=True)
        except Exception as e:
            print(f"Error loading data: {e}")
            return
    
    elif choice == "2":
        # Option 2: Create sample data
        print("\nNote: Since DIME data requires authentication, we'll create a sample dataset structure.")
        print("You'll need to replace this with actual DIME data later.")
        
        # Create sample structure matching DIME recipients
        sample_data = {
            'recipient_id': ['R001', 'R002', 'R003', 'R004', 'R005'],
            'recipient_name': ['Sample Candidate 1', 'Sample Candidate 2', 'Sample Candidate 3', 'Sample Candidate 4', 'Sample Candidate 5'],
            'recipient_type': ['candidate', 'candidate', 'committee', 'candidate', 'party'],
            'party': ['D', 'R', 'D', 'R', 'D'],
            'state': ['CA', 'TX', 'NY', 'FL', 'IL'],
            'cycle': [2020, 2020, 2020, 2018, 2018],
            'office': ['House', 'Senate', 'NA', 'House', 'NA'],
            'cf_score': [-0.5, 0.8, -0.3, 1.2, -0.7]
        }
        recipients_df = pd.DataFrame(sample_data)
        print("Created sample dataset for testing.")
        
    elif choice == "3":
        # Option 3: Filtered data
        print("\nTo use filtered data, you need to:")
        print("1. Go to https://data.stanford.edu/dime")
        print("2. Use their web interface to filter data (by year, state, etc.)")
        print("3. Download the filtered subset")
        print("4. Run this script again with option 1")
        return
    
    else:
        print("Invalid choice.")
        return
    
    print(f"\nLoaded {len(recipients_df)} recipients")
    print(f"Columns: {recipients_df.columns.tolist()}")
    
    # Optional: Preview the data
    print("\nFirst 5 rows of data:")
    print(recipients_df.head())
    
    # Show data size
    print(f"\nDataset size: {recipients_df.shape[0]} rows, {recipients_df.shape[1]} columns")
    print(f"Memory usage: {recipients_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Ask if user wants to filter before uploading
    if len(recipients_df) > 100000:
        print("\nThis is a large dataset. Consider filtering to reduce size.")
        filter_choice = input("Filter by recent years? (y/n): ")
        if filter_choice.lower() == 'y':
            min_year = int(input("Enter minimum year (e.g., 2010): "))
            recipients_df = recipients_df[recipients_df['cycle'] >= min_year]
            print(f"Filtered to {len(recipients_df)} recipients from {min_year} onwards.")
    
    # Clean data before conversion
    print("\nCleaning data before conversion...")
    
    # Fix party column - convert all to string
    if 'party' in recipients_df.columns:
        recipients_df['party'] = recipients_df['party'].astype(str)
        print(f"Unique party values: {recipients_df['party'].value_counts().head(10).to_dict()}")
    
    # Fix any other problematic columns
    # Convert all object columns to string to avoid type issues
    for col in recipients_df.columns:
        if recipients_df[col].dtype == 'object':
            recipients_df[col] = recipients_df[col].astype(str)
            # Replace 'nan' strings with empty strings
            recipients_df[col] = recipients_df[col].replace('nan', '')
    
    # Convert to Hugging Face dataset
    print("\nConverting to Hugging Face dataset format...")
    try:
        dataset = Dataset.from_pandas(recipients_df)
    except Exception as e:
        print(f"Error during conversion: {e}")
        print("\nTrying alternative conversion method...")
        # Alternative: save to parquet first, then load
        temp_file = "temp_recipients.parquet"
        recipients_df.to_parquet(temp_file)
        dataset = Dataset.from_parquet(temp_file)
        os.remove(temp_file)
    
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
    try:
        dataset.push_to_hub(
            repo_id,
            private=True,
            commit_message="Upload DIME recipients database with ideology scores"
        )
        print("\n✅ Upload successful!")
        print(f"Dataset available at: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"Error uploading dataset: {e}")
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
    upload_dime_recipients()