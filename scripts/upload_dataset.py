#!/usr/bin/env python3
"""
Upload Political Text Scaling Dataset to Hugging Face

Simple script to upload the congressional speech dataset using the consolidated
data utilities. Handles authentication and large dataset uploads.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scaling.data_utils import TextScalingDataset, TextScalingUploader


def main():
    """Main upload function."""
    print("🚀 Political Text Scaling Dataset Upload")
    print("=" * 50)
    
    # Configuration
    DATASET_SOURCE = "/Users/menglinliu/Documents/Text Scaling/text_db.csv"
    REPO_NAME = "political-text-scaling"
    TOKEN = "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"
    
    # Get username
    username = input("Enter your Hugging Face username: ").strip()
    if not username:
        print("❌ Username required")
        return
    
    try:
        # Load dataset from local file
        print(f"\n📂 Loading dataset from {DATASET_SOURCE}...")
        dataset = TextScalingDataset(source=DATASET_SOURCE)
        
        print(f"✅ Loaded {len(dataset):,} congressional speeches")
        print(f"📊 Found {len(dataset.topic_cols)} topic weight columns")
        
        # Show sample
        sample = dataset.get_sample(1)
        doc = sample.iloc[0]
        print(f"\n📝 Sample document:")
        print(f"   Date: {doc.get('date', 'N/A')}")
        print(f"   Congress: {doc.get('congno', 'N/A')}")
        print(f"   Text: {doc.get('text', '')[:150]}...")
        
        # Confirm upload
        proceed = input(f"\n🤔 Upload to {username}/{REPO_NAME}? (y/n): ").lower()
        if proceed != 'y':
            print("❌ Upload cancelled")
            return
        
        # Initialize uploader and authenticate
        uploader = TextScalingUploader(token=TOKEN)
        uploader.authenticate()
        
        # Upload dataset
        repo_url = uploader.upload_dataset(
            dataset=dataset,
            repo_name=REPO_NAME,
            username=username,
            private=True
        )
        
        print(f"\n🎉 Success! Dataset available at:")
        print(f"   {repo_url}")
        
        # Create test script
        test_script = f'''#!/usr/bin/env python3
"""Test script for uploaded dataset"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scaling.data_utils import TextScalingDataset

# Load and test dataset
dataset = TextScalingDataset(source="{username}/{REPO_NAME}", split="train")
print(f"✅ Dataset loaded: {{len(dataset):,}} documents")
print(f"📊 Topic columns: {{len(dataset.topic_cols)}}")

# Show sample
sample = dataset.get_sample(1)
print(f"\\n📝 Sample: {{sample.iloc[0]['text'][:200]}}...")
'''
        
        with open("test_uploaded_dataset.py", "w") as f:
            f.write(test_script)
        
        print(f"\n📋 Created test script: test_uploaded_dataset.py")
        print(f"   Run: python test_uploaded_dataset.py")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())