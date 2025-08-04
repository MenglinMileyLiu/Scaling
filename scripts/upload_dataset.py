#!/usr/bin/env python3
"""
Upload Political Text Scaling Dataset to Hugging Face Hub

Professional upload script using the consolidated data utilities.
Handles authentication, large dataset uploads, and progress tracking.

Usage:
    python scripts/upload_dataset.py
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from scaling.dataset import TextScalingDataset
from scaling.processors import TextScalingProcessor

# Optional Hugging Face imports
try:
    from huggingface_hub import login, create_repo
    from datasets import Dataset, DatasetDict
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class DatasetUploader:
    """Professional dataset uploader with comprehensive features."""
    
    def __init__(self, token: str = None):
        """
        Initialize uploader.
        
        Args:
            token: Hugging Face API token
        """
        if not HF_AVAILABLE:
            raise ImportError("huggingface_hub not available. Install with: pip install huggingface_hub datasets")
        
        self.token = token
        self.authenticated = False
    
    def authenticate(self):
        """Authenticate with Hugging Face Hub."""
        if not self.token:
            raise ValueError("No token provided. Get one from https://huggingface.co/settings/tokens")
        
        try:
            login(token=self.token)
            self.authenticated = True
            print("✅ Successfully authenticated with Hugging Face")
        except Exception as e:
            raise RuntimeError(f"Authentication failed: {e}")
    
    def upload_dataset(self,
                      dataset: TextScalingDataset,
                      repo_name: str,
                      username: str,
                      private: bool = True,
                      description: str = None) -> str:
        """
        Upload dataset to Hugging Face Hub.
        
        Args:
            dataset: TextScalingDataset to upload
            repo_name: Repository name
            username: Hugging Face username
            private: Whether repository should be private
            description: Repository description
            
        Returns:
            Repository URL
        """
        if not self.authenticated:
            raise RuntimeError("Not authenticated. Call authenticate() first.")
        
        repo_id = f"{username}/{repo_name}"
        
        print(f"📤 Uploading dataset to {repo_id}")
        print(f"   Dataset size: {len(dataset):,} documents")
        print(f"   Topic columns: {len(dataset.topic_cols)}")
        print(f"   Memory usage: {dataset.df.memory_usage(deep=True).sum() / (1024**2):.1f} MB")
        
        # Create repository
        try:
            create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=private,
                exist_ok=True
            )
            print(f"📁 Repository created/verified: {repo_id}")
        except Exception as e:
            print(f"⚠️  Repository creation warning: {e}")
        
        # Prepare dataset splits
        dataset_dict = self._prepare_dataset_splits(dataset)
        
        # Upload with progress tracking
        try:
            print("🚀 Starting upload...")
            dataset_dict.push_to_hub(
                repo_id,
                private=private,
                commit_message=f"Upload political text scaling dataset\n\n"
                              f"- {len(dataset):,} congressional speeches\n"
                              f"- {len(dataset.topic_cols)} topic weight columns\n"
                              f"- Train/validation/test splits included\n"
                              f"- Generated with Claude Code"
            )
            
            repo_url = f"https://huggingface.co/datasets/{repo_id}"
            print(f"✅ Upload successful!")
            print(f"   Dataset URL: {repo_url}")
            
            return repo_url
            
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            # Save locally as backup
            backup_path = f"{repo_name}_backup"
            dataset_dict.save_to_disk(backup_path)
            print(f"💾 Dataset saved locally to '{backup_path}' as backup")
            raise
    
    def _prepare_dataset_splits(self, dataset: TextScalingDataset) -> DatasetDict:
        """
        Prepare dataset splits for upload.
        
        Args:
            dataset: Source dataset
            
        Returns:
            DatasetDict with train/validation/test splits
        """
        df = dataset.df
        
        # Create splits based on training.set column if available
        if 'training.set' in df.columns:
            print("📊 Creating splits based on training.set column...")
            
            train_df = df[df['training.set'] == 1].copy()
            test_df = df[df['training.set'] == 0].copy()
            
            # Create validation set from training data (10%)
            val_size = max(100, int(0.1 * len(train_df)))
            val_df = train_df.sample(n=val_size, random_state=42)
            train_df = train_df.drop(val_df.index).reset_index(drop=True)
            val_df = val_df.reset_index(drop=True)
            test_df = test_df.reset_index(drop=True)
            
            print(f"   Train: {len(train_df):,} documents")
            print(f"   Validation: {len(val_df):,} documents")
            print(f"   Test: {len(test_df):,} documents")
            
            dataset_dict = DatasetDict({
                'train': Dataset.from_pandas(train_df, preserve_index=False),
                'validation': Dataset.from_pandas(val_df, preserve_index=False),
                'test': Dataset.from_pandas(test_df, preserve_index=False)
            })
            
        else:
            print("📊 Creating train/test split (80/20)...")
            
            # Simple train/test split
            train_size = int(0.8 * len(df))
            train_df = df[:train_size].reset_index(drop=True)
            test_df = df[train_size:].reset_index(drop=True)
            
            print(f"   Train: {len(train_df):,} documents")
            print(f"   Test: {len(test_df):,} documents")
            
            dataset_dict = DatasetDict({
                'train': Dataset.from_pandas(train_df, preserve_index=False),
                'test': Dataset.from_pandas(test_df, preserve_index=False)
            })
        
        return dataset_dict


def main():
    """Main upload workflow."""
    print("🚀 Political Text Scaling Dataset Upload")
    print("=" * 50)
    
    # Configuration
    DEFAULT_SOURCE = "/Users/menglinliu/Documents/Text Scaling/text_db.csv"
    DEFAULT_REPO = "political-text-scaling"
    DEFAULT_TOKEN = "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"
    
    # Get user inputs
    print("\n📋 Configuration:")
    
    # Data source
    source = input(f"Dataset source [{DEFAULT_SOURCE}]: ").strip()
    if not source:
        source = DEFAULT_SOURCE
    
    # Check if source exists
    if source.startswith("/") and not Path(source).exists():
        print(f"❌ File not found: {source}")
        return 1
    
    # Repository name
    repo_name = input(f"Repository name [{DEFAULT_REPO}]: ").strip()
    if not repo_name:
        repo_name = DEFAULT_REPO
    
    # Username
    username = input("Hugging Face username: ").strip()
    if not username:
        print("❌ Username is required")
        return 1
    
    # Token
    token = input(f"HF Token [using default]: ").strip()
    if not token:
        token = DEFAULT_TOKEN
    
    # Privacy setting
    private_input = input("Make repository private? [y/N]: ").strip().lower()
    private = private_input in ['y', 'yes']
    
    print(f"\n📄 Upload Summary:")
    print(f"   Source: {source}")
    print(f"   Repository: {username}/{repo_name}")
    print(f"   Privacy: {'Private' if private else 'Public'}")
    
    # Confirm
    confirm = input("\n🤔 Proceed with upload? [y/N]: ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("❌ Upload cancelled")
        return 0
    
    try:
        # Load dataset
        print(f"\n📂 Loading dataset...")
        dataset = TextScalingDataset(source=source)
        
        print(f"\n📊 Dataset Summary:")
        print(f"   Documents: {len(dataset):,}")
        print(f"   Columns: {len(dataset.df.columns)}")
        print(f"   Topic columns: {len(dataset.topic_cols)}")
        print(f"   Memory usage: {dataset.df.memory_usage(deep=True).sum() / (1024**2):.1f} MB")
        
        # Show sample
        if len(dataset) > 0:
            sample = dataset.get_sample(1)
            sample_doc = sample.iloc[0]
            print(f"\n📝 Sample document:")
            print(f"   ID: {sample_doc.get('doc.id', 'N/A')}")
            print(f"   Date: {sample_doc.get('date', 'N/A')}")
            print(f"   Congress: {sample_doc.get('congno', 'N/A')}")
            print(f"   Text: {sample_doc.get('text', '')[:150]}...")
        
        # Initialize uploader
        uploader = DatasetUploader(token=token)
        uploader.authenticate()
        
        # Upload dataset
        repo_url = uploader.upload_dataset(
            dataset=dataset,
            repo_name=repo_name,
            username=username,
            private=private
        )
        
        # Create test script
        test_script_content = f'''#!/usr/bin/env python3
"""Test script for uploaded dataset"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from scaling.dataset import TextScalingDataset

def test_uploaded_dataset():
    """Test the uploaded dataset."""
    print("🧪 Testing uploaded dataset...")
    
    try:
        # Load dataset
        dataset = TextScalingDataset(
            source="{username}/{repo_name}",
            split="train",
            sample_size=100  # Small sample for testing
        )
        
        print(f"✅ Dataset loaded: {{len(dataset):,}} documents")
        print(f"📊 Topic columns: {{len(dataset.topic_cols)}}")
        
        # Show sample
        sample = dataset.get_sample(1)
        sample_doc = sample.iloc[0]
        print(f"\\n📝 Sample document:")
        print(f"   Text: {{sample_doc.get('text', '')[:200]}}...")
        
        # Show topic analysis
        topic_summary = dataset.get_topic_summary()
        if 'dominant_topic_distribution' in topic_summary:
            print(f"\\n🏷️  Top topics:")
            for topic, count in list(topic_summary['dominant_topic_distribution'].items())[:3]:
                clean_name = topic.replace('tw.', '').replace('.', ' ').title()
                print(f"   {{clean_name}}: {{count}} documents")
        
        print("\\n✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {{e}}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(test_uploaded_dataset())
'''
        
        test_script_path = project_root / "scripts" / "test_uploaded_dataset.py"
        with open(test_script_path, 'w') as f:
            f.write(test_script_content)
        
        print(f"\n🎉 Upload Complete!")
        print(f"   Dataset URL: {repo_url}")
        print(f"   Test script: {test_script_path}")
        print(f"\n📋 Next steps:")
        print(f"   1. Test access: python scripts/test_uploaded_dataset.py")
        print(f"   2. Visit dataset page: {repo_url}")
        print(f"   3. Add collaborators in dataset settings if needed")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n❌ Upload interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())