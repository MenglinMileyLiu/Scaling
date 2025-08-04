#!/usr/bin/env python3
"""
Political Text Scaling Data Utilities

Consolidated data processing, loading, and analysis utilities for congressional speech
text scaling research. Follows modular design patterns for maintainable code.

Classes:
    TextScalingDataset: Main dataset wrapper with filtering and batching
    TextScalingProcessor: Data processing and optimization pipeline
    TextScalingUploader: Utilities for uploading to Hugging Face
    
Functions:
    load_dataset_splits: Load train/validation/test splits
    compute_topic_statistics: Analyze topic weight distributions
    create_filtered_versions: Generate commonly used data subsets
"""

import pandas as pd
import numpy as np
import os
import json
import random
from typing import Dict, List, Optional, Union, Iterator, Tuple
from pathlib import Path

# Core data science libraries
from datasets import load_dataset, Dataset, DatasetDict
import pyarrow as pa
import pyarrow.parquet as pq

# Optional PyTorch imports
try:
    from torch.utils.data import DataLoader, IterableDataset
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Optional Hugging Face imports
try:
    from huggingface_hub import login, create_repo
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class TextScalingDataset:
    """
    Dataset wrapper for political text scaling data with flexible loading and filtering.
    
    Supports loading from Hugging Face, local files, and provides batch processing
    capabilities for congressional speech analysis.
    """
    
    def __init__(self, 
                 source: str = "mliliu/political-text-scaling",
                 split: str = "train",
                 filters: Optional[Dict] = None,
                 columns: Optional[List[str]] = None,
                 sample_size: Optional[int] = None):
        """
        Initialize the text scaling dataset.
        
        Args:
            source: HuggingFace dataset name or local file path
            split: Dataset split to load (train/validation/test)
            filters: Dict of column filters e.g. {'congno': lambda x: x >= 110}
            columns: List of columns to keep (None = all columns)
            sample_size: If specified, randomly sample this many documents
        """
        self.source = source
        self.split = split
        self.filters = filters or {}
        self.columns = columns
        self.sample_size = sample_size
        
        # Load and process data
        self._load_data()
        self._extract_topic_columns()
        
    def _load_data(self):
        """Load data from specified source."""
        print(f"Loading data from {self.source} (split: {self.split})...")
        
        if self.source.startswith("mliliu/") or "/" in self.source:
            # Load from HuggingFace
            try:
                dataset = load_dataset(self.source)
                self.df = dataset[self.split].to_pandas()
            except Exception:
                # Try with token if private dataset
                dataset = load_dataset(self.source, token=True)
                self.df = dataset[self.split].to_pandas()
        elif self.source.endswith('.parquet'):
            self.df = pd.read_parquet(self.source)
        elif self.source.endswith('.csv'):
            self.df = pd.read_csv(self.source)
        else:
            raise ValueError(f"Unsupported source format: {self.source}")
        
        print(f"Loaded {len(self.df):,} documents")
        
        # Apply filters
        if self.filters:
            original_size = len(self.df)
            for col, filter_func in self.filters.items():
                if col in self.df.columns:
                    self.df = self.df[self.df[col].apply(filter_func)]
            print(f"After filtering: {len(self.df):,} documents ({original_size - len(self.df):,} removed)")
        
        # Select columns
        if self.columns:
            available_cols = [col for col in self.columns if col in self.df.columns]
            self.df = self.df[available_cols]
            print(f"Selected {len(available_cols)} columns")
        
        # Sample if requested
        if self.sample_size and len(self.df) > self.sample_size:
            self.df = self.df.sample(n=self.sample_size, random_state=42).reset_index(drop=True)
            print(f"Sampled {self.sample_size:,} documents")
    
    def _extract_topic_columns(self):
        """Extract and store topic weight column names."""
        self.topic_cols = [col for col in self.df.columns if col.startswith('tw.')]
        if self.topic_cols:
            print(f"Found {len(self.topic_cols)} topic weight columns")
            # Add dominant topic analysis
            self.df['dominant_topic'] = self.df[self.topic_cols].idxmax(axis=1)
            self.df['max_topic_weight'] = self.df[self.topic_cols].max(axis=1)
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get single document by index."""
        return self.df.iloc[idx].to_dict()
    
    def get_batch(self, indices: List[int]) -> pd.DataFrame:
        """Get batch of documents as DataFrame."""
        return self.df.iloc[indices]
    
    def get_sample(self, n: int = 5) -> pd.DataFrame:
        """Get random sample of documents."""
        if TORCH_AVAILABLE:
            indices = torch.randperm(len(self.df))[:n].tolist()
        else:
            indices = random.sample(range(len(self.df)), min(n, len(self.df)))
        return self.get_batch(indices)
    
    def get_topic_summary(self) -> Dict:
        """Get summary statistics for topic distributions."""
        if not self.topic_cols:
            return {}
        
        topic_stats = {}
        for topic in self.topic_cols:
            topic_stats[topic] = {
                'mean': float(self.df[topic].mean()),
                'std': float(self.df[topic].std()),
                'max': float(self.df[topic].max()),
                'documents_with_weight': int((self.df[topic] > 0).sum())
            }
        
        # Most common dominant topics
        if 'dominant_topic' in self.df.columns:
            topic_stats['dominant_topic_counts'] = self.df['dominant_topic'].value_counts().head(10).to_dict()
        
        return topic_stats
    
    def iterate_batches(self, batch_size: int = 32, shuffle: bool = True) -> Iterator[pd.DataFrame]:
        """Iterate through dataset in batches."""
        indices = list(range(len(self.df)))
        
        if shuffle:
            random.shuffle(indices)
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            yield self.get_batch(batch_indices)


class TextScalingProcessor:
    """
    Processing pipeline for text scaling data optimization and analysis.
    
    Handles format conversion, data cleaning, filtering, and statistical analysis
    for congressional speech datasets.
    """
    
    def __init__(self, dataset: Optional[TextScalingDataset] = None):
        """
        Initialize processor.
        
        Args:
            dataset: TextScalingDataset to process (can be set later)
        """
        self.dataset = dataset
        self.df = dataset.df if dataset else None
        self.topic_cols = dataset.topic_cols if dataset else []
    
    def load_from_source(self, source: str, split: str = "train") -> 'TextScalingProcessor':
        """Load dataset from source and return self for chaining."""
        self.dataset = TextScalingDataset(source=source, split=split)
        self.df = self.dataset.df
        self.topic_cols = self.dataset.topic_cols
        return self
    
    def clean_data(self) -> 'TextScalingProcessor':
        """Clean and standardize the data."""
        if self.df is None:
            raise ValueError("No dataset loaded. Use load_from_source() first.")
        
        print("Cleaning data...")
        
        # Clean text fields
        for text_col in ['text', 'stemmed.text']:
            if text_col in self.df.columns:
                self.df[text_col] = self.df[text_col].fillna('').astype(str)
        
        # Parse dates
        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
            self.df['year'] = self.df['date'].dt.year
        
        # Clean congress numbers
        if 'congno' in self.df.columns:
            self.df['congno'] = pd.to_numeric(self.df['congno'], errors='coerce')
        
        print(f"Data cleaning complete. {len(self.df):,} documents ready.")
        return self
    
    def optimize_formats(self, output_dir: str = "text_scaling_optimized") -> Dict:
        """Convert to optimized storage formats."""
        if self.df is None:
            raise ValueError("No dataset loaded.")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"Optimizing formats in {output_dir}...")
        
        # Save as Parquet (best for analytics)
        parquet_path = output_path / "congressional_speeches.parquet"
        self.df.to_parquet(parquet_path, compression='snappy')
        parquet_size = parquet_path.stat().st_size / (1024**2)
        
        # Save as compressed CSV
        csv_path = output_path / "congressional_speeches.csv.gz"
        self.df.to_csv(csv_path, compression='gzip', index=False)
        csv_size = csv_path.stat().st_size / (1024**2)
        
        # Create shards for large-scale processing
        self._create_shards(output_path)
        
        format_info = {
            'parquet_path': str(parquet_path),
            'csv_path': str(csv_path),
            'parquet_size_mb': parquet_size,
            'csv_size_mb': csv_size
        }
        
        print(f"Parquet: {parquet_size:.1f} MB")
        print(f"CSV.gz: {csv_size:.1f} MB")
        print(f"Space savings: {(1 - parquet_size/csv_size)*100:.1f}%")
        
        return format_info
    
    def _create_shards(self, output_dir: Path, shard_size: int = 50000):
        """Create sharded dataset for distributed processing."""
        shard_dir = output_dir / "shards"
        shard_dir.mkdir(exist_ok=True)
        
        n_shards = len(self.df) // shard_size + 1
        
        for i in range(n_shards):
            start_idx = i * shard_size
            end_idx = min((i + 1) * shard_size, len(self.df))
            
            if start_idx < len(self.df):
                shard_df = self.df.iloc[start_idx:end_idx]
                shard_path = shard_dir / f"shard_{i:04d}.parquet"
                shard_df.to_parquet(shard_path)
        
        print(f"Created {n_shards} shards for distributed processing")
    
    def create_filtered_versions(self, output_dir: str = "text_scaling_filtered") -> Dict:
        """Create commonly used filtered versions."""
        if self.df is None:
            raise ValueError("No dataset loaded.")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        filters = {}
        
        # Legislative body filters
        if 'legis.body' in self.df.columns:
            filters['house_speeches'] = self.df[self.df['legis.body'] == 'US House']
            filters['senate_speeches'] = self.df[self.df['legis.body'] == 'US Senate']
        
        # Congress number filters
        if 'congno' in self.df.columns:
            filters['recent_congress'] = self.df[self.df['congno'] >= 110]
            filters['congress_108'] = self.df[self.df['congno'] == 108]
            filters['congress_109'] = self.df[self.df['congno'] == 109]
        
        # Training split filters
        if 'training.set' in self.df.columns:
            filters['training_speeches'] = self.df[self.df['training.set'] == 1]
            filters['test_speeches'] = self.df[self.df['training.set'] == 0]
        
        # Topic-focused filters
        high_weight_topics = ['tw.healthcare', 'tw.defense.and.foreign.policy', 'tw.economy']
        for topic in high_weight_topics:
            if topic in self.df.columns:
                topic_name = topic.replace('tw.', '').replace('.', '_')
                filters[f'{topic_name}_focused'] = self.df[self.df[topic] > 0.1]
        
        # Save filtered versions
        filter_info = {}
        for name, filtered_df in filters.items():
            if len(filtered_df) > 0:
                path = output_path / f"speeches_{name}.parquet"
                filtered_df.to_parquet(path)
                filter_info[name] = {
                    'rows': len(filtered_df),
                    'path': str(path),
                    'size_mb': path.stat().st_size / (1024**2)
                }
                print(f"Created {name}: {len(filtered_df):,} speeches")
        
        return filter_info
    
    def compute_comprehensive_statistics(self) -> Dict:
        """Compute comprehensive dataset statistics."""
        if self.df is None:
            raise ValueError("No dataset loaded.")
        
        print("Computing comprehensive statistics...")
        
        stats = {
            'basic_stats': {
                'total_speeches': len(self.df),
                'unique_documents': self.df['doc.id'].nunique() if 'doc.id' in self.df.columns else len(self.df),
                'unique_legislators': self.df['bonica.rid'].nunique() if 'bonica.rid' in self.df.columns else None,
                'congress_range': f"{self.df['congno'].min()}-{self.df['congno'].max()}" if 'congno' in self.df.columns else None,
                'date_range': f"{self.df['date'].min().date()}-{self.df['date'].max().date()}" if 'date' in self.df.columns and self.df['date'].notna().any() else None
            }
        }
        
        # Distribution statistics
        if 'legis.body' in self.df.columns:
            stats['legislative_body_distribution'] = self.df['legis.body'].value_counts().to_dict()
        
        if 'congno' in self.df.columns:
            stats['congress_distribution'] = self.df['congno'].value_counts().to_dict()
        
        if 'training.set' in self.df.columns:
            stats['training_split'] = self.df['training.set'].value_counts().to_dict()
        
        # Text statistics
        if 'text' in self.df.columns:
            stats['text_length_stats'] = {
                'mean_chars': float(self.df['text'].str.len().mean()),
                'median_chars': float(self.df['text'].str.len().median()),
                'mean_words': float(self.df['text'].str.split().str.len().mean())
            }
        
        # Topic weight statistics
        if self.topic_cols:
            topic_stats = {}
            for topic in self.topic_cols:
                topic_stats[topic] = {
                    'mean': float(self.df[topic].mean()),
                    'std': float(self.df[topic].std()),
                    'max': float(self.df[topic].max()),
                    'active_documents': int((self.df[topic] > 0.01).sum())
                }
            stats['topic_weight_stats'] = topic_stats
            
            if 'dominant_topic' in self.df.columns:
                stats['dominant_topic_distribution'] = self.df['dominant_topic'].value_counts().head(15).to_dict()
        
        # Missing data analysis
        key_columns = ['text', 'date', 'congno', 'bonica.rid']
        stats['missing_data'] = {
            col: float(self.df[col].isna().sum() / len(self.df))
            for col in key_columns if col in self.df.columns
        }
        
        return stats


class TextScalingUploader:
    """
    Utilities for uploading text scaling datasets to Hugging Face Hub.
    
    Handles authentication, repository creation, and large dataset uploads
    with proper error handling and progress tracking.
    """
    
    def __init__(self, token: Optional[str] = None):
        """
        Initialize uploader.
        
        Args:
            token: Hugging Face API token (will prompt if not provided)
        """
        if not HF_AVAILABLE:
            raise ImportError("huggingface_hub not available. Install with: pip install huggingface_hub")
        
        self.token = token
        self._authenticated = False
    
    def authenticate(self, token: Optional[str] = None):
        """Authenticate with Hugging Face Hub."""
        if token:
            self.token = token
        
        if not self.token:
            raise ValueError("No token provided. Get one from https://huggingface.co/settings/tokens")
        
        try:
            login(token=self.token)
            self._authenticated = True
            print("✅ Successfully authenticated with Hugging Face")
        except Exception as e:
            raise RuntimeError(f"Authentication failed: {e}")
    
    def upload_dataset(self, 
                      dataset: TextScalingDataset,
                      repo_name: str,
                      username: str,
                      private: bool = True,
                      chunk_size: int = 50000) -> str:
        """
        Upload dataset to Hugging Face Hub.
        
        Args:
            dataset: TextScalingDataset to upload
            repo_name: Repository name on Hugging Face
            username: Hugging Face username
            private: Whether to make repository private
            chunk_size: Process data in chunks of this size
            
        Returns:
            Repository URL
        """
        if not self._authenticated:
            raise RuntimeError("Not authenticated. Call authenticate() first.")
        
        repo_id = f"{username}/{repo_name}"
        
        # Create repository
        try:
            create_repo(repo_id=repo_id, repo_type="dataset", private=private)
            print(f"Created repository: {repo_id}")
        except Exception as e:
            print(f"Repository might already exist: {e}")
        
        # Process data in chunks for large datasets
        df = dataset.df
        if len(df) > chunk_size:
            print(f"Large dataset detected. Processing in chunks of {chunk_size:,}...")
            
            # Create splits based on training.set if available
            if 'training.set' in df.columns:
                train_df = df[df['training.set'] == 1].copy()
                test_df = df[df['training.set'] == 0].copy()
                
                # Create validation from training
                val_size = int(0.1 * len(train_df))
                val_df = train_df.sample(n=val_size, random_state=42)
                train_df = train_df.drop(val_df.index)
                
                dataset_dict = DatasetDict({
                    'train': Dataset.from_pandas(train_df, preserve_index=False),
                    'validation': Dataset.from_pandas(val_df, preserve_index=False),
                    'test': Dataset.from_pandas(test_df, preserve_index=False)
                })
            else:
                # Simple train/test split
                train_size = int(0.8 * len(df))
                train_df = df[:train_size]
                test_df = df[train_size:]
                
                dataset_dict = DatasetDict({
                    'train': Dataset.from_pandas(train_df, preserve_index=False),
                    'test': Dataset.from_pandas(test_df, preserve_index=False)
                })
        else:
            dataset_dict = DatasetDict({
                'train': Dataset.from_pandas(df, preserve_index=False)
            })
        
        # Upload
        print(f"Uploading to {repo_id}...")
        try:
            dataset_dict.push_to_hub(
                repo_id,
                private=private,
                commit_message="Upload political text scaling dataset with congressional speeches and topic weights"
            )
            
            repo_url = f"https://huggingface.co/datasets/{repo_id}"
            print(f"✅ Upload successful! Dataset available at: {repo_url}")
            return repo_url
            
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            # Save locally as fallback
            dataset_dict.save_to_disk("text_scaling_dataset_backup")
            print("💾 Dataset saved locally as backup")
            raise


# Utility Functions
def load_dataset_splits(source: str = "mliliu/political-text-scaling") -> Tuple[TextScalingDataset, TextScalingDataset, TextScalingDataset]:
    """
    Load train, validation, and test splits of the text scaling dataset.
    
    Args:
        source: Dataset source (Hugging Face repo or local path)
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    train_dataset = TextScalingDataset(source=source, split="train")
    val_dataset = TextScalingDataset(source=source, split="validation") 
    test_dataset = TextScalingDataset(source=source, split="test")
    
    return train_dataset, val_dataset, test_dataset


def compute_topic_statistics(dataset: TextScalingDataset) -> Dict:
    """
    Compute detailed topic weight statistics for analysis.
    
    Args:
        dataset: TextScalingDataset to analyze
        
    Returns:
        Dictionary with comprehensive topic statistics
    """
    return dataset.get_topic_summary()


def create_quick_sample(source: str = "mliliu/political-text-scaling", 
                       sample_size: int = 1000,
                       congress_filter: Optional[int] = None) -> TextScalingDataset:
    """
    Create a quick sample dataset for development and testing.
    
    Args:
        source: Dataset source
        sample_size: Number of documents to sample
        congress_filter: If specified, filter to this congress number
        
    Returns:
        Sampled TextScalingDataset
    """
    filters = {}
    if congress_filter:
        filters['congno'] = lambda x: x == congress_filter
    
    return TextScalingDataset(
        source=source,
        split="train",
        filters=filters,
        sample_size=sample_size
    )


# Example usage and testing functions
def example_basic_usage():
    """Example: Basic dataset loading and analysis."""
    print("=== Basic Usage Example ===")
    
    # Load dataset with filters
    dataset = TextScalingDataset(
        source="mliliu/political-text-scaling",
        split="train",
        filters={'congno': lambda x: x >= 110},  # Recent congresses only
        sample_size=1000  # Sample for demo
    )
    
    print(f"Dataset size: {len(dataset):,}")
    print(f"Topic columns: {len(dataset.topic_cols)}")
    
    # Get topic summary
    topic_stats = dataset.get_topic_summary()
    if topic_stats:
        print("\nTop 3 dominant topics:")
        for topic, count in list(topic_stats.get('dominant_topic_counts', {}).items())[:3]:
            clean_name = topic.replace('tw.', '').replace('.', ' ').title()
            print(f"  {clean_name}: {count} documents")
    
    # Get sample documents
    sample = dataset.get_sample(3)
    print(f"\nSample document preview:")
    for i, (_, doc) in enumerate(sample.iterrows()):
        print(f"  Doc {i+1}: {doc.get('text', '')[:100]}...")


def example_processing_pipeline():
    """Example: Full processing pipeline."""
    print("\n=== Processing Pipeline Example ===")
    
    # Create processor and load data
    processor = TextScalingProcessor().load_from_source(
        "mliliu/political-text-scaling", 
        split="train"
    )
    
    # Clean and optimize
    processor.clean_data()
    
    # Create filtered versions
    filter_info = processor.create_filtered_versions("demo_filtered")
    print(f"Created {len(filter_info)} filtered versions")
    
    # Compute statistics
    stats = processor.compute_comprehensive_statistics()
    print(f"Statistics computed for {stats['basic_stats']['total_speeches']:,} speeches")


if __name__ == "__main__":
    # Run examples
    example_basic_usage()
    example_processing_pipeline()