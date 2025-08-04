#!/usr/bin/env python3
"""
Political Text Scaling Dataset Classes

Modular dataset classes for congressional speech analysis following clean
architectural patterns. Provides flexible loading, filtering, and analysis
capabilities for text scaling research.

Classes:
    TextScalingDataset: Main dataset wrapper with comprehensive functionality
    StreamingTextDataset: Memory-efficient streaming for large datasets
    BatchDataLoader: Optimized batch processing with PyTorch integration
"""

import pandas as pd
import numpy as np
import random
from typing import Dict, List, Optional, Union, Iterator, Tuple, Any
from pathlib import Path

# Core data libraries
from datasets import load_dataset, Dataset
import pyarrow.parquet as pq

# Optional PyTorch support
try:
    from torch.utils.data import DataLoader, IterableDataset
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TextScalingDataset:
    """
    Primary dataset class for political text scaling data.
    
    Provides comprehensive functionality for loading, filtering, and analyzing
    congressional speech data with topic weights. Supports multiple data sources
    and flexible filtering options.
    
    Example:
        >>> dataset = TextScalingDataset(
        ...     source="mliliu/political-text-scaling",
        ...     filters={'congno': lambda x: x >= 110},
        ...     sample_size=1000
        ... )
        >>> print(f"Loaded {len(dataset)} speeches")
    """
    
    def __init__(self, 
                 source: str = "mliliu/political-text-scaling",
                 split: str = "train",
                 filters: Optional[Dict[str, Any]] = None,
                 columns: Optional[List[str]] = None,
                 sample_size: Optional[int] = None,
                 random_seed: int = 42):
        """
        Initialize the text scaling dataset.
        
        Args:
            source: HuggingFace dataset name, local file path, or URL
            split: Dataset split to load (train/validation/test)
            filters: Column filters as {column: filter_function} pairs
            columns: Specific columns to keep (None = all columns)
            sample_size: Random sample size (None = full dataset)
            random_seed: Random seed for reproducible sampling
        """
        self.source = source
        self.split = split
        self.filters = filters or {}
        self.columns = columns
        self.sample_size = sample_size
        self.random_seed = random_seed
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        if TORCH_AVAILABLE:
            torch.manual_seed(random_seed)
        
        # Load and process data
        self._load_data()
        self._process_topic_columns()
        self._validate_data()
        
    def _load_data(self):
        """Load data from the specified source."""
        print(f"Loading data from {self.source} (split: {self.split})...")
        
        try:
            if self.source.startswith("mliliu/") or "/" in self.source:
                # Load from HuggingFace Hub
                try:
                    dataset = load_dataset(self.source)
                    self.df = dataset[self.split].to_pandas()
                except Exception:
                    # Try with authentication token
                    dataset = load_dataset(self.source, token=True)
                    self.df = dataset[self.split].to_pandas()
                    
            elif Path(self.source).suffix == '.parquet':
                self.df = pd.read_parquet(self.source)
                
            elif Path(self.source).suffix == '.csv':
                # Handle large CSV files efficiently
                if Path(self.source).stat().st_size > 1024**3:  # > 1GB
                    print("Large CSV detected, loading in chunks...")
                    chunks = pd.read_csv(self.source, chunksize=50000)
                    self.df = pd.concat(chunks, ignore_index=True)
                else:
                    self.df = pd.read_csv(self.source)
                    
            else:
                raise ValueError(f"Unsupported source format: {self.source}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load data from {self.source}: {e}")
        
        print(f"Loaded {len(self.df):,} documents")
        
        # Apply filters if specified
        self._apply_filters()
        
        # Select specific columns if specified
        self._select_columns()
        
        # Apply sampling if specified
        self._apply_sampling()
    
    def _apply_filters(self):
        """Apply filtering conditions to the dataset."""
        if not self.filters:
            return
        
        original_size = len(self.df)
        print("Applying filters...")
        
        for column, filter_func in self.filters.items():
            if column not in self.df.columns:
                print(f"Warning: Filter column '{column}' not found in dataset")
                continue
                
            # Apply filter function
            mask = self.df[column].apply(filter_func)
            self.df = self.df[mask].reset_index(drop=True)
            
        filtered_out = original_size - len(self.df)
        if filtered_out > 0:
            print(f"Filtered out {filtered_out:,} documents, {len(self.df):,} remaining")
    
    def _select_columns(self):
        """Select specific columns if specified."""
        if not self.columns:
            return
        
        # Ensure requested columns exist
        available_columns = [col for col in self.columns if col in self.df.columns]
        missing_columns = set(self.columns) - set(available_columns)
        
        if missing_columns:
            print(f"Warning: Columns not found: {missing_columns}")
        
        if available_columns:
            self.df = self.df[available_columns]
            print(f"Selected {len(available_columns)} columns")
    
    def _apply_sampling(self):
        """Apply random sampling if specified."""
        if not self.sample_size or len(self.df) <= self.sample_size:
            return
        
        self.df = self.df.sample(
            n=self.sample_size, 
            random_state=self.random_seed
        ).reset_index(drop=True)
        
        print(f"Randomly sampled {self.sample_size:,} documents")
    
    def _process_topic_columns(self):
        """Extract and process topic weight columns."""
        # Identify topic weight columns
        self.topic_cols = [col for col in self.df.columns if col.startswith('tw.')]
        
        if self.topic_cols:
            print(f"Found {len(self.topic_cols)} topic weight columns")
            
            # Add derived columns for analysis
            self.df['dominant_topic'] = self.df[self.topic_cols].idxmax(axis=1)
            self.df['max_topic_weight'] = self.df[self.topic_cols].max(axis=1)
            self.df['topic_entropy'] = self._calculate_topic_entropy()
            
            # Count active topics per document
            active_threshold = 0.01
            active_topics = (self.df[self.topic_cols] > active_threshold).sum(axis=1)
            self.df['active_topic_count'] = active_topics
        else:
            print("No topic weight columns found")
    
    def _calculate_topic_entropy(self) -> pd.Series:
        """Calculate entropy of topic distributions for each document."""
        if not self.topic_cols:
            return pd.Series(index=self.df.index, dtype=float)
        
        # Normalize topic weights to probabilities
        topic_weights = self.df[self.topic_cols].values
        topic_probs = topic_weights / (topic_weights.sum(axis=1, keepdims=True) + 1e-10)
        
        # Calculate entropy: -sum(p * log(p))
        entropy = -np.sum(topic_probs * np.log(topic_probs + 1e-10), axis=1)
        return pd.Series(entropy, index=self.df.index)
    
    def _validate_data(self):
        """Validate loaded data quality."""
        if len(self.df) == 0:
            raise ValueError("Dataset is empty after loading and filtering")
        
        # Check for required columns
        required_cols = ['text'] if 'text' in self.df.columns else []
        missing_required = [col for col in required_cols if col not in self.df.columns]
        
        if missing_required:
            raise ValueError(f"Missing required columns: {missing_required}")
        
        print(f"✅ Dataset validation passed: {len(self.df):,} documents ready")
    
    def __len__(self) -> int:
        """Return the number of documents in the dataset."""
        return len(self.df)
    
    def __getitem__(self, idx: Union[int, slice, List[int]]) -> Union[Dict, pd.DataFrame]:
        """
        Get document(s) by index.
        
        Args:
            idx: Index, slice, or list of indices
            
        Returns:
            Single document dict or DataFrame for multiple documents
        """
        if isinstance(idx, int):
            return self.df.iloc[idx].to_dict()
        elif isinstance(idx, (slice, list)):
            return self.df.iloc[idx]
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")
    
    def get_sample(self, n: int = 5, random_state: Optional[int] = None) -> pd.DataFrame:
        """
        Get a random sample of documents.
        
        Args:
            n: Number of documents to sample
            random_state: Random seed for sampling
            
        Returns:
            DataFrame with sampled documents
        """
        if random_state is None:
            random_state = self.random_seed
        
        sample_size = min(n, len(self.df))
        return self.df.sample(n=sample_size, random_state=random_state)
    
    def get_topic_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive topic analysis summary.
        
        Returns:
            Dictionary with topic statistics and distributions
        """
        if not self.topic_cols:
            return {"message": "No topic columns available"}
        
        summary = {}
        
        # Basic topic statistics
        topic_stats = {}
        for topic in self.topic_cols:
            topic_stats[topic] = {
                'mean': float(self.df[topic].mean()),
                'std': float(self.df[topic].std()),
                'max': float(self.df[topic].max()),
                'min': float(self.df[topic].min()),
                'active_docs': int((self.df[topic] > 0.01).sum()),
                'high_weight_docs': int((self.df[topic] > 0.1).sum())
            }
        
        summary['topic_statistics'] = topic_stats
        
        # Dominant topic distribution
        if 'dominant_topic' in self.df.columns:
            dominant_counts = self.df['dominant_topic'].value_counts()
            summary['dominant_topic_distribution'] = dominant_counts.head(10).to_dict()
        
        # Topic diversity metrics
        if 'topic_entropy' in self.df.columns:
            summary['topic_diversity'] = {
                'mean_entropy': float(self.df['topic_entropy'].mean()),
                'entropy_std': float(self.df['topic_entropy'].std()),
                'low_diversity_docs': int((self.df['topic_entropy'] < 1.0).sum()),
                'high_diversity_docs': int((self.df['topic_entropy'] > 2.0).sum())
            }
        
        # Active topic statistics
        if 'active_topic_count' in self.df.columns:
            summary['active_topics'] = {
                'mean_active_topics': float(self.df['active_topic_count'].mean()),
                'single_topic_docs': int((self.df['active_topic_count'] == 1).sum()),
                'multi_topic_docs': int((self.df['active_topic_count'] > 3).sum())
            }
        
        return summary
    
    def filter_by_topic(self, topic: str, min_weight: float = 0.1) -> 'TextScalingDataset':
        """
        Create a new dataset filtered by topic weight.
        
        Args:
            topic: Topic column name (e.g., 'tw.healthcare')
            min_weight: Minimum topic weight threshold
            
        Returns:
            New TextScalingDataset with filtered documents
        """
        if topic not in self.topic_cols:
            raise ValueError(f"Topic '{topic}' not found in dataset")
        
        # Create new dataset with topic filter
        topic_filter = {topic: lambda x: x >= min_weight}
        combined_filters = {**self.filters, **topic_filter}
        
        return TextScalingDataset(
            source=self.source,
            split=self.split,
            filters=combined_filters,
            columns=self.columns,
            random_seed=self.random_seed
        )
    
    def iterate_batches(self, 
                       batch_size: int = 32, 
                       shuffle: bool = True) -> Iterator[pd.DataFrame]:
        """
        Iterate through the dataset in batches.
        
        Args:
            batch_size: Number of documents per batch
            shuffle: Whether to shuffle the data
            
        Yields:
            DataFrame batches
        """
        indices = list(range(len(self.df)))
        
        if shuffle:
            random.shuffle(indices)
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            yield self.df.iloc[batch_indices]
    
    def split_by_column(self, column: str) -> Dict[Any, 'TextScalingDataset']:
        """
        Split dataset by unique values in a column.
        
        Args:
            column: Column name to split by
            
        Returns:
            Dictionary mapping column values to datasets
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in dataset")
        
        splits = {}
        for value in self.df[column].unique():
            value_filter = {column: lambda x, v=value: x == v}
            combined_filters = {**self.filters, **value_filter}
            
            splits[value] = TextScalingDataset(
                source=self.source,
                split=self.split,
                filters=combined_filters,
                columns=self.columns,
                random_seed=self.random_seed
            )
        
        return splits
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get comprehensive dataset information.
        
        Returns:
            Dictionary with dataset metadata and statistics
        """
        info = {
            'source': self.source,
            'split': self.split,
            'size': len(self.df),
            'columns': list(self.df.columns),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / (1024**2)
        }
        
        # Add column type information
        info['column_types'] = self.df.dtypes.astype(str).to_dict()
        
        # Add topic information if available
        if self.topic_cols:
            info['topic_columns'] = len(self.topic_cols)
            info['topic_names'] = self.topic_cols
        
        # Add text statistics if text column exists
        if 'text' in self.df.columns:
            text_lengths = self.df['text'].str.len()
            info['text_statistics'] = {
                'mean_length': float(text_lengths.mean()),
                'median_length': float(text_lengths.median()),
                'min_length': int(text_lengths.min()),
                'max_length': int(text_lengths.max())
            }
        
        # Add temporal information if available
        if 'date' in self.df.columns:
            dates = pd.to_datetime(self.df['date'], errors='coerce')
            info['temporal_range'] = {
                'start_date': str(dates.min().date()) if dates.notna().any() else None,
                'end_date': str(dates.max().date()) if dates.notna().any() else None
            }
        
        # Add congress information if available
        if 'congno' in self.df.columns:
            congress_nums = self.df['congno'].dropna()
            if len(congress_nums) > 0:
                info['congress_range'] = {
                    'min_congress': int(congress_nums.min()),
                    'max_congress': int(congress_nums.max()),
                    'unique_congresses': congress_nums.nunique()
                }
        
        return info


class StreamingTextDataset:
    """
    Memory-efficient streaming dataset for large text collections.
    
    Loads data in chunks to handle datasets too large for memory.
    Useful for processing very large congressional speech collections.
    """
    
    def __init__(self,
                 shard_directory: Union[str, Path],
                 batch_size: int = 32,
                 shuffle: bool = True):
        """
        Initialize streaming dataset.
        
        Args:
            shard_directory: Directory containing parquet shards
            batch_size: Number of documents per batch
            shuffle: Whether to shuffle shard order
        """
        self.shard_dir = Path(shard_directory)
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Find all parquet files
        self.shard_files = sorted(list(self.shard_dir.glob("*.parquet")))
        
        if not self.shard_files:
            raise ValueError(f"No parquet files found in {shard_directory}")
        
        print(f"Found {len(self.shard_files)} shards for streaming")
    
    def __iter__(self) -> Iterator[pd.DataFrame]:
        """Iterate through shards yielding batches."""
        shard_order = list(range(len(self.shard_files)))
        
        if self.shuffle:
            random.shuffle(shard_order)
        
        for shard_idx in shard_order:
            shard_path = self.shard_files[shard_idx]
            
            try:
                shard_df = pd.read_parquet(shard_path)
                
                # Shuffle within shard if requested
                if self.shuffle:
                    shard_df = shard_df.sample(frac=1).reset_index(drop=True)
                
                # Yield batches from this shard
                for i in range(0, len(shard_df), self.batch_size):
                    batch = shard_df.iloc[i:i + self.batch_size]
                    yield batch
                    
            except Exception as e:
                print(f"Warning: Failed to load shard {shard_path}: {e}")
                continue


class BatchDataLoader:
    """
    Optimized batch data loader with PyTorch integration.
    
    Provides efficient batching with optional PyTorch tensor conversion
    for machine learning workflows.
    """
    
    def __init__(self,
                 dataset: TextScalingDataset,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 num_workers: int = 0,
                 convert_to_tensors: bool = False):
        """
        Initialize batch data loader.
        
        Args:
            dataset: TextScalingDataset to load from
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of workers for parallel loading
            convert_to_tensors: Convert numeric fields to PyTorch tensors
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.convert_to_tensors = convert_to_tensors and TORCH_AVAILABLE
        
        if convert_to_tensors and not TORCH_AVAILABLE:
            print("Warning: PyTorch not available, tensor conversion disabled")
    
    def get_pytorch_dataloader(self):
        """
        Get PyTorch DataLoader instance.
        
        Returns:
            Configured PyTorch DataLoader
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available for DataLoader")
        
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self._collate_batch
        )
    
    def _collate_batch(self, batch: List[Dict]) -> Dict[str, Any]:
        """
        Collate function for PyTorch DataLoader.
        
        Args:
            batch: List of document dictionaries
            
        Returns:
            Collated batch dictionary
        """
        if not batch:
            return {}
        
        collated = {}
        
        for key in batch[0].keys():
            values = [item[key] for item in batch]
            
            # Convert numeric fields to tensors if requested
            if (self.convert_to_tensors and 
                (key.startswith('tw.') or key in ['congno', 'training.set'])):
                try:
                    collated[key] = torch.tensor(values, dtype=torch.float32)
                except (ValueError, TypeError):
                    collated[key] = values  # Keep as list if conversion fails
            else:
                collated[key] = values
        
        return collated
    
    def iterate_batches(self) -> Iterator[Dict[str, Any]]:
        """
        Iterate through batches without PyTorch dependency.
        
        Yields:
            Batch dictionaries
        """
        for batch_df in self.dataset.iterate_batches(
            batch_size=self.batch_size, 
            shuffle=self.shuffle
        ):
            # Convert DataFrame batch to list of dicts
            batch_list = batch_df.to_dict('records')
            
            # Use collate function to process batch
            yield self._collate_batch(batch_list)


# Utility functions for dataset operations
def load_dataset_splits(source: str = "mliliu/political-text-scaling",
                       **kwargs) -> Tuple[TextScalingDataset, TextScalingDataset, TextScalingDataset]:
    """
    Load train, validation, and test splits.
    
    Args:
        source: Dataset source
        **kwargs: Additional arguments for TextScalingDataset
        
    Returns:
        Tuple of (train, validation, test) datasets
    """
    train_ds = TextScalingDataset(source=source, split="train", **kwargs)
    val_ds = TextScalingDataset(source=source, split="validation", **kwargs)
    test_ds = TextScalingDataset(source=source, split="test", **kwargs)
    
    return train_ds, val_ds, test_ds


def create_topic_focused_datasets(source: str = "mliliu/political-text-scaling",
                                 topics: List[str] = None,
                                 min_weight: float = 0.1) -> Dict[str, TextScalingDataset]:
    """
    Create datasets focused on specific topics.
    
    Args:
        source: Dataset source
        topics: List of topic names (e.g., ['tw.healthcare', 'tw.economy'])
        min_weight: Minimum topic weight threshold
        
    Returns:
        Dictionary mapping topic names to datasets
    """
    if topics is None:
        topics = ['tw.healthcare', 'tw.economy', 'tw.defense.and.foreign.policy']
    
    # Load base dataset to get available topics
    base_dataset = TextScalingDataset(source=source, sample_size=100)
    available_topics = [t for t in topics if t in base_dataset.topic_cols]
    
    if not available_topics:
        raise ValueError(f"None of the specified topics found in dataset: {topics}")
    
    topic_datasets = {}
    for topic in available_topics:
        topic_name = topic.replace('tw.', '').replace('.', '_')
        topic_datasets[topic_name] = base_dataset.filter_by_topic(topic, min_weight)
    
    return topic_datasets


if __name__ == "__main__":
    # Example usage
    print("🧪 Testing TextScalingDataset...")
    
    try:
        # Load a small sample for testing
        dataset = TextScalingDataset(
            source="mliliu/political-text-scaling",
            split="train",
            sample_size=100
        )
        
        print(f"✅ Loaded dataset: {len(dataset)} documents")
        print(f"📊 Topics: {len(dataset.topic_cols)} columns")
        
        # Test topic analysis
        topic_summary = dataset.get_topic_summary()
        if 'dominant_topic_distribution' in topic_summary:
            print("🏷️  Top topics:")
            for topic, count in list(topic_summary['dominant_topic_distribution'].items())[:3]:
                clean_name = topic.replace('tw.', '').replace('.', ' ').title()
                print(f"   {clean_name}: {count} docs")
        
        print("✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")