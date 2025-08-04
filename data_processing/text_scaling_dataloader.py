#!/usr/bin/env python3
"""
Political Text Scaling Dataset DataLoader with Batch and Streaming Support
"""

import pandas as pd
from datasets import load_dataset, Dataset
from typing import Iterator, Dict, List, Optional, Union
import pyarrow.parquet as pq
import os
import random

# Optional PyTorch imports
try:
    from torch.utils.data import DataLoader, IterableDataset
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Using pandas-only implementation.")

class TextScalingDataset:
    """Dataset wrapper for political text scaling data"""
    
    def __init__(self, 
                 source: str = "mliliu/political-text-scaling",
                 filters: Optional[Dict] = None,
                 columns: Optional[List[str]] = None):
        """
        Args:
            source: HuggingFace dataset name or local file path
            filters: Dict of column filters e.g. {'congno': lambda x: x >= 110}
            columns: List of columns to keep (None = all columns)
        """
        self.source = source
        self.filters = filters
        self.columns = columns
        self._load_data()
    
    def _load_data(self):
        """Load data from source"""
        if self.source.startswith("mliliu/"):
            # Load from HuggingFace
            dataset = load_dataset(self.source)
            self.df = dataset['train'].to_pandas()
        elif self.source.endswith('.parquet'):
            # Load from local parquet
            self.df = pd.read_parquet(self.source)
        elif self.source.endswith('.csv'):
            # Load from local CSV
            self.df = pd.read_csv(self.source)
        else:
            raise ValueError(f"Unsupported source: {self.source}")
        
        # Apply filters
        if self.filters:
            for col, filter_func in self.filters.items():
                self.df = self.df[self.df[col].apply(filter_func)]
        
        # Select columns
        if self.columns:
            self.df = self.df[self.columns]
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return self.df.iloc[idx].to_dict()
    
    def get_batch(self, indices: List[int]) -> pd.DataFrame:
        """Get batch of data as DataFrame"""
        return self.df.iloc[indices]


class TextScalingStreamingDataset:
    """Streaming dataset for large text scaling data"""
    
    def __init__(self, 
                 shard_dir: str,
                 batch_size: int = 32,
                 shuffle: bool = True):
        """
        Args:
            shard_dir: Directory containing parquet shards
            batch_size: Batch size for iteration
            shuffle: Whether to shuffle shards
        """
        self.shard_dir = shard_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.shard_files = sorted([
            f for f in os.listdir(shard_dir) 
            if f.endswith('.parquet')
        ])
    
    def __iter__(self) -> Iterator[Dict]:
        """Iterate through shards"""
        if TORCH_AVAILABLE and self.shuffle:
            shard_order = torch.randperm(len(self.shard_files)).tolist()
        elif self.shuffle:
            shard_order = list(range(len(self.shard_files)))
            random.shuffle(shard_order)
        else:
            shard_order = range(len(self.shard_files))
        
        for shard_idx in shard_order:
            shard_path = os.path.join(self.shard_dir, self.shard_files[shard_idx])
            shard_df = pd.read_parquet(shard_path)
            
            # Shuffle within shard if needed
            if self.shuffle:
                shard_df = shard_df.sample(frac=1).reset_index(drop=True)
            
            # Yield batches
            for i in range(0, len(shard_df), self.batch_size):
                batch = shard_df.iloc[i:i+self.batch_size]
                yield batch.to_dict('records')


class TextScalingDataLoader:
    """Main dataloader class with multiple loading strategies"""
    
    def __init__(self, 
                 source: Union[str, TextScalingDataset],
                 batch_size: int = 32,
                 shuffle: bool = True,
                 streaming: bool = False,
                 num_workers: int = 0):
        """
        Args:
            source: Data source (dataset name, file path, or DIMEDataset)
            batch_size: Batch size
            shuffle: Shuffle data
            streaming: Use streaming mode for large data
            num_workers: Number of workers for parallel loading
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.streaming = streaming
        self.num_workers = num_workers
        
        if isinstance(source, TextScalingDataset):
            self.dataset = source
        else:
            self.dataset = TextScalingDataset(source)
    
    def get_pytorch_dataloader(self):
        """Get PyTorch DataLoader (if available)"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available. Use iterate_batches() instead.")
        
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch: List[Dict]) -> Dict:
        """Custom collate function"""
        # Convert to tensors where appropriate
        collated = {}
        for key in batch[0].keys():
            values = [item[key] for item in batch]
            
            # Convert numeric fields to tensors (if torch available)
            if TORCH_AVAILABLE and (key.startswith('tw.') or key in ['congno', 'training.set']):
                try:
                    collated[key] = torch.tensor(values, dtype=torch.float32)
                except:
                    collated[key] = values  # Keep as list if conversion fails
            else:
                collated[key] = values
        
        return collated
    
    def iterate_batches(self) -> Iterator[pd.DataFrame]:
        """Iterate through batches as DataFrames"""
        indices = list(range(len(self.dataset)))
        
        if self.shuffle:
            random.shuffle(indices)
        
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i+self.batch_size]
            yield self.dataset.get_batch(batch_indices)
    
    def get_sample(self, n: int = 5) -> pd.DataFrame:
        """Get sample of data"""
        if TORCH_AVAILABLE:
            indices = torch.randperm(len(self.dataset))[:n].tolist()
        else:
            indices = random.sample(range(len(self.dataset)), min(n, len(self.dataset)))
        return self.dataset.get_batch(indices)


# Example usage functions
def example_batch_loading():
    """Example: Load data in batches"""
    print("Example: Batch Loading")
    print("-" * 50)
    
    # Create dataloader with filters
    loader = TextScalingDataLoader(
        source="mliliu/political-text-scaling",
        batch_size=64,
        shuffle=True
    )
    
    # Get a sample
    sample = loader.get_sample(5)
    print(f"Sample data shape: {sample.shape}")
    print(f"Sample columns: {sample.columns.tolist()[:5]}...")
    
    # Iterate through first few batches
    for i, batch in enumerate(loader.iterate_batches()):
        print(f"Batch {i}: {batch.shape}")
        if i >= 2:  # Just show first 3 batches
            break


def example_filtered_loading():
    """Example: Load filtered data"""
    print("\nExample: Filtered Loading")
    print("-" * 50)
    
    # Create dataset with filters
    dataset = TextScalingDataset(
        source="mliliu/political-text-scaling",
        filters={
            'congno': lambda x: x >= 110,
            'legis.body': lambda x: str(x) == 'US House'
        },
        columns=['doc.id', 'text', 'date', 'congno', 'legis.body', 'tw.healthcare', 'tw.economy']
    )
    
    print(f"Filtered dataset size: {len(dataset)}")
    print(f"Columns: {dataset.df.columns.tolist()}")
    
    # Create loader
    loader = TextScalingDataLoader(dataset, batch_size=32)
    
    # Get sample
    sample = loader.get_sample(3)
    print("\nSample filtered data:")
    print(sample)


def example_streaming():
    """Example: Streaming large dataset"""
    print("\nExample: Streaming Mode")
    print("-" * 50)
    
    # Note: This requires sharded data created by the processing pipeline
    shard_dir = "text_scaling_optimized/shards"
    
    if os.path.exists(shard_dir):
        streaming_dataset = TextScalingStreamingDataset(
            shard_dir=shard_dir,
            batch_size=128,
            shuffle=True
        )
        
        # Process first few batches
        for i, batch in enumerate(streaming_dataset):
            print(f"Streaming batch {i}: {len(batch)} records")
            if i >= 2:
                break
    else:
        print("Sharded data not found. Run processing pipeline first.")


if __name__ == "__main__":
    # Run examples
    example_batch_loading()
    example_filtered_loading()
    example_streaming()