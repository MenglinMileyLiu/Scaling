#!/usr/bin/env python3
"""
Political Text Scaling Dataset Processing Pipeline
Handles format optimization, analysis, and data preparation for congressional speeches
"""

import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
import pyarrow as pa
import pyarrow.parquet as pq
import os
import numpy as np
from typing import Dict, List, Optional
import json

class TextScalingProcessor:
    def __init__(self, dataset_name: str = "mliliu/political-text-scaling"):
        self.dataset_name = dataset_name
        self.dataset = None
        self.df = None
        
    def load_data(self):
        """Load the dataset from Hugging Face"""
        print("Loading dataset from Hugging Face...")
        self.dataset = load_dataset(self.dataset_name)
        self.df = self.dataset['train'].to_pandas()
        print(f"Loaded {len(self.df):,} congressional speeches")
        return self.df
    
    def clean_data(self):
        """Clean and standardize the text scaling data"""
        print("\nCleaning data...")
        
        # 1. Clean text fields
        for text_col in ['text', 'stemmed.text']:
            if text_col in self.df.columns:
                self.df[text_col] = self.df[text_col].fillna('').astype(str)
        
        # 2. Parse dates
        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
            self.df['year'] = self.df['date'].dt.year
        
        # 3. Clean congress numbers
        if 'congno' in self.df.columns:
            self.df['congno'] = pd.to_numeric(self.df['congno'], errors='coerce')
        
        # 4. Extract topic weights
        self.topic_cols = [col for col in self.df.columns if col.startswith('tw.')]
        print(f"Found {len(self.topic_cols)} topic weight columns")
        
        # 5. Create dominant topic column
        if self.topic_cols:
            self.df['dominant_topic'] = self.df[self.topic_cols].idxmax(axis=1)
            self.df['max_topic_weight'] = self.df[self.topic_cols].max(axis=1)
        
        print(f"Cleaned {len(self.df):,} documents")
        return self.df
    
    def optimize_format(self, output_dir: str = "text_scaling_optimized"):
        """Convert to optimized formats"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Save as Parquet (best for analytics)
        print("\nSaving as Parquet...")
        parquet_path = os.path.join(output_dir, "congressional_speeches.parquet")
        self.df.to_parquet(parquet_path, compression='snappy')
        parquet_size = os.path.getsize(parquet_path) / (1024**2)
        print(f"Parquet size: {parquet_size:.1f} MB")
        
        # 2. Save as compressed CSV
        print("\nSaving as compressed CSV...")
        csv_path = os.path.join(output_dir, "congressional_speeches.csv.gz")
        self.df.to_csv(csv_path, compression='gzip', index=False)
        csv_size = os.path.getsize(csv_path) / (1024**2)
        print(f"Compressed CSV size: {csv_size:.1f} MB")
        
        # 3. Create sharded version for large-scale processing
        print("\nCreating sharded dataset...")
        self.create_shards(output_dir)
        
        return {
            'parquet_path': parquet_path,
            'csv_path': csv_path,
            'parquet_size_mb': parquet_size,
            'csv_size_mb': csv_size
        }
    
    def create_shards(self, output_dir: str, shard_size: int = 100000):
        """Create sharded dataset for distributed processing"""
        shard_dir = os.path.join(output_dir, "shards")
        os.makedirs(shard_dir, exist_ok=True)
        
        n_shards = len(self.df) // shard_size + 1
        
        for i in range(n_shards):
            start_idx = i * shard_size
            end_idx = min((i + 1) * shard_size, len(self.df))
            
            shard_df = self.df.iloc[start_idx:end_idx]
            shard_path = os.path.join(shard_dir, f"shard_{i:04d}.parquet")
            shard_df.to_parquet(shard_path)
            
            if i % 10 == 0:
                print(f"Created shard {i+1}/{n_shards}")
        
        print(f"Created {n_shards} shards in {shard_dir}")
        return n_shards
    
    def create_filtered_versions(self, output_dir: str = "text_scaling_filtered"):
        """Create commonly used filtered versions"""
        os.makedirs(output_dir, exist_ok=True)
        
        filters = {}
        
        # Filter by legislative body
        if 'legis.body' in self.df.columns:
            filters['house_speeches'] = self.df[self.df['legis.body'] == 'US House']
            filters['senate_speeches'] = self.df[self.df['legis.body'] == 'US Senate']
        
        # Filter by congress number
        if 'congno' in self.df.columns:
            filters['recent_congress'] = self.df[self.df['congno'] >= 110]  # 110th Congress and after
            filters['congress_108'] = self.df[self.df['congno'] == 108]
            filters['congress_109'] = self.df[self.df['congno'] == 109]
        
        # Filter by training set
        if 'training.set' in self.df.columns:
            filters['training_speeches'] = self.df[self.df['training.set'] == 1]
            filters['test_speeches'] = self.df[self.df['training.set'] == 0]
        
        # Filter by dominant topics
        if hasattr(self, 'topic_cols') and self.topic_cols:
            for topic in ['tw.healthcare', 'tw.defense.and.foreign.policy', 'tw.economy']:
                if topic in self.df.columns:
                    topic_name = topic.replace('tw.', '').replace('.', '_')
                    filters[f'{topic_name}_focused'] = self.df[self.df[topic] > 0.1]
        
        filter_info = {}
        for name, filtered_df in filters.items():
            if len(filtered_df) > 0:
                path = os.path.join(output_dir, f"speeches_{name}.parquet")
                filtered_df.to_parquet(path)
                filter_info[name] = {
                    'rows': len(filtered_df),
                    'path': path,
                    'size_mb': os.path.getsize(path) / (1024**2)
                }
                print(f"Created {name} version: {len(filtered_df):,} rows")
        
        return filter_info
    
    def compute_statistics(self) -> Dict:
        """Compute comprehensive dataset statistics"""
        print("\nComputing dataset statistics...")
        
        stats = {
            'basic_stats': {
                'total_speeches': len(self.df),
                'unique_documents': self.df['doc.id'].nunique() if 'doc.id' in self.df.columns else len(self.df),
                'unique_legislators': self.df['bonica.rid'].nunique() if 'bonica.rid' in self.df.columns else None,
                'congress_range': f"{self.df['congno'].min()}-{self.df['congno'].max()}" if 'congno' in self.df.columns else None,
                'date_range': f"{self.df['date'].min().date()}-{self.df['date'].max().date()}" if 'date' in self.df.columns else None
            },
            'legislative_body_distribution': self.df['legis.body'].value_counts().to_dict() if 'legis.body' in self.df.columns else {},
            'congress_distribution': self.df['congno'].value_counts().to_dict() if 'congno' in self.df.columns else {},
            'training_split': self.df['training.set'].value_counts().to_dict() if 'training.set' in self.df.columns else {},
            'text_length_stats': {
                'mean_chars': float(self.df['text'].str.len().mean()) if 'text' in self.df.columns else None,
                'median_chars': float(self.df['text'].str.len().median()) if 'text' in self.df.columns else None,
                'mean_words': float(self.df['text'].str.split().str.len().mean()) if 'text' in self.df.columns else None
            },
            'missing_data': {
                col: float(self.df[col].isna().sum() / len(self.df))
                for col in ['text', 'date', 'congno', 'bonica.rid'] if col in self.df.columns
            }
        }
        
        # Add topic weight statistics if available
        if hasattr(self, 'topic_cols') and self.topic_cols:
            topic_stats = {}
            for topic in self.topic_cols:
                topic_stats[topic] = {
                    'mean': float(self.df[topic].mean()),
                    'std': float(self.df[topic].std()),
                    'max': float(self.df[topic].max())
                }
            stats['topic_weight_stats'] = topic_stats
            
            # Most common dominant topics
            if 'dominant_topic' in self.df.columns:
                stats['dominant_topic_distribution'] = self.df['dominant_topic'].value_counts().head(10).to_dict()
        
        # Save statistics
        with open('text_scaling_statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        print("Statistics saved to text_scaling_statistics.json")
        return stats

if __name__ == "__main__":
    # Initialize processor
    processor = TextScalingProcessor()
    
    # Run full pipeline
    processor.load_data()
    processor.clean_data()
    
    # Optimize formats
    format_info = processor.optimize_format()
    print(f"\nFormat comparison:")
    print(f"Parquet: {format_info['parquet_size_mb']:.1f} MB")
    print(f"CSV.gz: {format_info['csv_size_mb']:.1f} MB")
    print(f"Savings: {(1 - format_info['parquet_size_mb']/format_info['csv_size_mb'])*100:.1f}%")
    
    # Create filtered versions
    filter_info = processor.create_filtered_versions()
    
    # Compute statistics
    stats = processor.compute_statistics()
    
    print("\n✅ Text scaling processing pipeline complete!")
    print("Created:")
    print("- Optimized formats (Parquet and CSV)")
    print("- Sharded dataset for distributed processing")
    print("- Filtered versions for common text analysis use cases")
    print("- Comprehensive statistics including topic distributions")