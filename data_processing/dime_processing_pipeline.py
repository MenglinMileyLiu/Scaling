#!/usr/bin/env python3
"""
DIME Dataset Processing Pipeline
Handles format optimization, sharding, and data preparation
"""

import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
import pyarrow as pa
import pyarrow.parquet as pq
import os
import numpy as np
from typing import Dict, List, Optional
import json

class DIMEProcessor:
    def __init__(self, dataset_name: str = "mliliu/dime-recipients"):
        self.dataset_name = dataset_name
        self.dataset = None
        self.df = None
        
    def load_data(self):
        """Load the dataset from Hugging Face"""
        print("Loading dataset from Hugging Face...")
        self.dataset = load_dataset(self.dataset_name)
        self.df = self.dataset['train'].to_pandas()
        print(f"Loaded {len(self.df):,} recipients")
        return self.df
    
    def clean_data(self):
        """Clean and standardize the data"""
        print("\nCleaning data...")
        
        # 1. Standardize party codes
        party_mapping = {
            '100': 'D', '200': 'R', '328': 'I',
            '400': 'G', '500': 'L', 'UNK': 'U',
            '': 'U', 'nan': 'U'
        }
        self.df['party_clean'] = self.df['party'].astype(str).map(
            lambda x: party_mapping.get(x, 'O')
        )
        
        # 2. Clean CF scores
        self.df['cf_score'] = pd.to_numeric(
            self.df['recipient.cfscore'], 
            errors='coerce'
        )
        
        # 3. Extract office type from nimsp.office
        if 'nimsp.office' in self.df.columns:
            self.df['office_type'] = self.df['nimsp.office'].fillna('other')
        else:
            self.df['office_type'] = 'unknown'
        
        # 4. Create decade column
        self.df['decade'] = (self.df['cycle'] // 10) * 10
        
        print(f"Cleaned {len(self.df):,} records")
        return self.df
    
    def optimize_format(self, output_dir: str = "dime_optimized"):
        """Convert to optimized formats"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Save as Parquet (best for analytics)
        print("\nSaving as Parquet...")
        parquet_path = os.path.join(output_dir, "dime_recipients.parquet")
        self.df.to_parquet(parquet_path, compression='snappy')
        parquet_size = os.path.getsize(parquet_path) / (1024**2)
        print(f"Parquet size: {parquet_size:.1f} MB")
        
        # 2. Save as compressed CSV
        print("\nSaving as compressed CSV...")
        csv_path = os.path.join(output_dir, "dime_recipients.csv.gz")
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
    
    def create_filtered_versions(self, output_dir: str = "dime_filtered"):
        """Create commonly used filtered versions"""
        os.makedirs(output_dir, exist_ok=True)
        
        filters = {
            'recent': self.df[self.df['cycle'] >= 2010],
            'candidates_only': self.df[self.df['recipient.type'] == 'cand'],
            'with_cf_scores': self.df[self.df['cf_score'].notna()],
            'federal_candidates': self.df[self.df['office_type'].isin(['house', 'senate'])],
            'house_candidates': self.df[self.df['office_type'] == 'house'],
            'senate_candidates': self.df[self.df['office_type'] == 'senate'],
            'major_parties': self.df[self.df['party_clean'].isin(['D', 'R'])]
        }
        
        filter_info = {}
        for name, filtered_df in filters.items():
            path = os.path.join(output_dir, f"dime_{name}.parquet")
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
                'total_recipients': len(self.df),
                'unique_recipients': self.df['bonica.rid'].nunique(),
                'time_range': f"{self.df['cycle'].min()}-{self.df['cycle'].max()}",
                'total_receipts': self.df['total.receipts'].sum() if 'total.receipts' in self.df.columns else None,
                'total_individual_contribs': self.df['total.indiv.contribs'].sum() if 'total.indiv.contribs' in self.df.columns else None
            },
            'party_distribution': self.df['party_clean'].value_counts().to_dict(),
            'office_distribution': self.df['office_type'].value_counts().to_dict(),
            'cf_score_stats': {
                'mean': float(self.df['cf_score'].mean()),
                'std': float(self.df['cf_score'].std()),
                'min': float(self.df['cf_score'].min()),
                'max': float(self.df['cf_score'].max()),
                'percentiles': {
                    '25%': float(self.df['cf_score'].quantile(0.25)),
                    '50%': float(self.df['cf_score'].quantile(0.50)),
                    '75%': float(self.df['cf_score'].quantile(0.75))
                }
            },
            'temporal_distribution': self.df.groupby('cycle').size().to_dict(),
            'state_coverage': self.df['state'].nunique(),
            'missing_data': {
                col: float(self.df[col].isna().sum() / len(self.df))
                for col in ['cf_score', 'party', 'state', 'district']
            }
        }
        
        # Save statistics
        with open('dime_statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        print("Statistics saved to dime_statistics.json")
        return stats

if __name__ == "__main__":
    # Initialize processor
    processor = DIMEProcessor()
    
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
    
    print("\n✅ Processing pipeline complete!")
    print("Created:")
    print("- Optimized formats (Parquet and CSV)")
    print("- Sharded dataset for distributed processing")
    print("- Filtered versions for common use cases")
    print("- Comprehensive statistics")