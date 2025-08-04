#!/usr/bin/env python3
"""
Political Text Scaling Data Processors

Comprehensive data processing pipelines for congressional speech analysis.
Handles cleaning, optimization, statistics computation, and format conversion
following modular design principles.

Classes:
    TextScalingProcessor: Main processing pipeline
    StatisticsComputer: Comprehensive statistical analysis
    FormatOptimizer: Data format optimization and conversion
    DataValidator: Data quality validation and reporting
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import warnings

# Import dataset classes
from .dataset import TextScalingDataset

# Optional libraries
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

try:
    from huggingface_hub import login, create_repo
    from datasets import Dataset, DatasetDict
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class TextScalingProcessor:
    """
    Main processing pipeline for text scaling datasets.
    
    Provides comprehensive data processing capabilities including cleaning,
    optimization, and statistical analysis. Designed for chaining operations
    and maintaining data provenance.
    
    Example:
        >>> processor = TextScalingProcessor()
        >>> processor.load_dataset("mliliu/political-text-scaling")
        >>> processor.clean_data().optimize_formats().compute_statistics()
    """
    
    def __init__(self, dataset: Optional[TextScalingDataset] = None):
        """
        Initialize processor.
        
        Args:
            dataset: Optional TextScalingDataset to process
        """
        self.dataset = dataset
        self.df = dataset.df if dataset else None
        self.topic_cols = dataset.topic_cols if dataset else []
        self.processing_history = []
        self.statistics = {}
        
    def load_dataset(self, 
                    source: str,
                    split: str = "train",
                    **kwargs) -> 'TextScalingProcessor':
        """
        Load dataset and return self for method chaining.
        
        Args:
            source: Dataset source (HuggingFace, local file, etc.)
            split: Dataset split to load
            **kwargs: Additional arguments for TextScalingDataset
            
        Returns:
            Self for method chaining
        """
        print(f"Loading dataset from {source}...")
        
        self.dataset = TextScalingDataset(source=source, split=split, **kwargs)
        self.df = self.dataset.df
        self.topic_cols = self.dataset.topic_cols
        
        self.processing_history.append({
            'operation': 'load_dataset',
            'source': source,
            'split': split,
            'size': len(self.df)
        })
        
        print(f"✅ Loaded {len(self.df):,} documents")
        return self
    
    def clean_data(self, 
                  fix_dates: bool = True,
                  standardize_text: bool = True,
                  handle_missing: bool = True) -> 'TextScalingProcessor':
        """
        Clean and standardize the dataset.
        
        Args:
            fix_dates: Parse and standardize date columns
            standardize_text: Clean and standardize text fields
            handle_missing: Handle missing values appropriately
            
        Returns:
            Self for method chaining
        """
        if self.df is None:
            raise ValueError("No dataset loaded. Use load_dataset() first.")
        
        print("Cleaning data...")
        original_size = len(self.df)
        
        # Clean text fields
        if standardize_text:
            self._clean_text_fields()
        
        # Parse and standardize dates
        if fix_dates:
            self._standardize_dates()
        
        # Handle missing values
        if handle_missing:
            self._handle_missing_values()
        
        # Clean numeric fields
        self._clean_numeric_fields()
        
        # Validate data after cleaning
        self._validate_after_cleaning()
        
        self.processing_history.append({
            'operation': 'clean_data',
            'original_size': original_size,
            'final_size': len(self.df),
            'removed': original_size - len(self.df)
        })
        
        print(f"✅ Data cleaning complete: {len(self.df):,} documents")
        return self
    
    def _clean_text_fields(self):
        """Clean text columns."""
        text_columns = ['text', 'stemmed.text']
        
        for col in text_columns:
            if col in self.df.columns:
                # Fill missing values with empty string
                self.df[col] = self.df[col].fillna('')
                
                # Convert to string type
                self.df[col] = self.df[col].astype(str)
                
                # Remove excessive whitespace
                self.df[col] = self.df[col].str.strip()
                self.df[col] = self.df[col].str.replace(r'\s+', ' ', regex=True)
        
        # Remove documents with empty text
        if 'text' in self.df.columns:
            empty_text_mask = (self.df['text'] == '') | (self.df['text'].str.len() < 10)
            if empty_text_mask.any():
                removed_count = empty_text_mask.sum()
                self.df = self.df[~empty_text_mask].reset_index(drop=True)
                print(f"Removed {removed_count} documents with empty/very short text")
    
    def _standardize_dates(self):
        """Parse and standardize date columns."""
        if 'date' in self.df.columns:
            # Parse dates with error handling
            self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')
            
            # Extract derived date features
            valid_dates = self.df['date'].notna()
            if valid_dates.any():
                self.df.loc[valid_dates, 'year'] = self.df.loc[valid_dates, 'date'].dt.year
                self.df.loc[valid_dates, 'month'] = self.df.loc[valid_dates, 'date'].dt.month
                self.df.loc[valid_dates, 'day_of_year'] = self.df.loc[valid_dates, 'date'].dt.dayofyear
                
                # Create decade column for temporal analysis
                self.df.loc[valid_dates, 'decade'] = (self.df.loc[valid_dates, 'year'] // 10) * 10
            
            invalid_dates = (~valid_dates).sum()
            if invalid_dates > 0:
                print(f"Warning: {invalid_dates} invalid dates found and set to NaT")
    
    def _handle_missing_values(self):
        """Handle missing values appropriately by column type."""
        # Categorical columns - fill with 'unknown' or most common value
        categorical_cols = ['legis.body', 'doc.type']
        for col in categorical_cols:
            if col in self.df.columns:
                if self.df[col].isna().any():
                    most_common = self.df[col].mode().iloc[0] if not self.df[col].mode().empty else 'unknown'
                    self.df[col] = self.df[col].fillna(most_common)
        
        # Numeric ID columns - keep as NaN or convert to string
        id_cols = ['doc.id', 'bonica.rid', 'bill.id', 'page.id']
        for col in id_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).replace('nan', '')
        
        # Topic weights - ensure no missing values
        for col in self.topic_cols:
            if self.df[col].isna().any():
                self.df[col] = self.df[col].fillna(0.0)
    
    def _clean_numeric_fields(self):
        """Clean and validate numeric fields."""
        # Clean congress numbers
        if 'congno' in self.df.columns:
            self.df['congno'] = pd.to_numeric(self.df['congno'], errors='coerce')
            
            # Validate congress numbers (should be reasonable range)
            valid_congress = (self.df['congno'] >= 100) & (self.df['congno'] <= 120)
            invalid_congress = (~valid_congress & self.df['congno'].notna()).sum()
            if invalid_congress > 0:
                print(f"Warning: {invalid_congress} invalid congress numbers found")
        
        # Clean training set indicators
        if 'training.set' in self.df.columns:
            self.df['training.set'] = pd.to_numeric(self.df['training.set'], errors='coerce')
            self.df['training.set'] = self.df['training.set'].fillna(0).astype(int)
        
        # Validate topic weights
        for col in self.topic_cols:
            # Ensure topic weights are between 0 and 1
            invalid_weights = (self.df[col] < 0) | (self.df[col] > 1)
            if invalid_weights.any():
                print(f"Warning: {invalid_weights.sum()} invalid weights in {col}")
                self.df.loc[invalid_weights, col] = self.df.loc[invalid_weights, col].clip(0, 1)
    
    def _validate_after_cleaning(self):
        """Validate data quality after cleaning."""
        issues = []
        
        # Check for completely empty documents
        if 'text' in self.df.columns:
            empty_text = (self.df['text'].str.len() == 0).sum()
            if empty_text > 0:
                issues.append(f"{empty_text} documents with empty text")
        
        # Check topic weight consistency
        if self.topic_cols:
            # Check if topic weights sum to reasonable values
            topic_sums = self.df[self.topic_cols].sum(axis=1)
            zero_weights = (topic_sums == 0).sum()
            if zero_weights > 0:
                issues.append(f"{zero_weights} documents with all zero topic weights")
        
        if issues:
            print("Data quality issues found:")
            for issue in issues:
                print(f"  - {issue}")
    
    def optimize_formats(self, 
                        output_dir: str = "optimized_data",
                        create_shards: bool = True,
                        shard_size: int = 50000) -> Dict[str, Any]:
        """
        Optimize data formats for storage and processing.
        
        Args:
            output_dir: Output directory for optimized files
            create_shards: Whether to create sharded versions
            shard_size: Number of documents per shard
            
        Returns:
            Dictionary with optimization results
        """
        if self.df is None:
            raise ValueError("No dataset loaded.")
        
        optimizer = FormatOptimizer(self.df)
        results = optimizer.optimize_all_formats(
            output_dir=output_dir,
            create_shards=create_shards,
            shard_size=shard_size
        )
        
        self.processing_history.append({
            'operation': 'optimize_formats',
            'output_dir': output_dir,
            'formats_created': list(results.keys())
        })
        
        return results
    
    def compute_statistics(self, 
                          include_topic_analysis: bool = True,
                          include_text_analysis: bool = True) -> Dict[str, Any]:
        """
        Compute comprehensive dataset statistics.
        
        Args:
            include_topic_analysis: Include detailed topic analysis
            include_text_analysis: Include text length and content analysis
            
        Returns:
            Dictionary with comprehensive statistics
        """
        if self.df is None:
            raise ValueError("No dataset loaded.")
        
        computer = StatisticsComputer(
            self.df, 
            self.topic_cols,
            include_topic_analysis=include_topic_analysis,
            include_text_analysis=include_text_analysis
        )
        
        self.statistics = computer.compute_all_statistics()
        
        self.processing_history.append({
            'operation': 'compute_statistics',
            'statistics_categories': list(self.statistics.keys())
        })
        
        return self.statistics
    
    def create_filtered_versions(self, 
                               output_dir: str = "filtered_data",
                               custom_filters: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
        """
        Create commonly used filtered versions of the dataset.
        
        Args:
            output_dir: Output directory for filtered datasets
            custom_filters: Custom filter definitions
            
        Returns:
            Dictionary with filter results
        """
        if self.df is None:
            raise ValueError("No dataset loaded.")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Define standard filters
        filters = self._get_standard_filters()
        
        # Add custom filters if provided
        if custom_filters:
            filters.update(custom_filters)
        
        filter_results = {}
        
        for filter_name, filter_config in filters.items():
            try:
                # Apply filter
                filtered_df = self._apply_filter(filter_config)
                
                if len(filtered_df) > 0:
                    # Save filtered version
                    output_file = output_path / f"speeches_{filter_name}.parquet"
                    filtered_df.to_parquet(output_file)
                    
                    filter_results[filter_name] = {
                        'documents': len(filtered_df),
                        'file_path': str(output_file),
                        'size_mb': output_file.stat().st_size / (1024**2),
                        'filter_config': filter_config
                    }
                    
                    print(f"Created {filter_name}: {len(filtered_df):,} documents")
                else:
                    print(f"Filter {filter_name} resulted in empty dataset")
                    
            except Exception as e:
                print(f"Failed to create filter {filter_name}: {e}")
        
        self.processing_history.append({
            'operation': 'create_filtered_versions',
            'output_dir': output_dir,
            'filters_created': list(filter_results.keys())
        })
        
        return filter_results
    
    def _get_standard_filters(self) -> Dict[str, Dict]:
        """Get standard filter definitions."""
        filters = {}
        
        # Legislative body filters
        if 'legis.body' in self.df.columns:
            filters['house_only'] = {'column': 'legis.body', 'value': 'US House'}
            filters['senate_only'] = {'column': 'legis.body', 'value': 'US Senate'}
        
        # Congress filters
        if 'congno' in self.df.columns:
            filters['recent_congress'] = {'column': 'congno', 'condition': 'ge', 'value': 110}
            filters['congress_108'] = {'column': 'congno', 'value': 108}
            filters['congress_109'] = {'column': 'congno', 'value': 109}
        
        # Training split filters
        if 'training.set' in self.df.columns:
            filters['training_only'] = {'column': 'training.set', 'value': 1}
            filters['test_only'] = {'column': 'training.set', 'value': 0}
        
        # Topic-based filters
        high_impact_topics = ['tw.healthcare', 'tw.economy', 'tw.defense.and.foreign.policy']
        for topic in high_impact_topics:
            if topic in self.topic_cols:
                topic_name = topic.replace('tw.', '').replace('.', '_')
                filters[f'{topic_name}_focused'] = {
                    'column': topic, 
                    'condition': 'gt', 
                    'value': 0.1
                }
        
        return filters
    
    def _apply_filter(self, filter_config: Dict) -> pd.DataFrame:
        """Apply a single filter configuration."""
        column = filter_config['column']
        
        if column not in self.df.columns:
            raise ValueError(f"Column {column} not found in dataset")
        
        condition = filter_config.get('condition', 'eq')
        value = filter_config['value']
        
        if condition == 'eq':
            mask = self.df[column] == value
        elif condition == 'gt':
            mask = self.df[column] > value
        elif condition == 'ge':
            mask = self.df[column] >= value
        elif condition == 'lt':
            mask = self.df[column] < value
        elif condition == 'le':
            mask = self.df[column] <= value
        else:
            raise ValueError(f"Unknown condition: {condition}")
        
        return self.df[mask].copy()
    
    def export_processing_report(self, output_file: str = "processing_report.json"):
        """Export processing history and statistics to a report file."""
        report = {
            'processing_history': self.processing_history,
            'final_statistics': self.statistics,
            'dataset_info': self.dataset.get_info() if self.dataset else None
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Processing report saved to {output_file}")
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of all processing operations."""
        return {
            'operations_performed': [op['operation'] for op in self.processing_history],
            'final_dataset_size': len(self.df) if self.df is not None else 0,
            'statistics_available': bool(self.statistics),
            'topic_columns_found': len(self.topic_cols)
        }


class StatisticsComputer:
    """Comprehensive statistical analysis for text scaling datasets."""
    
    def __init__(self, 
                 df: pd.DataFrame, 
                 topic_cols: List[str],
                 include_topic_analysis: bool = True,
                 include_text_analysis: bool = True):
        """
        Initialize statistics computer.
        
        Args:
            df: DataFrame to analyze
            topic_cols: List of topic weight column names
            include_topic_analysis: Include topic-related statistics
            include_text_analysis: Include text-related statistics
        """
        self.df = df
        self.topic_cols = topic_cols
        self.include_topic_analysis = include_topic_analysis
        self.include_text_analysis = include_text_analysis
    
    def compute_all_statistics(self) -> Dict[str, Any]:
        """Compute comprehensive statistics."""
        print("Computing comprehensive statistics...")
        
        stats = {}
        
        # Basic dataset statistics
        stats['basic_stats'] = self._compute_basic_stats()
        
        # Distribution statistics
        stats['distributions'] = self._compute_distributions()
        
        # Topic analysis
        if self.include_topic_analysis and self.topic_cols:
            stats['topic_analysis'] = self._compute_topic_statistics()
        
        # Text analysis
        if self.include_text_analysis and 'text' in self.df.columns:
            stats['text_analysis'] = self._compute_text_statistics()
        
        # Data quality metrics
        stats['data_quality'] = self._compute_data_quality()
        
        # Temporal analysis
        if 'date' in self.df.columns:
            stats['temporal_analysis'] = self._compute_temporal_statistics()
        
        print("✅ Statistics computation complete")
        return stats
    
    def _compute_basic_stats(self) -> Dict[str, Any]:
        """Compute basic dataset statistics."""
        return {
            'total_documents': len(self.df),
            'total_columns': len(self.df.columns),
            'unique_documents': self.df['doc.id'].nunique() if 'doc.id' in self.df.columns else len(self.df),
            'unique_legislators': self.df['bonica.rid'].nunique() if 'bonica.rid' in self.df.columns else None,
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / (1024**2),
            'topic_columns': len(self.topic_cols)
        }
    
    def _compute_distributions(self) -> Dict[str, Any]:
        """Compute distribution statistics for key columns."""
        distributions = {}
        
        # Legislative body distribution
        if 'legis.body' in self.df.columns:
            distributions['legislative_body'] = self.df['legis.body'].value_counts().to_dict()
        
        # Congress distribution
        if 'congno' in self.df.columns:
            distributions['congress'] = self.df['congno'].value_counts().sort_index().to_dict()
        
        # Training split distribution
        if 'training.set' in self.df.columns:
            distributions['training_split'] = self.df['training.set'].value_counts().to_dict()
        
        return distributions
    
    def _compute_topic_statistics(self) -> Dict[str, Any]:
        """Compute comprehensive topic-related statistics."""
        topic_stats = {}
        
        # Individual topic statistics
        topic_details = {}
        for topic in self.topic_cols:
            topic_data = self.df[topic]
            topic_details[topic] = {
                'mean': float(topic_data.mean()),
                'std': float(topic_data.std()),
                'min': float(topic_data.min()),
                'max': float(topic_data.max()),
                'median': float(topic_data.median()),
                'q25': float(topic_data.quantile(0.25)),
                'q75': float(topic_data.quantile(0.75)),
                'active_documents': int((topic_data > 0.01).sum()),
                'high_weight_documents': int((topic_data > 0.1).sum()),
                'zero_weight_documents': int((topic_data == 0).sum())
            }
        
        topic_stats['individual_topics'] = topic_details
        
        # Dominant topic analysis
        if 'dominant_topic' in self.df.columns:
            dominant_counts = self.df['dominant_topic'].value_counts()
            topic_stats['dominant_topic_distribution'] = dominant_counts.head(15).to_dict()
            
            # Clean topic names for display
            clean_dominant = {}
            for topic, count in dominant_counts.head(10).items():
                clean_name = topic.replace('tw.', '').replace('.', ' ').title()
                clean_dominant[clean_name] = count
            topic_stats['dominant_topic_distribution_clean'] = clean_dominant
        
        # Topic diversity analysis
        if 'topic_entropy' in self.df.columns:
            entropy_data = self.df['topic_entropy']
            topic_stats['diversity_metrics'] = {
                'mean_entropy': float(entropy_data.mean()),
                'std_entropy': float(entropy_data.std()),
                'low_diversity_docs': int((entropy_data < 1.0).sum()),
                'high_diversity_docs': int((entropy_data > 2.0).sum()),
                'entropy_percentiles': {
                    '10%': float(entropy_data.quantile(0.1)),
                    '25%': float(entropy_data.quantile(0.25)),
                    '50%': float(entropy_data.quantile(0.5)),
                    '75%': float(entropy_data.quantile(0.75)),
                    '90%': float(entropy_data.quantile(0.9))
                }
            }
        
        # Active topic analysis
        if 'active_topic_count' in self.df.columns:
            active_counts = self.df['active_topic_count']
            topic_stats['active_topic_analysis'] = {
                'mean_active_topics': float(active_counts.mean()),
                'std_active_topics': float(active_counts.std()),
                'single_topic_docs': int((active_counts == 1).sum()),
                'multi_topic_docs': int((active_counts > 3).sum()),
                'max_active_topics': int(active_counts.max()),
                'active_topic_distribution': active_counts.value_counts().head(10).to_dict()
            }
        
        # Topic correlation analysis (top correlations)
        if len(self.topic_cols) > 1:
            topic_corr = self.df[self.topic_cols].corr()
            
            # Find strongest positive correlations (excluding diagonal)
            mask = np.triu(np.ones_like(topic_corr), k=1).astype(bool)
            corr_pairs = topic_corr.where(mask).stack().sort_values(ascending=False)
            
            topic_stats['top_correlations'] = {
                f"{pair[0]} vs {pair[1]}": float(corr)
                for pair, corr in corr_pairs.head(10).items()
            }
        
        return topic_stats
    
    def _compute_text_statistics(self) -> Dict[str, Any]:
        """Compute text-related statistics."""
        text_data = self.df['text']
        text_lengths = text_data.str.len()
        word_counts = text_data.str.split().str.len()
        
        text_stats = {
            'character_statistics': {
                'mean_length': float(text_lengths.mean()),
                'median_length': float(text_lengths.median()),
                'std_length': float(text_lengths.std()),
                'min_length': int(text_lengths.min()),
                'max_length': int(text_lengths.max()),
                'length_percentiles': {
                    '10%': int(text_lengths.quantile(0.1)),
                    '25%': int(text_lengths.quantile(0.25)),
                    '75%': int(text_lengths.quantile(0.75)),
                    '90%': int(text_lengths.quantile(0.9))
                }
            },
            'word_statistics': {
                'mean_words': float(word_counts.mean()),
                'median_words': float(word_counts.median()),
                'std_words': float(word_counts.std()),
                'min_words': int(word_counts.min()),
                'max_words': int(word_counts.max())
            }
        }
        
        # Text length categories
        text_stats['length_categories'] = {
            'very_short': int((text_lengths < 100).sum()),
            'short': int(((text_lengths >= 100) & (text_lengths < 500)).sum()),
            'medium': int(((text_lengths >= 500) & (text_lengths < 2000)).sum()),
            'long': int(((text_lengths >= 2000) & (text_lengths < 5000)).sum()),
            'very_long': int((text_lengths >= 5000).sum())
        }
        
        return text_stats
    
    def _compute_data_quality(self) -> Dict[str, Any]:
        """Compute data quality metrics."""
        quality_metrics = {}
        
        # Missing data analysis
        missing_data = {}
        for col in self.df.columns:
            missing_count = self.df[col].isna().sum()
            if missing_count > 0:
                missing_data[col] = {
                    'count': int(missing_count),
                    'percentage': float(missing_count / len(self.df) * 100)
                }
        
        quality_metrics['missing_data'] = missing_data
        
        # Duplicate analysis
        if 'doc.id' in self.df.columns:
            duplicate_ids = self.df['doc.id'].duplicated().sum()
            quality_metrics['duplicate_document_ids'] = int(duplicate_ids)
        
        # Text quality issues
        if 'text' in self.df.columns:
            empty_text = (self.df['text'].str.len() == 0).sum()
            very_short_text = (self.df['text'].str.len() < 10).sum()
            
            quality_metrics['text_quality'] = {
                'empty_text_documents': int(empty_text),
                'very_short_documents': int(very_short_text),
                'percentage_problematic': float((empty_text + very_short_text) / len(self.df) * 100)
            }
        
        # Topic weight quality
        if self.topic_cols:
            # Documents with all zero topic weights
            topic_sums = self.df[self.topic_cols].sum(axis=1)
            zero_topic_docs = (topic_sums == 0).sum()
            
            # Documents with topic weights summing to > 1 (potential normalization issues)
            high_sum_docs = (topic_sums > 1.1).sum()
            
            quality_metrics['topic_weight_quality'] = {
                'zero_weight_documents': int(zero_topic_docs),
                'high_sum_documents': int(high_sum_docs),
                'mean_topic_sum': float(topic_sums.mean()),
                'std_topic_sum': float(topic_sums.std())
            }
        
        return quality_metrics
    
    def _compute_temporal_statistics(self) -> Dict[str, Any]:
        """Compute temporal analysis statistics."""
        if 'date' not in self.df.columns:
            return {}
        
        dates = pd.to_datetime(self.df['date'], errors='coerce')
        valid_dates = dates.dropna()
        
        if len(valid_dates) == 0:
            return {'error': 'No valid dates found'}
        
        temporal_stats = {
            'date_range': {
                'start_date': str(valid_dates.min().date()),
                'end_date': str(valid_dates.max().date()),
                'span_days': int((valid_dates.max() - valid_dates.min()).days),
                'valid_dates': len(valid_dates),
                'invalid_dates': len(self.df) - len(valid_dates)
            }
        }
        
        # Yearly distribution
        if 'year' in self.df.columns:
            year_counts = self.df['year'].value_counts().sort_index()
            temporal_stats['yearly_distribution'] = year_counts.to_dict()
            temporal_stats['years_covered'] = int(year_counts.nunique())
        
        # Congressional distribution over time
        if 'congno' in self.df.columns and 'year' in self.df.columns:
            congress_years = self.df.groupby('congno')['year'].agg(['min', 'max', 'count'])
            temporal_stats['congress_temporal_info'] = {
                int(congress): {
                    'min_year': int(row['min']) if pd.notna(row['min']) else None,
                    'max_year': int(row['max']) if pd.notna(row['max']) else None,
                    'document_count': int(row['count'])
                }
                for congress, row in congress_years.iterrows()
            }
        
        return temporal_stats


class FormatOptimizer:
    """Data format optimization and conversion utilities."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize format optimizer.
        
        Args:
            df: DataFrame to optimize
        """
        self.df = df
    
    def optimize_all_formats(self, 
                           output_dir: str = "optimized_data",
                           create_shards: bool = True,
                           shard_size: int = 50000) -> Dict[str, Any]:
        """
        Create optimized versions in multiple formats.
        
        Args:
            output_dir: Output directory
            create_shards: Whether to create sharded versions
            shard_size: Documents per shard
            
        Returns:
            Dictionary with optimization results
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        results = {}
        
        print(f"Optimizing formats in {output_dir}...")
        
        # Parquet format (best for analytics)
        parquet_result = self._create_parquet(output_path)
        results['parquet'] = parquet_result
        
        # Compressed CSV
        csv_result = self._create_compressed_csv(output_path)
        results['compressed_csv'] = csv_result
        
        # JSON Lines format
        jsonl_result = self._create_jsonlines(output_path)
        results['jsonlines'] = jsonl_result
        
        # Create shards if requested
        if create_shards:
            shard_result = self._create_shards(output_path, shard_size)
            results['shards'] = shard_result
        
        # Summary comparison
        results['summary'] = self._create_format_comparison(results)
        
        return results
    
    def _create_parquet(self, output_path: Path) -> Dict[str, Any]:
        """Create optimized Parquet file."""
        parquet_file = output_path / "congressional_speeches.parquet"
        
        # Optimize data types before saving
        optimized_df = self._optimize_datatypes(self.df.copy())
        
        # Save with optimal compression
        optimized_df.to_parquet(
            parquet_file, 
            compression='snappy',
            index=False
        )
        
        file_size = parquet_file.stat().st_size
        
        return {
            'file_path': str(parquet_file),
            'size_bytes': file_size,
            'size_mb': file_size / (1024**2),
            'compression': 'snappy'
        }
    
    def _create_compressed_csv(self, output_path: Path) -> Dict[str, Any]:
        """Create compressed CSV file."""
        csv_file = output_path / "congressional_speeches.csv.gz"
        
        self.df.to_csv(
            csv_file,
            compression='gzip',
            index=False
        )
        
        file_size = csv_file.stat().st_size
        
        return {
            'file_path': str(csv_file),
            'size_bytes': file_size,
            'size_mb': file_size / (1024**2),
            'compression': 'gzip'
        }
    
    def _create_jsonlines(self, output_path: Path) -> Dict[str, Any]:
        """Create JSON Lines format file."""
        jsonl_file = output_path / "congressional_speeches.jsonl.gz"
        
        # Convert to JSON Lines with compression
        with pd.io.common.get_handle(jsonl_file, 'w', compression='gzip') as handle:
            for _, row in self.df.iterrows():
                json.dump(row.to_dict(), handle.handle, default=str)
                handle.handle.write('\n')
        
        file_size = jsonl_file.stat().st_size
        
        return {
            'file_path': str(jsonl_file),
            'size_bytes': file_size,
            'size_mb': file_size / (1024**2),
            'compression': 'gzip'
        }
    
    def _create_shards(self, output_path: Path, shard_size: int) -> Dict[str, Any]:
        """Create sharded dataset for distributed processing."""
        shard_dir = output_path / "shards"
        shard_dir.mkdir(exist_ok=True)
        
        n_shards = len(self.df) // shard_size + (1 if len(self.df) % shard_size > 0 else 0)
        shard_files = []
        total_size = 0
        
        for i in range(n_shards):
            start_idx = i * shard_size
            end_idx = min((i + 1) * shard_size, len(self.df))
            
            shard_df = self.df.iloc[start_idx:end_idx]
            shard_file = shard_dir / f"shard_{i:04d}.parquet"
            
            shard_df.to_parquet(shard_file, compression='snappy', index=False)
            
            shard_files.append(str(shard_file))
            total_size += shard_file.stat().st_size
        
        return {
            'shard_directory': str(shard_dir),
            'num_shards': n_shards,
            'shard_files': shard_files,
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024**2),
            'avg_shard_size_mb': (total_size / n_shards) / (1024**2)
        }
    
    def _optimize_datatypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame data types for storage."""
        # Convert object columns with few unique values to category
        for col in df.select_dtypes(include=['object']).columns:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.5:  # Less than 50% unique values
                df[col] = df[col].astype('category')
        
        # Optimize integer columns
        for col in df.select_dtypes(include=['int64']).columns:
            col_min, col_max = df[col].min(), df[col].max()
            
            if col_min >= 0:
                if col_max <= 255:
                    df[col] = df[col].astype('uint8')
                elif col_max <= 65535:
                    df[col] = df[col].astype('uint16')
                elif col_max <= 4294967295:
                    df[col] = df[col].astype('uint32')
            else:
                if col_min >= -128 and col_max <= 127:
                    df[col] = df[col].astype('int8')
                elif col_min >= -32768 and col_max <= 32767:
                    df[col] = df[col].astype('int16')
                elif col_min >= -2147483648 and col_max <= 2147483647:
                    df[col] = df[col].astype('int32')
        
        # Optimize float columns
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        return df
    
    def _create_format_comparison(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create comparison summary of different formats."""
        comparison = {}
        
        formats_to_compare = ['parquet', 'compressed_csv', 'jsonlines']
        
        for fmt in formats_to_compare:
            if fmt in results:
                comparison[fmt] = {
                    'size_mb': results[fmt]['size_mb'],
                    'compression': results[fmt].get('compression', 'none')
                }
        
        # Calculate space savings relative to compressed CSV (baseline)
        if 'compressed_csv' in comparison:
            baseline_size = comparison['compressed_csv']['size_mb']
            
            for fmt in comparison:
                if fmt != 'compressed_csv':
                    savings = (1 - comparison[fmt]['size_mb'] / baseline_size) * 100
                    comparison[fmt]['savings_vs_csv_percent'] = round(savings, 1)
        
        return comparison


class DataValidator:
    """Data quality validation and reporting."""
    
    def __init__(self, df: pd.DataFrame, topic_cols: List[str]):
        """
        Initialize data validator.
        
        Args:
            df: DataFrame to validate
            topic_cols: List of topic column names
        """
        self.df = df
        self.topic_cols = topic_cols
        self.validation_results = {}
    
    def validate_all(self) -> Dict[str, Any]:
        """Run comprehensive data validation."""
        print("Running data validation...")
        
        self.validation_results = {
            'basic_validation': self._validate_basic_structure(),
            'content_validation': self._validate_content_quality(),
            'topic_validation': self._validate_topic_weights(),
            'consistency_validation': self._validate_data_consistency()
        }
        
        # Generate overall validation score
        self.validation_results['overall_score'] = self._calculate_validation_score()
        
        print("✅ Data validation complete")
        return self.validation_results
    
    def _validate_basic_structure(self) -> Dict[str, Any]:
        """Validate basic data structure."""
        issues = []
        
        # Check for empty dataset
        if len(self.df) == 0:
            issues.append("Dataset is empty")
        
        # Check for required columns
        required_cols = ['text']
        missing_required = [col for col in required_cols if col not in self.df.columns]
        if missing_required:
            issues.append(f"Missing required columns: {missing_required}")
        
        # Check for duplicate rows
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            issues.append(f"{duplicates} duplicate rows found")
        
        return {
            'issues': issues,
            'passed': len(issues) == 0,
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns)
        }
    
    def _validate_content_quality(self) -> Dict[str, Any]:
        """Validate content quality."""
        issues = []
        
        if 'text' in self.df.columns:
            empty_text = (self.df['text'].str.len() == 0).sum()
            if empty_text > 0:
                issues.append(f"{empty_text} documents with empty text")
            
            very_short = (self.df['text'].str.len() < 10).sum()
            if very_short > len(self.df) * 0.05:  # More than 5% very short
                issues.append(f"{very_short} documents with very short text (< 10 characters)")
        
        return {
            'issues': issues,
            'passed': len(issues) == 0
        }
    
    def _validate_topic_weights(self) -> Dict[str, Any]:
        """Validate topic weight consistency."""
        issues = []
        
        if not self.topic_cols:
            return {'issues': [], 'passed': True, 'message': 'No topic columns to validate'}
        
        for col in self.topic_cols:
            # Check for valid range [0, 1]
            invalid_range = ((self.df[col] < 0) | (self.df[col] > 1)).sum()
            if invalid_range > 0:
                issues.append(f"{col}: {invalid_range} values outside [0,1] range")
            
            # Check for missing values
            missing_vals = self.df[col].isna().sum()
            if missing_vals > 0:
                issues.append(f"{col}: {missing_vals} missing values")
        
        # Check for documents with all zero topic weights
        topic_sums = self.df[self.topic_cols].sum(axis=1)
        zero_docs = (topic_sums == 0).sum()
        if zero_docs > 0:
            issues.append(f"{zero_docs} documents with all zero topic weights")
        
        return {
            'issues': issues,
            'passed': len(issues) == 0,
            'topic_columns_validated': len(self.topic_cols)
        }
    
    def _validate_data_consistency(self) -> Dict[str, Any]:
        """Validate data consistency across related columns."""
        issues = []
        
        # Validate congress numbers
        if 'congno' in self.df.columns:
            invalid_congress = self.df['congno'].dropna()
            invalid_congress = ((invalid_congress < 100) | (invalid_congress > 120)).sum()
            if invalid_congress > 0:
                issues.append(f"{invalid_congress} invalid congress numbers")
        
        # Validate dates
        if 'date' in self.df.columns:
            invalid_dates = pd.to_datetime(self.df['date'], errors='coerce').isna().sum()
            if invalid_dates > 0:
                issues.append(f"{invalid_dates} invalid dates")
        
        # Validate training set indicators
        if 'training.set' in self.df.columns:
            invalid_training = (~self.df['training.set'].isin([0, 1])).sum()
            if invalid_training > 0:
                issues.append(f"{invalid_training} invalid training set indicators")
        
        return {
            'issues': issues,
            'passed': len(issues) == 0
        }
    
    def _calculate_validation_score(self) -> float:
        """Calculate overall validation score (0-100)."""
        total_checks = 0
        passed_checks = 0
        
        for category in self.validation_results.values():
            if isinstance(category, dict) and 'passed' in category:
                total_checks += 1
                if category['passed']:
                    passed_checks += 1
        
        return (passed_checks / total_checks * 100) if total_checks > 0 else 0.0


if __name__ == "__main__":
    # Example usage
    print("🧪 Testing TextScalingProcessor...")
    
    try:
        # Initialize and run processing pipeline
        processor = TextScalingProcessor()
        
        # Load dataset
        processor.load_dataset(
            "mliliu/political-text-scaling",
            split="train",
            sample_size=1000  # Small sample for testing
        )
        
        # Run processing pipeline
        processor.clean_data()
        
        # Compute statistics
        stats = processor.compute_statistics()
        
        # Print summary
        summary = processor.get_processing_summary()
        print(f"✅ Processing complete:")
        print(f"   Operations: {', '.join(summary['operations_performed'])}")
        print(f"   Final size: {summary['final_dataset_size']:,} documents")
        print(f"   Topic columns: {summary['topic_columns_found']}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")