# Political Text Scaling Dataset - Documentation

## Overview

This repository provides a comprehensive toolkit for political text scaling research using congressional speeches. The dataset contains 307,793+ congressional speeches with pre-computed topic weights, designed for ideology scaling and political analysis.

## Dataset Description

### Key Features
- **307,793+ congressional speeches** from the U.S. Congress
- **25 pre-computed topic weights** for each document
- **Multiple congress sessions** covered (108th Congress and beyond)
- **Train/validation/test splits** for machine learning workflows
- **Comprehensive metadata** including dates, legislators, and legislative context

### Topic Categories
The dataset includes topic weights for 25 political dimensions:
- Healthcare Policy
- Defense and Foreign Policy
- Economic Policy
- Civil Rights
- Education Policy
- Environmental Issues
- Immigration Policy
- And 18 additional political topics

## Quick Start

### Loading the Dataset

```python
from scaling.dataset import TextScalingDataset

# Load the full dataset
dataset = TextScalingDataset(
    source="mliliu/political-text-scaling",
    split="train"
)

print(f"Loaded {len(dataset):,} congressional speeches")
print(f"Topic columns: {len(dataset.topic_cols)}")
```

### Filtering and Analysis

```python
# Filter to recent congresses only
recent_dataset = TextScalingDataset(
    source="mliliu/political-text-scaling",
    filters={'congno': lambda x: x >= 110},
    sample_size=1000
)

# Analyze topic distributions
topic_summary = recent_dataset.get_topic_summary()
print("Top dominant topics:")
for topic, count in topic_summary['dominant_topic_distribution'].items():
    clean_name = topic.replace('tw.', '').replace('.', ' ').title()
    print(f"  {clean_name}: {count} documents")
```

### Processing Pipeline

```python
from scaling.processors import TextScalingProcessor

# Initialize processor
processor = TextScalingProcessor()

# Load and process data
processor.load_dataset("mliliu/political-text-scaling")
processor.clean_data()

# Compute comprehensive statistics
stats = processor.compute_statistics()

# Create optimized formats
processor.optimize_formats(output_dir="optimized_data")
```

## Dataset Schema

### Document Identifiers
- `doc.id`: Unique document identifier
- `bonica.rid`: Legislator (Bonica) identifier  
- `bill.id`: Associated bill identifier
- `page.id`: Congressional record page reference

### Metadata
- `date`: Date of speech (YYYY-MM-DD)
- `congno`: Congress number (e.g., 108 = 108th Congress)
- `legis.body`: Legislative body (US House/US Senate)
- `training.set`: Binary indicator (1 = training, 0 = test)

### Text Content
- `text`: Raw speech text
- `stemmed.text`: Preprocessed stemmed version

### Topic Weights (25 dimensions)
All topic weights are normalized to [0,1] range:
- `tw.healthcare`: Healthcare policy weight
- `tw.defense.and.foreign.policy`: Defense/foreign policy weight
- `tw.economy`: Economic policy weight
- `tw.civil.rights`: Civil rights weight
- ... (21 additional topic dimensions)

## Usage Examples

### 1. Topic-Focused Analysis

```python
# Analyze healthcare-focused speeches
healthcare_dataset = dataset.filter_by_topic('tw.healthcare', min_weight=0.1)
print(f"Healthcare-focused speeches: {len(healthcare_dataset):,}")

# Get topic statistics
healthcare_stats = healthcare_dataset.get_topic_summary()
```

### 2. Temporal Analysis

```python
# Filter by congress and time period
recent_congress = TextScalingDataset(
    source="mliliu/political-text-scaling",
    filters={
        'congno': lambda x: x >= 110,
        'legis.body': lambda x: x == 'US House'
    }
)

# Analyze by year if date parsing is enabled
processor = TextScalingProcessor(recent_congress)
processor.clean_data()  # This will parse dates and add 'year' column
```

### 3. Batch Processing

```python
from scaling.dataset import BatchDataLoader

# Create batch loader
loader = BatchDataLoader(
    dataset=dataset,
    batch_size=64,
    shuffle=True
)

# Process in batches
for batch in loader.iterate_batches():
    # batch is a dictionary with lists of values
    texts = batch['text']
    topic_weights = {k: v for k, v in batch.items() if k.startswith('tw.')}
    
    # Your processing code here
    process_batch(texts, topic_weights)
```

### 4. Large Dataset Streaming

```python
from scaling.dataset import StreamingTextDataset

# For very large datasets, use streaming
streaming_dataset = StreamingTextDataset(
    shard_directory="path/to/shards",
    batch_size=128,
    shuffle=True
)

for batch_df in streaming_dataset:
    # Process batch DataFrame
    analyze_batch(batch_df)
```

## Data Splits

The dataset includes pre-defined splits for machine learning workflows:

- **Training Set**: Documents with `training.set = 1` (majority of data)
- **Validation Set**: 10% random sample from training set
- **Test Set**: Documents with `training.set = 0`

```python
from scaling.dataset import load_dataset_splits

train_ds, val_ds, test_ds = load_dataset_splits("mliliu/political-text-scaling")
print(f"Train: {len(train_ds):,}, Val: {len(val_ds):,}, Test: {len(test_ds):,}")
```

## Performance Optimization

### Memory-Efficient Loading

```python
# Load specific columns only
dataset = TextScalingDataset(
    source="mliliu/political-text-scaling",
    columns=['doc.id', 'text', 'date', 'tw.healthcare', 'tw.economy'],
    sample_size=10000  # Work with smaller sample
)
```

### Format Optimization

```python
# Create optimized formats for faster loading
processor = TextScalingProcessor(dataset)
optimization_results = processor.optimize_formats()

print(f"Parquet size: {optimization_results['parquet']['size_mb']:.1f} MB")
print(f"Compression savings: {optimization_results['summary']['parquet']['savings_vs_csv_percent']:.1f}%")
```

## Analysis Workflows

### Statistical Analysis

```python
# Comprehensive dataset statistics
processor = TextScalingProcessor(dataset)
stats = processor.compute_statistics()

# Access different analysis categories
basic_stats = stats['basic_stats']  # Document counts, memory usage
topic_analysis = stats['topic_analysis']  # Topic weight distributions
text_analysis = stats['text_analysis']  # Text length statistics
```

### Topic Correlation Analysis

```python
# Find correlated topics
topic_stats = dataset.get_topic_summary()
if 'top_correlations' in topic_stats:
    print("Highly correlated topics:")
    for pair, correlation in topic_stats['top_correlations'].items():
        print(f"  {pair}: {correlation:.3f}")
```

### Quality Assessment

```python
from scaling.processors import DataValidator

# Validate data quality
validator = DataValidator(dataset.df, dataset.topic_cols)
validation_results = validator.validate_all()

print(f"Overall validation score: {validation_results['overall_score']:.1f}%")
for category, result in validation_results.items():
    if isinstance(result, dict) and 'issues' in result:
        if result['issues']:
            print(f"{category} issues: {result['issues']}")
```

## Advanced Features

### Custom Filtering

```python
# Complex filtering conditions
custom_filters = {
    'congno': lambda x: x in [110, 111, 112],  # Specific congresses
    'text': lambda x: len(x) > 500,  # Longer speeches only
    'tw.healthcare': lambda x: x > 0.05  # Some healthcare content
}

filtered_dataset = TextScalingDataset(
    source="mliliu/political-text-scaling",
    filters=custom_filters
)
```

### Topic-Focused Datasets

```python
from scaling.dataset import create_topic_focused_datasets

# Create datasets for major topics
topic_datasets = create_topic_focused_datasets(
    source="mliliu/political-text-scaling",
    topics=['tw.healthcare', 'tw.economy', 'tw.defense.and.foreign.policy'],
    min_weight=0.1
)

for topic_name, topic_dataset in topic_datasets.items():
    print(f"{topic_name}: {len(topic_dataset):,} documents")
```

## Data Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{political_text_scaling_2024,
  title={Political Text Scaling Dataset: Congressional Speeches with Topic Weights},
  author={Liu, Menglin},
  year={2024},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/mliliu/political-text-scaling}
}
```

## License and Usage

This dataset is derived from public congressional records. Please:
- Use responsibly for research purposes
- Be aware of potential political biases in the data
- Consider temporal context when analyzing speeches
- Respect privacy and ethical guidelines for political data analysis

## Support and Issues

For questions, issues, or contributions:
1. Check this documentation first
2. Review the API reference in `docs/API.md`
3. See usage examples in `examples/`
4. Open an issue in the GitHub repository

## Technical Requirements

- Python 3.8+
- pandas >= 1.3.0
- datasets >= 2.0.0 (for Hugging Face integration)
- pyarrow >= 5.0.0 (for Parquet support)
- Optional: torch >= 1.9.0 (for PyTorch integration)