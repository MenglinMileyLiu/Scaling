# Political Text Scaling Package

A comprehensive toolkit for political text scaling research using congressional speeches with pre-computed topic weights. This package provides professional-grade tools for loading, processing, and analyzing large-scale political text datasets.

## 🌟 Features

- **307,793+ Congressional Speeches** with pre-computed topic weights
- **25 Political Topic Dimensions** (healthcare, economy, defense, etc.)
- **Professional Data Pipeline** following clean architectural patterns
- **Multiple Analysis Workflows** for political science research
- **Optimized Performance** for large datasets (3.7GB+)
- **Comprehensive Documentation** and examples

## 📁 Repository Structure

```
📁 Scaling/
├── 📄 README.md                    # This file
├── 📄 pyproject.toml               # Project configuration
├── 📄 requirements.txt             # Dependencies
│
├── 📁 src/scaling/                 # Core package
│   ├── 📄 dataset.py               # Dataset classes and loading utilities
│   ├── 📄 processors.py            # Data processing and analysis pipelines
│   └── 📁 utils/                   # Utility functions and configuration
│
├── 📁 scripts/                     # Essential scripts
│   ├── 📄 upload_dataset.py        # Upload datasets to Hugging Face
│   └── 📄 test_access.py           # Comprehensive testing suite
│
├── 📁 config/                      # Configuration files
│   └── 📄 dataset_config.yaml      # Dataset schema and settings
│
├── 📁 docs/                        # Documentation
│   ├── 📄 README.md                # Comprehensive usage guide
│   └── 📄 API.md                   # API reference
│
├── 📁 notebooks/                   # Jupyter notebooks
├── 📁 tests/                       # Unit and integration tests
└── 📁 examples/                    # Usage examples
```

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/MenglinMileyLiu/Scaling
cd Scaling
```

2. **Set up development environment:**
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

3. **Install additional dependencies (optional):**
```bash
pip install torch datasets huggingface_hub  # For ML workflows
```

### Basic Usage

```python
from scaling.dataset import TextScalingDataset
from scaling.processors import TextScalingProcessor

# Load congressional speech dataset
dataset = TextScalingDataset(
    source="mliliu/political-text-scaling",
    split="train",
    sample_size=1000  # Use smaller sample for development
)

print(f"Loaded {len(dataset):,} congressional speeches")
print(f"Found {len(dataset.topic_cols)} topic dimensions")

# Analyze topic distributions
topic_summary = dataset.get_topic_summary()
print("Top political topics:")
for topic, count in list(topic_summary['dominant_topic_distribution'].items())[:5]:
    clean_name = topic.replace('tw.', '').replace('.', ' ').title()
    print(f"  {clean_name}: {count} documents")
```

### Processing Pipeline

```python
# Initialize processor with data cleaning and optimization
processor = TextScalingProcessor()
processor.load_dataset("mliliu/political-text-scaling", sample_size=5000)
processor.clean_data()

# Compute comprehensive statistics
stats = processor.compute_statistics()
print(f"Analyzed {stats['basic_stats']['total_speeches']:,} speeches")
print(f"Text analysis: {stats['text_analysis']['character_statistics']['mean_length']:.0f} avg chars")

# Create optimized formats
optimization_results = processor.optimize_formats(output_dir="optimized_data")
print(f"Parquet format: {optimization_results['parquet']['size_mb']:.1f} MB")
```

## 📊 Dataset Overview

### Political Text Scaling Dataset
- **Size**: 307,793+ congressional speeches
- **Source**: U.S. Congressional Record
- **Topic Weights**: 25 pre-computed political dimensions
- **Time Coverage**: Multiple congressional sessions
- **Format**: Available on Hugging Face Hub

### Topic Dimensions Include:
- Healthcare Policy (`tw.healthcare`)
- Defense & Foreign Policy (`tw.defense.and.foreign.policy`)
- Economic Policy (`tw.economy`)
- Civil Rights (`tw.civil.rights`)
- Education Policy (`tw.education`)
- Environmental Issues (`tw.environment`)
- Immigration Policy (`tw.immigration`)
- And 18 additional political topics

## 🛠️ Advanced Features

### Filtering and Analysis
```python
# Filter to recent congresses with healthcare focus
healthcare_dataset = TextScalingDataset(
    source="mliliu/political-text-scaling",
    filters={
        'congno': lambda x: x >= 110,  # Recent congresses
        'tw.healthcare': lambda x: x > 0.1  # Healthcare-focused speeches
    }
)

# Analyze by legislative body
house_vs_senate = dataset.split_by_column('legis.body')
print(f"House speeches: {len(house_vs_senate['US House']):,}")
print(f"Senate speeches: {len(house_vs_senate['US Senate']):,}")
```

### Batch Processing
```python
from scaling.dataset import BatchDataLoader

# Process data in batches for ML workflows
loader = BatchDataLoader(dataset, batch_size=64, shuffle=True)

for batch in loader.iterate_batches():
    texts = batch['text']
    topic_weights = {k: v for k, v in batch.items() if k.startswith('tw.')}
    # Your analysis code here
```

### Memory-Efficient Streaming
```python
from scaling.dataset import StreamingTextDataset

# For very large datasets
streaming_dataset = StreamingTextDataset(
    shard_directory="path/to/shards",
    batch_size=128
)

for batch_df in streaming_dataset:
    analyze_large_batch(batch_df)  # Process without loading full dataset
```

## 📈 Research Applications

This toolkit is designed for:
- **Political Ideology Scaling**: Position legislators on policy dimensions
- **Topic Modeling Analysis**: Understand political discourse patterns  
- **Temporal Political Analysis**: Track policy focus over time
- **Legislative Behavior Studies**: Analyze speech patterns by party/region
- **Comparative Political Research**: Cross-congress and cross-topic analysis

## 🧪 Testing

```bash
# Run comprehensive dataset tests
python scripts/test_access.py

# Test specific functionality
python -m pytest tests/

# Validate data quality
python -c "from scaling.processors import DataValidator; print('Tests passed!')"
```

## 📚 Documentation

- **[Complete Usage Guide](docs/README.md)** - Comprehensive documentation with examples
- **[API Reference](docs/API.md)** - Detailed API documentation
- **[Examples](examples/)** - Jupyter notebooks and usage examples
- **[Configuration](config/)** - Dataset schema and configuration files

## 🔧 Scripts

- **`scripts/upload_dataset.py`** - Professional dataset upload to Hugging Face
- **`scripts/test_access.py`** - Comprehensive testing and validation suite

## 📖 Data Citation

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

## 📄 License

This dataset is derived from public congressional records. Please use responsibly for research purposes and be aware of potential political biases in the data.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes following the existing code style
4. Add tests for new functionality
5. Submit a pull request

## 📞 Support

For questions or issues:
- Check the [documentation](docs/README.md)
- Review existing [issues](https://github.com/MenglinMileyLiu/Scaling/issues)
- Open a new issue with detailed information

## ⭐ Acknowledgments

This project follows professional software engineering practices and clean architectural patterns. Special thanks to the open-source community for the foundational tools and libraries that make this research possible.
