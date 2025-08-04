---
license: other
task_categories:
- text-classification
- text2text-generation
language:
- en
tags:
- political-science
- congressional-speeches
- topic-modeling
- ideology-scaling
size_categories:
- 100K<n<1M
---

# Political Text Scaling Dataset

## Dataset Description

This dataset contains 307,793 congressional speeches from the U.S. Congress with pre-computed topic weights for political text scaling research.

### Dataset Statistics
- **Total Documents**: 307,793
- **Time Period**: Multiple congressional sessions
- **File Size**: 3.7 GB
- **Languages**: English
- **Task**: Political ideology scaling, topic modeling, text classification

## Dataset Structure

### Columns (38 total)

#### Document Identifiers
- `doc.id`: Unique document identifier
- `bonica.rid`: Legislator (Bonica) identifier
- `bill.id`: Associated bill identifier
- `sponsor.rid`: Bill sponsor identifier
- `page.id`: Congressional record page reference
- `congno`: Congress number (e.g., 108 = 108th Congress)
- `legis.body`: Legislative body (US Senate, US House)
- `date`: Date of speech (YYYY-MM-DD)

#### Text Content
- `text`: Raw speech text
- `stemmed.text`: Preprocessed stemmed version of the text

#### Training Indicators
- `training.set`: Binary indicator (1 = training, 0 = test)
- `doc.type`: Document type (e.g., "speech")
- `doc.labels`: Topic labels for the document

#### Topic Weights (25 dimensions)
Pre-computed topic weights (0-1 scale) for each document:
- `tw.latent1`: Latent dimension 1
- `tw.abortion.and.social.conservatism`: Abortion and social issues
- `tw.agriculture`: Agricultural policy
- `tw.banking.and.finance`: Banking and financial regulation
- `tw.civil.rights`: Civil rights issues
- `tw.congress.and.procedural`: Congressional procedures
- `tw.crime`: Crime and law enforcement
- `tw.defense.and.foreign.policy`: Defense and foreign policy
- `tw.economy`: Economic policy
- `tw.education`: Education policy
- `tw.energy`: Energy policy
- `tw.environment`: Environmental issues
- `tw.fair.elections`: Election and voting rights
- `tw.federal.agencies.and.gov.regulation`: Federal agencies and regulation
- `tw.guns`: Gun policy
- `tw.healthcare`: Healthcare policy
- `tw.higher.education`: Higher education
- `tw.immigration`: Immigration policy
- `tw.indian.affairs`: Native American affairs
- `tw.intelligence.and.surveillance`: Intelligence and surveillance
- `tw.labor`: Labor policy
- `tw.law.courts.and.judges`: Legal system and judiciary
- `tw.transportation`: Transportation policy
- `tw.veterans.affairs`: Veterans affairs
- `tw.womens.issues`: Women's issues

## Usage

### Loading the Dataset

```python
from datasets import load_dataset

# Load the full dataset
dataset = load_dataset("mliliu/political-text-scaling")

# Access different splits
train_data = dataset['train']
val_data = dataset['validation']
test_data = dataset['test']

# Convert to pandas for analysis
import pandas as pd
df_train = train_data.to_pandas()
```

### Example: Analyzing Topic Distributions

```python
# Get topic columns
topic_cols = [col for col in df_train.columns if col.startswith('tw.')]

# Find dominant topic for each document
df_train['dominant_topic'] = df_train[topic_cols].idxmax(axis=1)
df_train['max_topic_weight'] = df_train[topic_cols].max(axis=1)

# Topic distribution
topic_counts = df_train['dominant_topic'].value_counts()
print(topic_counts.head(10))
```

### Example: Text and Topic Weight Analysis

```python
# Sample a document
sample = df_train.iloc[0]

# Show text
print(f"Text: {sample['text'][:500]}...")

# Show top topics
topic_weights = sample[topic_cols].sort_values(ascending=False)
print("\nTop 5 topics:")
for topic, weight in topic_weights.head().items():
    print(f"{topic.replace('tw.', '')}: {weight:.3f}")
```

### Example: Filtering by Congress and Date

```python
# Filter to specific congress
congress_110 = df_train[df_train['congno'] == 110]

# Filter by date range
df_train['date'] = pd.to_datetime(df_train['date'])
recent_speeches = df_train[df_train['date'] >= '2010-01-01']

# Filter by legislator
legislator_speeches = df_train[df_train['bonica.rid'] == 'cand1360']
```

## Dataset Splits

- **Train**: Documents with `training.set = 1` (90% of training data)
- **Validation**: Random 10% sample from training set
- **Test**: Documents with `training.set = 0`

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{political_text_scaling_2024,
  title={Political Text Scaling Dataset: Congressional Speeches with Topic Weights},
  author={[Your Name]},
  year={2024},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/mliliu/political-text-scaling}
}
```

## License

This dataset is derived from public congressional records. Please check specific usage terms.

## Ethical Considerations

This dataset contains political speeches from public congressional records. Users should:
- Be aware of potential political biases in the data
- Consider the temporal context of speeches
- Use the data responsibly for research purposes

## Known Limitations

- Topic weights are pre-computed and may not capture all nuances
- Dataset is limited to English language speeches
- Temporal coverage may not be uniform across all years