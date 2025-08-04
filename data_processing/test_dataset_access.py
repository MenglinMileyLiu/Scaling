
# Test script to verify dataset upload
from datasets import load_dataset
import pandas as pd

# Load the dataset (no auth token needed in newer versions)
print("Loading dataset...")
dataset = load_dataset("mliliu/dime-recipients")

# Check the data
print(f"\nDataset info: {dataset}")
print(f"Number of recipients: {len(dataset['train'])}")

# Convert to pandas to see columns better
df = dataset['train'].to_pandas()
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst 5 rows:")
print(df.head())

# Check party values
print(f"\nParty distribution:")
print(df['party'].value_counts().head(10))
