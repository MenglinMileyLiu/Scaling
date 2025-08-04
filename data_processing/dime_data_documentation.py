#!/usr/bin/env python3
"""
DIME Dataset Documentation and Processing Pipeline
"""

import pandas as pd
from datasets import load_dataset, DatasetDict
import numpy as np
from collections import Counter
import json

# Step 1: Data License and Rights
DIME_LICENSE = """
DIME (Database on Ideology, Money in Politics, and Elections) License Information:

Source: Stanford University - Adam Bonica
URL: https://data.stanford.edu/dime

Usage Rights:
- Academic research: YES
- Redistribution: Check with Stanford/Adam Bonica
- Commercial use: Requires permission
- Citation required: Bonica, Adam. 2014. "Mapping the Ideological Marketplace." 
  American Journal of Political Science 58(2): 367-386.

IMPORTANT: Before public distribution on Hugging Face, confirm with:
- Stanford's data terms
- Email Adam Bonica for redistribution permission
"""

# Step 2: Dataset Schema Documentation
def document_dataset_schema():
    """Document the DIME recipients dataset structure"""
    
    print("Loading dataset to analyze schema...")
    dataset = load_dataset("mliliu/dime-recipients")
    df = dataset['train'].to_pandas()
    
    schema = {
        "dataset_name": "DIME Recipients Database",
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "time_range": f"{df['cycle'].min()}-{df['cycle'].max()}",
        "columns": {}
    }
    
    # Document each column
    for col in df.columns:
        col_info = {
            "dtype": str(df[col].dtype),
            "null_count": df[col].isna().sum(),
            "null_percentage": f"{(df[col].isna().sum() / len(df)) * 100:.2f}%",
            "unique_values": df[col].nunique() if df[col].nunique() < 100 else f"{df[col].nunique()} (too many to list)",
            "sample_values": df[col].dropna().unique()[:5].tolist() if df[col].nunique() < 100 else df[col].dropna().head(5).tolist()
        }
        schema["columns"][col] = col_info
    
    return schema, df

# Step 3: Key Column Definitions
COLUMN_DEFINITIONS = {
    "bonica.rid": "Unique recipient identifier",
    "name": "Recipient name (candidate or committee)",
    "party": "Party code (100=Dem, 200=Rep, 328=Ind)",
    "state": "State abbreviation",
    "district": "Congressional district number",
    "cycle": "Election cycle year",
    "recipient.cfscore": "Campaign Finance ideology score (-2 to +2, negative=liberal)",
    "office": "Office sought (federal:house, federal:senate, federal:president)",
    "recipient.type": "Type of recipient (cand, comm)",
    "total.contribs": "Total contributions received",
    "total.indivs": "Total individual contributors"
}

# Party code mapping
PARTY_CODES = {
    '100': 'Democrat',
    '200': 'Republican', 
    '328': 'Independent',
    '400': 'Green',
    '500': 'Libertarian',
    'UNK': 'Unknown',
    '': 'Not specified'
}

if __name__ == "__main__":
    print(DIME_LICENSE)
    print("\n" + "="*50 + "\n")
    
    schema, df = document_dataset_schema()
    
    print(f"Dataset Overview:")
    print(f"- Total Recipients: {schema['total_rows']:,}")
    print(f"- Total Columns: {schema['total_columns']}")
    print(f"- Time Period: {schema['time_range']}")
    
    print(f"\nAll column names:")
    for i, col in enumerate(df.columns):
        print(f"{i+1:2d}. {col}")
    
    print("\nKey Columns:")
    for col, desc in COLUMN_DEFINITIONS.items():
        if col in df.columns:
            print(f"- {col}: {desc}")
        else:
            print(f"- {col}: {desc} (NOT FOUND)")
    
    # Look for office-related columns
    office_cols = [col for col in df.columns if 'office' in col.lower()]
    print(f"\nOffice-related columns: {office_cols}")
    
    # Look for other key columns
    key_patterns = ['party', 'score', 'state', 'name', 'cycle']
    for pattern in key_patterns:
        matching_cols = [col for col in df.columns if pattern.lower() in col.lower()]
        print(f"Columns containing '{pattern}': {matching_cols}")
    
    # Save schema to file
    with open('dime_schema.json', 'w') as f:
        json.dump(schema, f, indent=2, default=str)
    
    print("\nSchema saved to dime_schema.json")