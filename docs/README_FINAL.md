---
license: other
task_categories:
- tabular-classification
- tabular-regression
language:
- en
tags:
- political-science
- campaign-finance
- ideology-scores
- elections
- united-states
size_categories:
- 100K<n<1M
---

# DIME Recipients Database with Campaign Finance Ideology Scores

## Dataset Description

This dataset contains comprehensive information about political recipients (candidates and committees) in the United States from 1980-2024, including their campaign finance-based ideology scores from the Database on Ideology, Money in Politics, and Elections (DIME).

### Key Features

- **479,502 recipients** across 1980-2024
- **Campaign Finance (CF) ideology scores** for ideological positioning
- **Multiple office levels**: Federal (House, Senate), State, Local
- **Party affiliations** with cleaned coding
- **Financial data**: Receipts, contributions, expenditures

## Dataset Source

- **Original Source**: Stanford University - Adam Bonica
- **Website**: https://data.stanford.edu/dime
- **Primary Citation**: Bonica, Adam. 2014. "Mapping the Ideological Marketplace." American Journal of Political Science 58(2): 367-386.

## Dataset Structure

### Basic Statistics

- **Total Records**: 479,502
- **Unique Recipients**: 216,371
- **Time Coverage**: 1980-2024
- **Total Receipts**: $1,114,933,155,977
- **Individual Contributions**: $573,053,829,718

### Key Columns

#### Identifiers
- `bonica.rid`: Unique recipient identifier (primary key)
- `name`: Recipient name (candidate or committee)
- `bonica.cid`: Contributor identifier (for matching with contributions)

#### Political Information  
- `party`: Party code (100=Democrat, 200=Republican, 328=Independent)
- `recipient.cfscore`: **Campaign Finance ideology score** (-2 to +2, negative=liberal, positive=conservative)
- `nimsp.office`: Office sought (house, senate, state:lower, local:council, etc.)
- `state`: State abbreviation
- `district`: Congressional district (for House candidates)

#### Financial Data
- `total.receipts`: Total money raised
- `total.indiv.contribs`: Individual contribution amounts  
- `total.pac.contribs`: PAC contribution amounts
- `num.givers`: Number of contributors

#### Additional Scores
- `recipient.cfscore.dyn`: Dynamic CF score (time-varying)
- `dwnom1`: DW-NOMINATE score (for legislators)
- `composite.score`: Composite ideology measure

### Data Distribution

#### Party Distribution
- I: 173,888 (36.3%)
- D: 154,373 (32.2%)
- R: 149,643 (31.2%)
- O: 1,036 (0.2%)
- L: 226 (0.0%)

#### Office Distribution  
- : 169,451
- house: 115,899
- senate: 42,089
- local:other: 16,436
- local:council: 15,877

#### CF Score Distribution
- **Mean**: 0.171
- **Std Dev**: 1.020  
- **Range**: -6.864 to 6.714
- **Median**: 0.074

## Usage Examples

### Basic Loading

```python
from datasets import load_dataset
import pandas as pd

# Load full dataset
dataset = load_dataset("mliliu/dime-recipients")
df = dataset['train'].to_pandas()

print(f"Dataset shape: {df.shape}")
print(f"CF scores available: {df['recipient.cfscore'].notna().sum():,}")
```

### Filtering Examples

```python
# Recent federal candidates only
federal_recent = df[
    (df['cycle'] >= 2016) & 
    (df['nimsp.office'].isin(['house', 'senate'])) &
    (df['recipient.type'] == 'cand')
]

# Major party candidates with CF scores  
major_parties = df[
    df['party'].isin(['100', '200']) &  # Dem/Rep
    df['recipient.cfscore'].notna()
]

# Senate candidates by ideology
senate_liberal = df[
    (df['nimsp.office'] == 'senate') &
    (df['recipient.cfscore'] < -0.5)
]
```

### Ideology Analysis

```python
import matplotlib.pyplot as plt

# Plot ideology distribution by party
dem_scores = df[df['party'] == '100']['recipient.cfscore'].dropna()
rep_scores = df[df['party'] == '200']['recipient.cfscore'].dropna()

plt.hist(dem_scores, alpha=0.7, label='Democrats', bins=50)
plt.hist(rep_scores, alpha=0.7, label='Republicans', bins=50)
plt.xlabel('CF Score (Liberal ← → Conservative)')
plt.ylabel('Frequency')
plt.legend()
plt.show()
```

## Pre-processed Versions Available

This dataset has been optimized and filtered into several versions:

- **`dime_recent.parquet`**: Records from 2010+ (277,297 rows)
- **`dime_federal_candidates.parquet`**: House + Senate candidates (157,988 rows)  
- **`dime_house_candidates.parquet`**: House candidates only (115,899 rows)
- **`dime_senate_candidates.parquet`**: Senate candidates only (42,089 rows)
- **`dime_major_parties.parquet`**: Democrat + Republican only (304,016 rows)

## Data Quality Notes

### Missing Data Rates
- cf_score: 0.0% missing
- party: 0.0% missing
- state: 0.0% missing
- district: 0.0% missing

### Data Cleaning Applied
- Party codes standardized (100→D, 200→R, 328→I, etc.)
- CF scores converted to numeric format
- Office types extracted from NIMSP data
- Decade groupings added for temporal analysis

## Methodology: Campaign Finance Scores

The CF scores are estimated using a Bradley-Terry model applied to campaign contribution patterns:

1. **Contributors** make donations reflecting ideological preferences
2. **Recipients** receive donations from ideologically-aligned contributors  
3. **Scaling algorithm** positions recipients on liberal-conservative dimension
4. **Scores** range from -2 (very liberal) to +2 (very conservative)

**Key advantages**:
- Covers candidates, PACs, and committees
- Available for all time periods  
- Not dependent on roll-call votes
- Captures fundraising-based ideology

## Licensing and Citation

### Usage Rights
- ✅ **Academic Research**: Permitted
- ❓ **Redistribution**: Contact original authors
- ❓ **Commercial Use**: Requires permission  

### Required Citation

```bibtex
@article{bonica2014mapping,
  title={Mapping the ideological marketplace},
  author={Bonica, Adam},
  journal={American Journal of Political Science},
  volume={58},
  number={2},
  pages={367--386},
  year={2014}
}
```

### Additional References

For methodology details:
- Bonica, Adam. 2016. "Avenues of influence: on the political expenditures of corporations and their directors and executives." *Business and Politics* 18(4): 367-394.

## Technical Details

### File Formats
- **Parquet**: 37.2 MB (recommended for analysis)
- **CSV.gz**: 28.8 MB (human-readable)
- **Sharded**: Available for distributed processing

### Performance Benchmarks
- **Loading time**: ~3-5 seconds for full dataset
- **Memory usage**: ~500MB RAM for full dataset in pandas
- **Query performance**: Optimized with column indices

## Contact

For questions about this dataset preparation:
- Dataset processing: Created for academic research
- Original data: Contact Adam Bonica (Stanford)
- Usage questions: See DIME project documentation

---

*Data card generated on 2025-08-03*
*Processing pipeline version: 1.0*
