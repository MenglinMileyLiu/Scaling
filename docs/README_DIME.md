---
license: [check DIME's license]
task_categories:
- tabular-classification
language:
- en
tags:
- political-science
- campaign-finance
- ideology-scores
---

# DIME Recipients Database with Ideology Scores

## Dataset Description

This dataset contains recipient information and ideology scores from the Database on Ideology, Money in Politics, and Elections (DIME).

### Dataset Source
- **Original Repository:** https://data.stanford.edu/dime
- **Paper:** Bonica, Adam. 2014. "Mapping the Ideological Marketplace." American Journal of Political Science 58(2): 367-386.

## Dataset Structure

### Data Fields

- `recipient_id`: Unique identifier for each recipient
- `recipient_name`: Name of the recipient (candidate/committee)
- `recipient_type`: Type of recipient (e.g., candidate, PAC, party committee)
- `cf_score`: Campaign finance-based ideology score (negative = liberal, positive = conservative)
- `party`: Political party affiliation
- `state`: State of the recipient
- `cycle`: Election cycle year
- `office`: Office sought (e.g., President, Senate, House)
- `district`: Congressional district (if applicable)
- `incumbent`: Incumbency status
- [Add other fields based on actual data]

## Usage

```python
from datasets import load_dataset

# Load the dataset
recipients = load_dataset("your-username/dime-recipients")

# Filter by party
democrats = recipients.filter(lambda x: x['party'] == 'D')

# Get ideology scores
cf_scores = recipients['train']['cf_score']
```

## Citation

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

## License

Please refer to Stanford's DIME dataset license terms for usage restrictions.