# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a political ideology text scaling research project that uses Large Language Models (LLMs) to analyze congressional speeches and derive ideological positioning scores. The pipeline extracts political stances from legislative speeches through structured summarization and pairwise comparisons, then applies Bradley-Terry modeling to create ideological scales.

## Core Architecture

### Main Pipeline (`ideology_pipeline_complete.py`)
- **LegislatorSpeech**: Data class for congressional speech records with metadata (legislator ID, date, issue area, bill name, speech text)
- **StructuredSummary**: Processed speech analysis including stance, arguments, policies, concerns, target groups, rhetorical style
- **PairwiseComparison**: Comparative analysis between two legislators on specific dimensions
- **LLMClient**: OpenAI GPT-4 interface for speech summarization and pairwise comparisons
- **Bradley-Terry Model**: Uses `choix` library to convert pairwise comparisons into ideological scores

### Data Structure
- Congressional speech data in CSV format with columns: `bonica.rid`, `date`, `text`, topic weights, CF scores
- Test data sample in `test_speech_sample.csv` for pipeline validation
- Demo datasets: `congress_demo.csv`, `cfscore_federal_demo.csv` containing legislator metadata and historical scores

## Development Setup

### Environment Configuration
- Requires `.env` file with `OPENAI_API_KEY` (currently references `/Users/menglinliu/Documents/JoshuaClinton/emotion_pipeline/.env`)
- Uses OpenAI GPT-4 model for text analysis

### Dependencies
- `openai` (v1.0+ compatible)
- `choix` for Bradley-Terry modeling
- `pandas` for data manipulation
- `python-dotenv` for environment variables

## Running the Pipeline

### Basic Execution
```python
python pipeline_test.py
```

This script:
1. Loads sample congressional speech data
2. Creates LegislatorSpeech objects with issue area "Environment" 
3. Runs summarization and pairwise comparison pipeline
4. Outputs Bradley-Terry scores and analysis results

### Pipeline Phases
1. **Speech Summarization**: Convert raw congressional speeches into structured political analysis
2. **Pairwise Comparison**: Compare legislators on specified dimensions (e.g., "pro-environmental stance")
3. **Scoring**: Apply Bradley-Terry model to derive relative ideological positions

## Key Implementation Notes

- The Bradley-Terry scoring is currently commented out in the main pipeline (`scores = [0.0 for _ in summaries]`)
- Pipeline supports different comparison dimensions - modify the `dimension` parameter in `run_pipeline()`
- Error handling for disconnected graphs in Bradley-Terry model (when legislators cannot be meaningfully compared)
- All LLM calls use `temperature=0` for reproducible results

## Data Processing

When working with new congressional speech data:
- Ensure CSV format matches expected schema with `bonica.rid`, `date`, `text` columns
- Speeches should include relevant metadata for analysis context
- Consider issue area categorization for focused ideological scaling

## Literature Context

The `literature/` directory contains academic papers on:
- Semantic scaling methodologies
- Spatial preference estimation from text
- Dynamic lexicon approaches to ideological scaling
- Chain-of-thought reasoning for political text analysis