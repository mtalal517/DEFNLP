# DEFNLP: Hidden Data Citation Extraction

A modular Python implementation of the three-phase DEFNLP methodology for identifying and extracting "hidden-in-plain-sight data citations" from scientific publications.

## Overview

This project implements the DEFNLP pipeline as described in the Coleridge Initiative "Show US the Data" competition. The methodology uses a three-phase approach combining baseline string matching, advanced NLP (SpaCy NER + BERT QA), and acronym extraction to identify dataset mentions in scientific publications.

## Features

- **Phase I**: Data cleaning and baseline string matching (internal + external datasets)
- **Phase II**: SpaCy Named Entity Recognition + BERT Question Answering
- **Phase III**: Acronym and abbreviation extraction with matching
- **Modular Design**: Each phase is independently implemented and configurable
- **Optimized for Low Resources**: Efficient text chunking and memory management
- **Transfer Learning**: Merges predictions from all phases for comprehensive coverage

## Project Structure

```
DEFNLP/
├── config.py              # Configuration and hyperparameters
├── utils.py               # Utility functions
├── phase1_baseline.py     # Phase I: Baseline matching
├── phase2_ner_qa.py       # Phase II: NER and QA modeling
├── phase3_acronyms.py     # Phase III: Acronym extraction
├── pipeline.py            # Main pipeline orchestrator
├── train_model.py         # BERT QA model fine-tuning
├── requirements.txt       # Dependencies
├── train.csv              # Training data
├── sample_submission.csv  # Test data template
├── train/                 # Training JSON files
├── test/                  # Test JSON files
└── output/                # Generated predictions
```

## Installation

1. **Clone or download the project**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download SpaCy model**:
```bash
python -m spacy download en_core_web_sm
```

4. **Download NLTK stopwords** (optional, done automatically):
```python
import nltk
nltk.download('stopwords')
```

## Usage

### Quick Start - Run Inference

```python
from pipeline import DEFNLPPipeline

# Create pipeline
pipeline = DEFNLPPipeline()

# Run inference on test data
predictions = pipeline.run_full_pipeline()

# Predictions saved to output/predictions.csv
```

### Run from Command Line

```bash
python pipeline.py
```

### Fine-tune BERT QA Model (Optional)

The pipeline uses a pre-trained BERT model by default. To fine-tune on your data:

```bash
python train_model.py
```

Then update `config.py` to use your fine-tuned model:
```python
QA_MODEL_NAME = "./models/qa_model"
```

### Custom Configuration

Edit `config.py` to customize:
- File paths
- Model parameters
- Keywords for filtering
- QA questions
- NER entity types
- Output formatting

## Pipeline Details

### Phase I: Data Cleaning & Baseline Modeling

1. Merges JSON publication text into DataFrame
2. Cleans text (lowercase, remove special chars, stopwords)
3. Matches against internal training labels
4. Matches against external dataset list (`big_gov_datasets.txt`)

### Phase II: SpaCy NER & BERT QA

1. Filters sentences containing data-related keywords
2. Chunks text for efficient processing
3. Extracts entities using SpaCy (DATE, ORG types)
4. Runs BERT QA with multiple targeted questions
5. Combines NER and QA predictions

### Phase III: Acronym & Abbreviation Extraction

1. Extracts acronyms using regex patterns
2. Identifies abbreviation-acronym pairs (e.g., "National Institutes of Health (NIH)")
3. Matches acronyms with Phase II predictions
4. Creates dataset name variants

### Final Output

- Merges all phase predictions
- Removes duplicates
- Sorts alphabetically
- Formats as pipe-separated string: `dataset1 | dataset2 | dataset3`

## Data Requirements

### Required Files

1. **train.csv**: Training data with columns:
   - `Id`: Publication ID
   - `pub_title`: Publication title
   - `dataset_title`: Dataset title
   - `dataset_label`: Dataset label
   - `cleaned_label`: Cleaned label

2. **sample_submission.csv**: Test data template with:
   - `Id`: Publication ID
   - `PredictionString`: (to be filled)

3. **train/** and **test/**: Directories with JSON files containing publication text

4. **big_gov_datasets.txt** (optional): External dataset names, one per line

### JSON Format

Each publication JSON should contain sections with `text` fields:
```json
[
  {
    "section_title": "Introduction",
    "text": "Publication text..."
  },
  ...
]
```

## Performance

- **Optimized for CPU**: Fast inference (~0.07s per query)
- **Memory Efficient**: Text chunking prevents memory overflow
- **State-of-the-art**: Achieved 0.554 score in competition

## Configuration Options

Key parameters in `config.py`:

```python
# Model settings
QA_MODEL_NAME = "salti/bert-base-multilingual-cased-finetuned-squad"
QA_MAX_SEQ_LENGTH = 512
QA_MAX_ANSWER_LENGTH = 64

# Keywords for sentence filtering
DATA_KEYWORDS = ["data", "dataset", "database", "sample", ...]

# QA questions
QA_QUESTIONS = [
    "Which datasets are used?",
    "Which data sources are used?",
    ...
]

# NER entity types
NER_ENTITY_TYPES = ["DATE", "ORG"]
```

## Example Output

```
Id,PredictionString
pub_001,adni | alzheimer s disease neuroimaging initiative | pubmed
pub_002,census data | american community survey | acs
```

## Troubleshooting

**SpaCy model not found**:
```bash
python -m spacy download en_core_web_sm
```

**CUDA out of memory**:
- Set `USE_GPU = False` in `config.py`
- Reduce `QA_BATCH_SIZE`

**Missing big_gov_datasets.txt**:
- Create an empty file or add external dataset names (one per line)

## Citation

Based on the DEFNLP methodology from the Coleridge Initiative "Show US the Data" competition.

## License

MIT License
