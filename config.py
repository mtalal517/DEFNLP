"""
Configuration module for DEFNLP pipeline.
Contains all hyperparameters, file paths, and model settings.
"""

import os

# ============================================================================
# FILE PATHS
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
TEST_CSV = os.path.join(BASE_DIR, "test.csv")
TRAIN_JSON_DIR = os.path.join(BASE_DIR, "train")
TEST_JSON_DIR = os.path.join(BASE_DIR, "test")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
BIG_GOV_DATASETS = os.path.join(BASE_DIR, "big_gov_datasets.txt")

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
# BERT QA Model
QA_MODEL_NAME = "salti/bert-base-multilingual-cased-finetuned-squad"
QA_MAX_SEQ_LENGTH = 512
QA_MAX_ANSWER_LENGTH = 64
QA_BATCH_SIZE = 16
QA_LEARNING_RATE = 5e-5
QA_NUM_EPOCHS = 3
QA_DOC_STRIDE = 128

# SpaCy Model
SPACY_MODEL = "en_core_web_sm"

# ============================================================================
# PHASE I: DATA CLEANING & BASELINE
# ============================================================================
# Stopwords configuration
USE_STOPWORDS = True
CUSTOM_STOPWORDS = set()  # Add custom stopwords if needed

# Text cleaning patterns
REMOVE_SPECIAL_CHARS = True
REMOVE_EMOJIS = True
REMOVE_MULTIPLE_SPACES = True

# ============================================================================
# PHASE II: NER & QA MODELING
# ============================================================================
# Keywords for sentence filtering (bag of words)
DATA_KEYWORDS = [
    "data", "datasource", "datasources", "dataset", "datasets",
    "database", "databases", "sample", "samples", "corpus",
    "repository", "repositories", "collection", "survey"
]

# NER entity types to extract
NER_ENTITY_TYPES = ["DATE", "ORG"]

# QA Questions to ask
QA_QUESTIONS = [
    "Which datasets are used?",
    "Which data sources are used?",
    "What datasets were analyzed?",
    "Which databases are mentioned?",
    "What data was collected?"
]

# Chunking parameters
CHUNK_SIZE = 3  # Number of sentences per chunk
CHUNK_OVERLAP = 1  # Overlap between chunks

# ============================================================================
# PHASE III: ACRONYM EXTRACTION
# ============================================================================
# Minimum acronym length
MIN_ACRONYM_LENGTH = 2
MAX_ACRONYM_LENGTH = 10

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================
# Prediction string separator
PREDICTION_SEPARATOR = " | "

# Minimum confidence threshold for predictions
MIN_CONFIDENCE = 0.0

# Maximum predictions per document
MAX_PREDICTIONS = 100

# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================
# Use GPU if available
USE_GPU = True

# Number of workers for data loading
NUM_WORKERS = 4

# Batch processing
PROCESS_BATCH_SIZE = 10

# Verbose logging
VERBOSE = True
