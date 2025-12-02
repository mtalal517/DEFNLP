"""
Utility functions for DEFNLP pipeline.
Contains helper functions for text processing, file I/O, and data manipulation.
"""

import os
import json
import re
import pandas as pd
from typing import List, Dict, Set, Tuple
import config


def load_json_publications(json_dir: str, pub_ids: List[str] = None) -> Dict[str, str]:
    """
    Load publication JSON files and merge all text content.
    
    Args:
        json_dir: Directory containing JSON files
        pub_ids: Optional list of publication IDs to load. If None, loads all.
    
    Returns:
        Dictionary mapping publication ID to merged text
    """
    pub_texts = {}
    
    if not os.path.exists(json_dir):
        print(f"Warning: Directory {json_dir} does not exist")
        return pub_texts
    
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    
    for json_file in json_files:
        pub_id = json_file.replace('.json', '')
        
        # Skip if specific IDs requested and this isn't one
        if pub_ids is not None and pub_id not in pub_ids:
            continue
        
        file_path = os.path.join(json_dir, json_file)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Merge all section texts
            text_parts = []
            for section in data:
                if 'text' in section and section['text']:
                    text_parts.append(section['text'])
            
            pub_texts[pub_id] = ' '.join(text_parts)
            
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    return pub_texts


def clean_text(text: str) -> str:
    """
    Clean text by removing special characters, emojis, and extra spaces.
    
    Args:
        text: Input text to clean
    
    Returns:
        Cleaned text in lowercase
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove emojis
    if config.REMOVE_EMOJIS:
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE
        )
        text = emoji_pattern.sub(r'', text)
    
    # Remove special characters but keep spaces and basic punctuation
    if config.REMOVE_SPECIAL_CHARS:
        text = re.sub(r'[^a-z0-9\s\.\,\-\(\)]', ' ', text)
    
    # Remove multiple spaces
    if config.REMOVE_MULTIPLE_SPACES:
        text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def remove_stopwords(text: str, stopwords: Set[str]) -> str:
    """
    Remove stopwords from text.
    
    Args:
        text: Input text
        stopwords: Set of stopwords to remove
    
    Returns:
        Text with stopwords removed
    """
    words = text.split()
    filtered_words = [w for w in words if w not in stopwords]
    return ' '.join(filtered_words)


def load_external_datasets(file_path: str) -> Set[str]:
    """
    Load external dataset names from file.
    
    Args:
        file_path: Path to external datasets file
    
    Returns:
        Set of dataset names (cleaned and lowercased)
    """
    datasets = set()
    
    if not os.path.exists(file_path):
        print(f"Warning: External datasets file {file_path} not found")
        return datasets
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                dataset = line.strip().lower()
                if dataset:
                    datasets.add(dataset)
    except Exception as e:
        print(f"Error loading external datasets: {e}")
    
    return datasets


def merge_prediction_strings(predictions: List[str]) -> str:
    """
    Merge multiple prediction strings, remove duplicates, and sort alphabetically.
    
    Args:
        predictions: List of prediction strings
    
    Returns:
        Single merged prediction string with pipe separator
    """
    # Flatten all predictions
    all_preds = set()
    
    for pred in predictions:
        if isinstance(pred, str) and pred:
            # Split by pipe and add each part
            parts = [p.strip() for p in pred.split('|')]
            all_preds.update([p for p in parts if p])
    
    # Sort alphabetically and join
    sorted_preds = sorted(list(all_preds))
    return config.PREDICTION_SEPARATOR.join(sorted_preds)


def format_prediction_string(datasets: Set[str]) -> str:
    """
    Format a set of datasets into a prediction string.
    
    Args:
        datasets: Set of dataset names
    
    Returns:
        Formatted prediction string (sorted, pipe-separated)
    """
    if not datasets:
        return ""
    
    # Clean and filter
    cleaned = [d.strip() for d in datasets if d and d.strip()]
    
    # Remove duplicates and sort
    unique_sorted = sorted(list(set(cleaned)))
    
    return config.PREDICTION_SEPARATOR.join(unique_sorted)


def extract_sentences_with_keywords(text: str, keywords: List[str]) -> List[str]:
    """
    Extract sentences containing specific keywords.
    
    Args:
        text: Input text
        keywords: List of keywords to search for
    
    Returns:
        List of sentences containing at least one keyword
    """
    # Split into sentences (simple approach)
    sentences = re.split(r'[.!?]+', text)
    
    matching_sentences = []
    keywords_lower = [k.lower() for k in keywords]
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        sentence_lower = sentence.lower()
        
        # Check if any keyword is in the sentence
        if any(keyword in sentence_lower for keyword in keywords_lower):
            matching_sentences.append(sentence)
    
    return matching_sentences


def chunk_sentences(sentences: List[str], chunk_size: int = 3, overlap: int = 1) -> List[str]:
    """
    Chunk sentences into overlapping groups.
    
    Args:
        sentences: List of sentences
        chunk_size: Number of sentences per chunk
        overlap: Number of sentences to overlap between chunks
    
    Returns:
        List of chunked text strings
    """
    chunks = []
    
    for i in range(0, len(sentences), chunk_size - overlap):
        chunk = sentences[i:i + chunk_size]
        if chunk:
            chunks.append(' '.join(chunk))
        
        # Stop if we've reached the end
        if i + chunk_size >= len(sentences):
            break
    
    return chunks


def save_predictions(predictions_df: pd.DataFrame, output_path: str):
    """
    Save predictions to CSV file.
    
    Args:
        predictions_df: DataFrame with Id and PredictionString columns
        output_path: Path to save CSV file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    predictions_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


def create_output_directory():
    """Create output directory if it doesn't exist."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
