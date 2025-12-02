"""
Phase I: Data Cleaning & Baseline Modeling
Implements baseline string matching for dataset identification.
"""

import pandas as pd
import re
from typing import Set, Dict
from nltk.corpus import stopwords
import config
import utils


class PhaseIBaseline:
    """Phase I: Data cleaning and baseline matching."""
    
    def __init__(self):
        """Initialize Phase I processor."""
        self.stopwords = set()
        if config.USE_STOPWORDS:
            try:
                import nltk
                nltk.download('stopwords', quiet=True)
                self.stopwords = set(stopwords.words('english'))
                self.stopwords.update(config.CUSTOM_STOPWORDS)
            except:
                print("Warning: Could not load stopwords")
        
        self.external_datasets = utils.load_external_datasets(config.BIG_GOV_DATASETS)
    
    def merge_text_to_dataframe(self, df: pd.DataFrame, json_dir: str) -> pd.DataFrame:
        """
        Merge publication text from JSON files into dataframe.
        
        Args:
            df: DataFrame with publication IDs
            json_dir: Directory containing JSON files
        
        Returns:
            DataFrame with added 'text' column
        """
        print("Loading publication texts from JSON files...")
        
        # Get publication IDs
        pub_ids = df['Id'].unique().tolist() if 'Id' in df.columns else None
        
        # Load texts
        pub_texts = utils.load_json_publications(json_dir, pub_ids)
        
        # Add text column
        df['text'] = df['Id'].map(pub_texts).fillna('')
        
        print(f"Loaded text for {len(pub_texts)} publications")
        return df
    
    def clean_dataframe_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply text cleaning to the text column.
        
        Args:
            df: DataFrame with 'text' column
        
        Returns:
            DataFrame with added 'cleaned_text' column
        """
        print("Cleaning text...")
        
        df['cleaned_text'] = df['text'].apply(utils.clean_text)
        
        # Remove stopwords if configured
        if config.USE_STOPWORDS and self.stopwords:
            print("Removing stopwords...")
            df['cleaned_text'] = df['cleaned_text'].apply(
                lambda x: utils.remove_stopwords(x, self.stopwords)
            )
        
        return df
    
    def create_internal_labels(self, train_df: pd.DataFrame) -> Set[str]:
        """
        Create set of internal dataset labels from training data.
        
        Args:
            train_df: Training DataFrame with dataset columns
        
        Returns:
            Set of cleaned dataset labels
        """
        labels = set()
        
        # Columns to extract labels from
        label_columns = ['dataset_title', 'dataset_label', 'cleaned_label']
        
        for col in label_columns:
            if col in train_df.columns:
                # Extract non-null values, clean, and add to set
                values = train_df[col].dropna().unique()
                for val in values:
                    if isinstance(val, str):
                        cleaned = utils.clean_text(val)
                        if cleaned:
                            labels.add(cleaned)
        
        print(f"Created {len(labels)} internal dataset labels")
        return labels
    
    def internal_baseline_matching(self, text: str, labels: Set[str]) -> Set[str]:
        """
        Match internal labels against text.
        
        Args:
            text: Cleaned publication text
            labels: Set of dataset labels to match
        
        Returns:
            Set of matched dataset labels
        """
        matches = set()
        
        for label in labels:
            if label in text:
                matches.add(label)
        
        return matches
    
    def external_baseline_matching(self, text: str) -> Set[str]:
        """
        Match external datasets against text.
        
        Args:
            text: Cleaned publication text
        
        Returns:
            Set of matched external dataset names
        """
        matches = set()
        
        for dataset in self.external_datasets:
            if dataset in text:
                matches.add(dataset)
        
        return matches
    
    def process(self, df: pd.DataFrame, json_dir: str, train_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Run complete Phase I pipeline.
        
        Args:
            df: DataFrame to process (test or train)
            json_dir: Directory with JSON files
            train_df: Training DataFrame for label extraction (optional)
        
        Returns:
            DataFrame with Phase I predictions
        """
        print("\n" + "="*60)
        print("PHASE I: DATA CLEANING & BASELINE MODELING")
        print("="*60)
        
        # Step 1: Merge text
        df = self.merge_text_to_dataframe(df, json_dir)
        
        # Step 2: Clean text
        df = self.clean_dataframe_text(df)
        
        # Step 3: Create internal labels
        internal_labels = set()
        if train_df is not None:
            internal_labels = self.create_internal_labels(train_df)
        
        # Step 4: Perform matching
        print("Performing baseline matching...")
        phase1_predictions = []
        
        for idx, row in df.iterrows():
            text = row['cleaned_text']
            
            # Internal matching
            internal_matches = self.internal_baseline_matching(text, internal_labels)
            
            # External matching
            external_matches = self.external_baseline_matching(text)
            
            # Combine matches
            all_matches = internal_matches.union(external_matches)
            
            # Format prediction string
            pred_string = utils.format_prediction_string(all_matches)
            phase1_predictions.append(pred_string)
        
        df['phase1_predictions'] = phase1_predictions
        
        print(f"Phase I complete. Generated predictions for {len(df)} publications")
        return df
