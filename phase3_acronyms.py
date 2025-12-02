"""
Phase III: Acronym & Abbreviation Extraction
Extracts acronyms and matches them with full forms.
"""

import pandas as pd
import re
from typing import Set, Dict, Tuple, List
import config
import utils


class PhaseIIIAcronyms:
    """Phase III: Acronym and abbreviation extraction."""
    
    def __init__(self):
        """Initialize Phase III processor."""
        pass
    
    def extract_acronyms(self, text: str) -> Set[str]:
        """
        Extract potential acronyms from text.
        
        Args:
            text: Input text
        
        Returns:
            Set of potential acronyms
        """
        acronyms = set()
        
        # Pattern 1: Uppercase words (2-10 characters)
        # Matches: ADNI, NASA, CDC, etc.
        pattern1 = r'\b[A-Z]{' + str(config.MIN_ACRONYM_LENGTH) + ',' + str(config.MAX_ACRONYM_LENGTH) + r'}\b'
        matches1 = re.findall(pattern1, text)
        acronyms.update(matches1)
        
        # Pattern 2: Acronyms in parentheses
        # Matches: "National Institutes of Health (NIH)"
        pattern2 = r'\(([A-Z]{' + str(config.MIN_ACRONYM_LENGTH) + ',' + str(config.MAX_ACRONYM_LENGTH) + r'})\)'
        matches2 = re.findall(pattern2, text)
        acronyms.update(matches2)
        
        return acronyms
    
    def extract_abbreviation_acronym_pairs(self, text: str) -> Dict[str, str]:
        """
        Extract abbreviation-acronym pairs from text.
        Looks for patterns like "Full Name (ACRONYM)".
        
        Args:
            text: Input text
        
        Returns:
            Dictionary mapping acronyms to their full forms
        """
        pairs = {}
        
        # Pattern: "Full Name (ACRONYM)"
        # Captures text before parentheses and acronym inside
        pattern = r'([A-Z][a-zA-Z\s\-]+?)\s*\(([A-Z]{' + str(config.MIN_ACRONYM_LENGTH) + ',' + str(config.MAX_ACRONYM_LENGTH) + r'})\)'
        
        matches = re.findall(pattern, text)
        
        for full_form, acronym in matches:
            full_form = full_form.strip()
            acronym = acronym.strip()
            
            # Validate that acronym matches full form
            if self._validate_acronym_match(full_form, acronym):
                pairs[acronym] = full_form
        
        return pairs
    
    def _validate_acronym_match(self, full_form: str, acronym: str) -> bool:
        """
        Validate that an acronym reasonably matches its full form.
        
        Args:
            full_form: Full form text
            acronym: Acronym
        
        Returns:
            True if valid match
        """
        # Simple validation: check if first letters match
        words = full_form.split()
        
        # Filter out common small words
        significant_words = [w for w in words if len(w) > 2 or w[0].isupper()]
        
        if len(significant_words) == 0:
            return False
        
        # Check if acronym length is reasonable
        if len(acronym) > len(significant_words) + 2:
            return False
        
        # Check if first letters roughly match
        first_letters = ''.join([w[0].upper() for w in significant_words])
        
        # Allow some flexibility
        return acronym in first_letters or first_letters.startswith(acronym)
    
    def match_acronyms_with_predictions(
        self,
        acronyms: Set[str],
        previous_predictions: str
    ) -> Set[str]:
        """
        Match extracted acronyms with previous predictions.
        
        Args:
            acronyms: Set of extracted acronyms
            previous_predictions: Pipe-separated prediction string from Phase II
        
        Returns:
            Set of matched dataset mentions (both acronym and full form)
        """
        matches = set()
        
        if not previous_predictions:
            return matches
        
        # Split previous predictions
        prev_preds = [p.strip() for p in previous_predictions.split('|')]
        
        for acronym in acronyms:
            acronym_lower = acronym.lower()
            
            # Check if acronym appears in any previous prediction
            for pred in prev_preds:
                if acronym_lower in pred.lower():
                    # Add both the acronym and the full prediction
                    matches.add(acronym_lower)
                    matches.add(pred)
        
        return matches
    
    def create_acronym_variants(
        self,
        acronym_pairs: Dict[str, str]
    ) -> Set[str]:
        """
        Create dataset name variants from acronym-abbreviation pairs.
        
        Args:
            acronym_pairs: Dictionary mapping acronyms to full forms
        
        Returns:
            Set of dataset name variants
        """
        variants = set()
        
        for acronym, full_form in acronym_pairs.items():
            # Add cleaned versions
            acronym_clean = utils.clean_text(acronym)
            full_form_clean = utils.clean_text(full_form)
            
            # Add individual forms
            variants.add(acronym_clean)
            variants.add(full_form_clean)
            
            # Add combined form: "full form acronym"
            combined = f"{full_form_clean} {acronym_clean}"
            variants.add(combined)
        
        return variants
    
    def process_single_text(
        self,
        text: str,
        previous_predictions: str
    ) -> Set[str]:
        """
        Process a single text through Phase III pipeline.
        
        Args:
            text: Input text (original, not cleaned)
            previous_predictions: Predictions from Phase II
        
        Returns:
            Set of extracted dataset mentions with acronyms
        """
        all_extractions = set()
        
        # Step 1: Extract acronyms
        acronyms = self.extract_acronyms(text)
        
        # Step 2: Extract abbreviation-acronym pairs
        acronym_pairs = self.extract_abbreviation_acronym_pairs(text)
        
        # Step 3: Match acronyms with previous predictions
        matched = self.match_acronyms_with_predictions(acronyms, previous_predictions)
        all_extractions.update(matched)
        
        # Step 4: Create variants from pairs
        variants = self.create_acronym_variants(acronym_pairs)
        all_extractions.update(variants)
        
        return all_extractions
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run complete Phase III pipeline.
        
        Args:
            df: DataFrame with 'text' and 'phase2_predictions' columns
        
        Returns:
            DataFrame with Phase III predictions
        """
        print("\n" + "="*60)
        print("PHASE III: ACRONYM & ABBREVIATION EXTRACTION")
        print("="*60)
        
        phase3_predictions = []
        
        for idx, row in df.iterrows():
            if config.VERBOSE and idx % 10 == 0:
                print(f"Processing {idx}/{len(df)}...")
            
            text = row.get('text', '')
            phase2_preds = row.get('phase2_predictions', '')
            
            # Extract acronyms and match with previous predictions
            extractions = self.process_single_text(text, phase2_preds)
            
            # Format prediction string
            pred_string = utils.format_prediction_string(extractions)
            phase3_predictions.append(pred_string)
        
        df['phase3_predictions'] = phase3_predictions
        
        print(f"Phase III complete. Generated predictions for {len(df)} publications")
        return df
