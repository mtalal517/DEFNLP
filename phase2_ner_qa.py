"""
Phase II: SpaCy NER & BERT QA Modeling
Implements advanced NLP techniques for dataset extraction.
"""

import pandas as pd
from typing import List, Set, Dict
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import torch
import config
import utils


class PhaseIINER_QA:
    """Phase II: Named Entity Recognition and Question Answering."""
    
    def __init__(self, use_gpu: bool = None):
        """
        Initialize Phase II processor.
        
        Args:
            use_gpu: Whether to use GPU. If None, uses config setting.
        """
        if use_gpu is None:
            use_gpu = config.USE_GPU
        
        self.device = 0 if (use_gpu and torch.cuda.is_available()) else -1
        
        # Load SpaCy model
        print("Loading SpaCy model...")
        try:
            self.nlp = spacy.load(config.SPACY_MODEL)
        except:
            print(f"SpaCy model {config.SPACY_MODEL} not found. Please run:")
            print(f"python -m spacy download {config.SPACY_MODEL}")
            raise
        
        # Load BERT QA model
        print("Loading BERT QA model...")
        self.qa_pipeline = pipeline(
            "question-answering",
            model=config.QA_MODEL_NAME,
            tokenizer=config.QA_MODEL_NAME,
            device=self.device
        )
        
        print("Phase II models loaded successfully")
    
    def extract_keyword_sentences(self, text: str) -> List[str]:
        """
        Extract sentences containing data-related keywords.
        
        Args:
            text: Input text
        
        Returns:
            List of sentences containing keywords
        """
        return utils.extract_sentences_with_keywords(text, config.DATA_KEYWORDS)
    
    def chunk_text(self, sentences: List[str]) -> List[str]:
        """
        Chunk sentences for efficient processing.
        
        Args:
            sentences: List of sentences
        
        Returns:
            List of text chunks
        """
        return utils.chunk_sentences(
            sentences,
            chunk_size=config.CHUNK_SIZE,
            overlap=config.CHUNK_OVERLAP
        )
    
    def extract_ner_entities(self, text: str) -> Set[str]:
        """
        Extract named entities using SpaCy.
        
        Args:
            text: Input text
        
        Returns:
            Set of extracted entity texts
        """
        entities = set()
        
        # Process text with SpaCy
        doc = self.nlp(text)
        
        # Extract entities of specified types
        for ent in doc.ents:
            if ent.label_ in config.NER_ENTITY_TYPES:
                # Clean and add entity text
                entity_text = utils.clean_text(ent.text)
                if entity_text:
                    entities.add(entity_text)
        
        return entities
    
    def qa_extraction(self, chunks: List[str]) -> Set[str]:
        """
        Extract dataset mentions using BERT QA model.
        
        Args:
            chunks: List of text chunks to query
        
        Returns:
            Set of extracted answers
        """
        answers = set()
        
        # Ask each question on each chunk
        for question in config.QA_QUESTIONS:
            for chunk in chunks:
                if not chunk.strip():
                    continue
                
                try:
                    # Query the model
                    result = self.qa_pipeline(
                        question=question,
                        context=chunk,
                        max_answer_len=config.QA_MAX_ANSWER_LENGTH
                    )
                    
                    # Extract answer if confidence is sufficient
                    if result['score'] >= config.MIN_CONFIDENCE:
                        answer = utils.clean_text(result['answer'])
                        if answer:
                            answers.add(answer)
                
                except Exception as e:
                    if config.VERBOSE:
                        print(f"QA error: {e}")
                    continue
        
        return answers
    
    def process_single_text(self, text: str) -> Set[str]:
        """
        Process a single text through Phase II pipeline.
        
        Args:
            text: Input text
        
        Returns:
            Set of extracted dataset mentions
        """
        all_extractions = set()
        
        # Step 1: Extract sentences with keywords
        keyword_sentences = self.extract_keyword_sentences(text)
        
        if not keyword_sentences:
            return all_extractions
        
        # Step 2: Chunk sentences
        chunks = self.chunk_text(keyword_sentences)
        
        # Step 3: NER extraction on full text
        ner_entities = self.extract_ner_entities(text)
        all_extractions.update(ner_entities)
        
        # Step 4: QA extraction on chunks
        qa_answers = self.qa_extraction(chunks)
        all_extractions.update(qa_answers)
        
        return all_extractions
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run complete Phase II pipeline.
        
        Args:
            df: DataFrame with 'text' column
        
        Returns:
            DataFrame with Phase II predictions
        """
        print("\n" + "="*60)
        print("PHASE II: SPACY NER & BERT QA MODELING")
        print("="*60)
        
        phase2_predictions = []
        
        for idx, row in df.iterrows():
            if config.VERBOSE and idx % 10 == 0:
                print(f"Processing {idx}/{len(df)}...")
            
            text = row.get('text', '')
            
            # Extract datasets using NER and QA
            extractions = self.process_single_text(text)
            
            # Format prediction string
            pred_string = utils.format_prediction_string(extractions)
            phase2_predictions.append(pred_string)
        
        df['phase2_predictions'] = phase2_predictions
        
        print(f"Phase II complete. Generated predictions for {len(df)} publications")
        return df
