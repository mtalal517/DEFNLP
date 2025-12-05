"""
Main DEFNLP Pipeline
Orchestrates all three phases and generates final predictions.
"""

import pandas as pd
import os
import time
from typing import Optional
import config
import utils
from phase1_baseline import PhaseIBaseline
from phase2_ner_qa import PhaseIINER_QA
from phase3_acronyms import PhaseIIIAcronyms


class DEFNLPPipeline:
    """Main pipeline orchestrator for DEFNLP methodology."""
    
    def __init__(self, use_gpu: bool = None):
        """
        Initialize DEFNLP pipeline.
        
        Args:
            use_gpu: Whether to use GPU for Phase II. If None, uses config setting.
        """
        print("\n" + "="*60)
        print("INITIALIZING DEFNLP PIPELINE")
        print("="*60)
        
        self.phase1 = PhaseIBaseline()
        self.phase2 = PhaseIINER_QA(use_gpu=use_gpu)
        self.phase3 = PhaseIIIAcronyms()
        
        print("Pipeline initialized successfully\n")
    
    def merge_all_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge predictions from all three phases.
        
        Args:
            df: DataFrame with phase1, phase2, and phase3 predictions
        
        Returns:
            DataFrame with final merged predictions
        """
        print("\n" + "="*60)
        print("MERGING ALL PHASE PREDICTIONS")
        print("="*60)
        
        final_predictions = []
        
        for idx, row in df.iterrows():
            # Collect all predictions
            all_preds = [
                row.get('phase1_predictions', ''),
                row.get('phase2_predictions', ''),
                row.get('phase3_predictions', '')
            ]
            
            # Merge and format
            merged = utils.merge_prediction_strings(all_preds)
            final_predictions.append(merged)
        
        df['PredictionString'] = final_predictions
        
        print(f"Merged predictions for {len(df)} publications")
        return df
    
    def run_inference(
        self,
        test_csv_path: str = None,
        test_json_dir: str = None,
        train_csv_path: str = None,
        output_path: str = None
    ) -> pd.DataFrame:
        """
        Run inference on test data.
        
        Args:
            test_csv_path: Path to test CSV file
            test_json_dir: Directory with test JSON files
            train_csv_path: Path to training CSV (for baseline labels)
            output_path: Path to save predictions
        
        Returns:
            DataFrame with final predictions
        """
        # Start total pipeline timer
        pipeline_start_time = time.time()
        
        # Use defaults from config if not provided
        test_csv_path = test_csv_path or config.TEST_CSV
        test_json_dir = test_json_dir or config.TEST_JSON_DIR
        train_csv_path = train_csv_path or config.TRAIN_CSV
        output_path = output_path or os.path.join(config.OUTPUT_DIR, "predictions2.csv")
        
        print("\n" + "="*60)
        print("RUNNING DEFNLP INFERENCE")
        print("="*60)
        print(f"Test CSV: {test_csv_path}")
        print(f"Test JSON Dir: {test_json_dir}")
        print(f"Train CSV: {train_csv_path}")
        print(f"Output: {output_path}")
        
        # Load data
        print("\nLoading test data...")
        test_df = pd.read_csv(test_csv_path)
        
        train_df = None
        if os.path.exists(train_csv_path):
            print("Loading training data for baseline labels...")
            train_df = pd.read_csv(train_csv_path)
        
        # Run Phase I
        phase1_start = time.time()
        test_df = self.phase1.process(test_df, test_json_dir, train_df)
        phase1_time = time.time() - phase1_start
        print(f"\n⏱️  Phase I completed in {phase1_time:.2f} seconds ({phase1_time/60:.2f} minutes)")
        
        # Run Phase II
        phase2_start = time.time()
        test_df = self.phase2.process(test_df)
        phase2_time = time.time() - phase2_start
        print(f"\n⏱️  Phase II completed in {phase2_time:.2f} seconds ({phase2_time/60:.2f} minutes)")
        
        # Run Phase III
        phase3_start = time.time()
        test_df = self.phase3.process(test_df)
        phase3_time = time.time() - phase3_start
        print(f"\n⏱️  Phase III completed in {phase3_time:.2f} seconds ({phase3_time/60:.2f} minutes)")
        
        # Merge all predictions
        test_df = self.merge_all_predictions(test_df)
        
        # Prepare output
        output_df = test_df[['Id', 'PredictionString']].copy()
        
        # Save predictions
        utils.create_output_directory()
        utils.save_predictions(output_df, output_path)
        
        # Calculate total time
        total_time = time.time() - pipeline_start_time
        
        print("\n" + "="*60)
        print("INFERENCE COMPLETE")
        print("="*60)
        print("\n📊 TIMING SUMMARY:")
        print(f"  Phase I (Baseline):     {phase1_time:.2f}s ({phase1_time/60:.2f} min)")
        print(f"  Phase II (NER & QA):    {phase2_time:.2f}s ({phase2_time/60:.2f} min)")
        print(f"  Phase III (Acronyms):   {phase3_time:.2f}s ({phase3_time/60:.2f} min)")
        print(f"  {'─'*40}")
        print(f"  TOTAL PIPELINE TIME:    {total_time:.2f}s ({total_time/60:.2f} min)")
        print("="*60)
        
        return output_df
    
    def run_full_pipeline(
        self,
        train_csv_path: str = None,
        train_json_dir: str = None,
        test_csv_path: str = None,
        test_json_dir: str = None,
        output_path: str = None
    ) -> pd.DataFrame:
        """
        Run complete pipeline on both training and test data.
        
        Args:
            train_csv_path: Path to training CSV
            train_json_dir: Directory with training JSON files
            test_csv_path: Path to test CSV file
            test_json_dir: Directory with test JSON files
            output_path: Path to save predictions
        
        Returns:
            DataFrame with final predictions
        """
        # Use defaults from config if not provided
        train_csv_path = train_csv_path or config.TRAIN_CSV
        train_json_dir = train_json_dir or config.TRAIN_JSON_DIR
        
        # Run inference (which uses training data for baseline)
        return self.run_inference(
            test_csv_path=test_csv_path,
            test_json_dir=test_json_dir,
            train_csv_path=train_csv_path,
            output_path=output_path
        )


def main():
    """Main entry point for running the pipeline."""
    # Create pipeline
    pipeline = DEFNLPPipeline()
    
    # Run full pipeline
    predictions = pipeline.run_full_pipeline()
    
    print(f"\nGenerated {len(predictions)} predictions")
    print(f"\nSample predictions:")
    print(predictions.head(10))


if __name__ == "__main__":
    main()
