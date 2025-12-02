"""
Example script demonstrating how to use the DEFNLP pipeline.
"""

import pandas as pd
from pipeline import DEFNLPPipeline
import config


def example_basic_inference():
    """Basic example: Run inference on test data."""
    print("="*60)
    print("EXAMPLE 1: Basic Inference")
    print("="*60)
    
    # Create pipeline
    pipeline = DEFNLPPipeline()
    
    # Run inference (uses default paths from config.py)
    predictions = pipeline.run_full_pipeline()
    
    print(f"\nGenerated {len(predictions)} predictions")
    print("\nFirst 5 predictions:")
    print(predictions.head())


def example_custom_paths():
    """Example with custom file paths."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Custom Paths")
    print("="*60)
    
    # Create pipeline
    pipeline = DEFNLPPipeline()
    
    # Run with custom paths
    predictions = pipeline.run_full_pipeline(
        train_csv_path="./data/train.csv",
        train_json_dir="./data/train",
        test_csv_path="./data/test.csv",
        test_json_dir="./data/test",
        output_path="./results/my_predictions.csv"
    )
    
    print(f"\nGenerated {len(predictions)} predictions")


def example_phase_by_phase():
    """Example: Run each phase separately for inspection."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Phase-by-Phase Execution")
    print("="*60)
    
    # Load data
    test_df = pd.read_csv(config.TEST_CSV)
    train_df = pd.read_csv(config.TRAIN_CSV)
    
    # Initialize phases
    from phase1_baseline import PhaseIBaseline
    from phase2_ner_qa import PhaseIINER_QA
    from phase3_acronyms import PhaseIIIAcronyms
    
    phase1 = PhaseIBaseline()
    phase2 = PhaseIINER_QA()
    phase3 = PhaseIIIAcronyms()
    
    # Run Phase I
    print("\nRunning Phase I...")
    test_df = phase1.process(test_df, config.TEST_JSON_DIR, train_df)
    print("Phase I predictions sample:")
    print(test_df[['Id', 'phase1_predictions']].head())
    
    # Run Phase II
    print("\nRunning Phase II...")
    test_df = phase2.process(test_df)
    print("Phase II predictions sample:")
    print(test_df[['Id', 'phase2_predictions']].head())
    
    # Run Phase III
    print("\nRunning Phase III...")
    test_df = phase3.process(test_df)
    print("Phase III predictions sample:")
    print(test_df[['Id', 'phase3_predictions']].head())
    
    # Merge all predictions
    from utils import merge_prediction_strings
    test_df['PredictionString'] = test_df.apply(
        lambda row: merge_prediction_strings([
            row['phase1_predictions'],
            row['phase2_predictions'],
            row['phase3_predictions']
        ]),
        axis=1
    )
    
    print("\nFinal merged predictions:")
    print(test_df[['Id', 'PredictionString']].head())


def example_single_document():
    """Example: Process a single document."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Single Document Processing")
    print("="*60)
    
    # Sample text
    sample_text = """
    This study uses data from the Alzheimer's Disease Neuroimaging Initiative (ADNI).
    We also utilized the National Health and Nutrition Examination Survey (NHANES) data.
    Additional samples were obtained from the CDC database and PubMed repository.
    """
    
    # Initialize phases
    from phase1_baseline import PhaseIBaseline
    from phase2_ner_qa import PhaseIINER_QA
    from phase3_acronyms import PhaseIIIAcronyms
    from utils import clean_text, merge_prediction_strings
    
    print("Original text:")
    print(sample_text)
    
    # Phase I: Baseline matching
    phase1 = PhaseIBaseline()
    cleaned = clean_text(sample_text)
    phase1_matches = phase1.external_baseline_matching(cleaned)
    print(f"\nPhase I matches: {phase1_matches}")
    
    # Phase II: NER + QA
    phase2 = PhaseIINER_QA()
    phase2_matches = phase2.process_single_text(sample_text)
    print(f"\nPhase II matches: {phase2_matches}")
    
    # Phase III: Acronyms
    phase3 = PhaseIIIAcronyms()
    from utils import format_prediction_string
    phase2_string = format_prediction_string(phase2_matches)
    phase3_matches = phase3.process_single_text(sample_text, phase2_string)
    print(f"\nPhase III matches: {phase3_matches}")
    
    # Merge all
    all_matches = phase1_matches.union(phase2_matches).union(phase3_matches)
    final_prediction = format_prediction_string(all_matches)
    print(f"\nFinal prediction string:")
    print(final_prediction)


def example_cpu_only():
    """Example: Run pipeline on CPU only (no GPU)."""
    print("\n" + "="*60)
    print("EXAMPLE 5: CPU-Only Execution")
    print("="*60)
    
    # Create pipeline with GPU disabled
    pipeline = DEFNLPPipeline(use_gpu=False)
    
    # Run inference
    predictions = pipeline.run_inference()
    
    print(f"\nGenerated {len(predictions)} predictions on CPU")


if __name__ == "__main__":
    # Run examples
    # Uncomment the examples you want to run
    
    example_basic_inference()
    example_custom_paths()
    example_phase_by_phase()
    example_single_document()
    example_cpu_only()
