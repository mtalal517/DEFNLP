"""
Quick start script for DEFNLP pipeline.
Run this after installing dependencies to verify everything works.
"""

import os
import sys


def check_dependencies():
    """Check if all required dependencies are installed."""
    print("Checking dependencies...")
    
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'transformers': 'transformers',
        'torch': 'torch',
        'spacy': 'spacy',
        'nltk': 'nltk',
        'tqdm': 'tqdm',
        'sklearn': 'scikit-learn'
    }
    
    missing = []
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (missing)")
            missing.append(pip_name)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    
    print("\n✓ All dependencies installed!")
    return True


def check_spacy_model():
    """Check if SpaCy model is downloaded."""
    print("\nChecking SpaCy model...")
    
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("✓ SpaCy model 'en_core_web_sm' is installed")
        return True
    except:
        print("✗ SpaCy model 'en_core_web_sm' not found")
        print("Install with: python -m spacy download en_core_web_sm")
        return False


def check_data_files():
    """Check if required data files exist."""
    print("\nChecking data files...")
    
    required_files = {
        'train.csv': 'Training data',
        'sample_submission.csv': 'Test data template',
        'train': 'Training JSON directory',
        'test': 'Test JSON directory'
    }
    
    missing = []
    for file, description in required_files.items():
        if os.path.exists(file):
            print(f"✓ {file} ({description})")
        else:
            print(f"✗ {file} ({description}) - missing")
            missing.append(file)
    
    # Check for optional external datasets file
    if os.path.exists('big_gov_datasets.txt'):
        print("✓ big_gov_datasets.txt (optional)")
    else:
        print("⚠ big_gov_datasets.txt (optional) - not found")
    
    if missing:
        print(f"\nMissing required files: {', '.join(missing)}")
        return False
    
    print("\n✓ All required data files present!")
    return True


def run_quick_test():
    """Run a quick test of the pipeline."""
    print("\n" + "="*60)
    print("Running quick test...")
    print("="*60)
    
    try:
        # Test imports
        print("\nTesting imports...")
        import config
        import utils
        from phase1_baseline import PhaseIBaseline
        from phase2_ner_qa import PhaseIINER_QA
        from phase3_acronyms import PhaseIIIAcronyms
        from pipeline import DEFNLPPipeline
        print("✓ All modules imported successfully")
        
        # Test text cleaning
        print("\nTesting text cleaning...")
        sample_text = "This is a TEST with SPECIAL characters!!! @#$"
        cleaned = utils.clean_text(sample_text)
        print(f"Original: {sample_text}")
        print(f"Cleaned: {cleaned}")
        print("✓ Text cleaning works")
        
        # Test acronym extraction
        print("\nTesting acronym extraction...")
        phase3 = PhaseIIIAcronyms()
        sample_text = "National Institutes of Health (NIH) and Centers for Disease Control (CDC)"
        acronyms = phase3.extract_acronyms(sample_text)
        pairs = phase3.extract_abbreviation_acronym_pairs(sample_text)
        print(f"Extracted acronyms: {acronyms}")
        print(f"Extracted pairs: {pairs}")
        print("✓ Acronym extraction works")
        
        print("\n" + "="*60)
        print("✓ QUICK TEST PASSED!")
        print("="*60)
        print("\nYou're ready to run the full pipeline!")
        print("Run: python pipeline.py")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main setup verification."""
    print("="*60)
    print("DEFNLP SETUP VERIFICATION")
    print("="*60)
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Check SpaCy model
    spacy_ok = check_spacy_model()
    
    # Check data files
    data_ok = check_data_files()
    
    # Summary
    print("\n" + "="*60)
    print("SETUP SUMMARY")
    print("="*60)
    
    if deps_ok and spacy_ok:
        print("✓ Software dependencies: OK")
    else:
        print("✗ Software dependencies: INCOMPLETE")
    
    if data_ok:
        print("✓ Data files: OK")
    else:
        print("✗ Data files: INCOMPLETE")
    
    # Run quick test if software is ready
    if deps_ok and spacy_ok:
        run_quick_test()
    else:
        print("\nPlease install missing dependencies before running the pipeline.")
        print("\nInstallation steps:")
        print("1. pip install -r requirements.txt")
        print("2. python -m spacy download en_core_web_sm")
    
    if not data_ok:
        print("\nPlease ensure all required data files are in the project directory.")


if __name__ == "__main__":
    main()
