"""
Test script to verify Groq API setup before running the full cleaning pipeline
"""
import os
from groq import Groq

def test_groq_connection():
    """Test if Groq API key is set and working"""
    
    # Check if API key is set
    api_key = os.environ.get("GROQ_API_KEY")
    
    if not api_key:
        print("❌ GROQ_API_KEY environment variable not found!")
        print("\nPlease set it using:")
        print("  PowerShell: $env:GROQ_API_KEY = 'your-api-key-here'")
        print("  Or see CLEANING_README.md for detailed instructions")
        return False
    
    print("✅ GROQ_API_KEY found")
    
    # Test API connection with a simple request
    try:
        print("\n🔄 Testing API connection...")
        client = Groq(api_key=api_key)
        
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": "Say 'API connection successful!' and nothing else.",
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=50,
        )
        
        response = chat_completion.choices[0].message.content
        print(f"✅ API Response: {response}")
        
        # Test with a sample dataset extraction
        print("\n🔄 Testing dataset extraction...")
        test_text = "1000 genomes | ADNI | framingham heart study | pubmed"
        
        prompt = f"""Extract dataset names from this text and return as JSON array:
{test_text}

Format: [{{"dataset_name": "name", "confidence": "high"}}]
Return ONLY the JSON, no other text."""

        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=500,
        )
        
        response = chat_completion.choices[0].message.content
        print(f"✅ Extraction test response:\n{response}")
        
        print("\n" + "="*50)
        print("✅ All tests passed! You're ready to run clean_predictions.py")
        print("="*50)
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing API: {str(e)}")
        print("\nPossible issues:")
        print("  1. Invalid API key")
        print("  2. Network connection problem")
        print("  3. Groq API service issue")
        return False

if __name__ == "__main__":
    print("="*50)
    print("Groq API Setup Test")
    print("="*50 + "\n")
    
    success = test_groq_connection()
    
    if success:
        print("\n📝 Next step: Run the main cleaning script")
        print("   python clean_predictions.py")
    else:
        print("\n📝 Please fix the issues above before proceeding")
        print("   See CLEANING_README.md for help")
