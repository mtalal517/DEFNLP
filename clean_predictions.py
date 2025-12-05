import pandas as pd
import os
from groq import Groq
import json
import time

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def clean_with_llm(messy_text, doc_id):
    """
    Use Groq LLM to extract clean dataset names and their potential sources/links
    """
    prompt = f"""You are a data extraction expert. Analyze the following messy text that contains extracted data citations from a scientific paper.

Your task:
1. Identify distinct dataset names, databases, or data sources mentioned
2. For each dataset, provide:
   - Clean dataset name
   - Full name/expansion (if it's an acronym)
   - Potential source/link (if you can infer it from context, e.g., government agencies, research institutions)
   - Brief description of what the dataset contains

Messy text:
{messy_text}

Return your response as a JSON array with this structure:
[
  {{
    "dataset_name": "Clean dataset name",
    "full_name": "Full expansion if acronym",
    "potential_source": "Organization or URL if known",
    "description": "Brief description",
    "confidence": "high/medium/low"
  }}
]

Only include actual datasets, databases, or data sources. Exclude:
- Generic terms (e.g., "data", "study", "analysis")
- Years or dates
- Statistical methods
- General concepts

Return ONLY the JSON array, no additional text."""

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.3-70b-versatile",  # Using a capable model
            temperature=0.1,  # Low temperature for more consistent extraction
            max_tokens=2000,
        )
        
        response_text = chat_completion.choices[0].message.content
        
        # Try to parse JSON from response
        # Sometimes LLMs add markdown code blocks, so we need to clean it
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        datasets = json.loads(response_text)
        return datasets
    
    except Exception as e:
        print(f"Error processing document {doc_id}: {str(e)}")
        return []

def main():
    # Read the predictions CSV
    print("Reading predictions.csv...")
    df = pd.read_csv('output/predictions.csv')
    
    print(f"Found {len(df)} documents to process")
    
    all_results = []
    
    # Process each row
    for idx, row in df.iterrows():
        doc_id = row['Id']
        messy_text = row['PredictionString']
        
        print(f"\nProcessing document {idx + 1}/{len(df)}: {doc_id}")
        
        # Clean with LLM
        datasets = clean_with_llm(messy_text, doc_id)
        
        # Add document ID to each dataset
        for dataset in datasets:
            dataset['document_id'] = doc_id
            all_results.append(dataset)
        
        print(f"  Found {len(datasets)} datasets")
        
        # Rate limiting - be nice to the API
        time.sleep(1)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save to CSV
    output_path = 'output/cleaned_datasets.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n✅ Cleaned data saved to {output_path}")
    
    # Also save as JSON for better structure
    json_output_path = 'output/cleaned_datasets.json'
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"✅ JSON version saved to {json_output_path}")
    
    # Print summary statistics
    print(f"\n📊 Summary:")
    print(f"Total documents processed: {len(df)}")
    print(f"Total datasets extracted: {len(results_df)}")
    print(f"Unique datasets: {results_df['dataset_name'].nunique()}")
    
    # Show high-confidence datasets
    if 'confidence' in results_df.columns:
        high_conf = results_df[results_df['confidence'] == 'high']
        print(f"High-confidence datasets: {len(high_conf)}")
    
    # Show top datasets by frequency
    print("\n🔝 Top 10 most mentioned datasets:")
    print(results_df['dataset_name'].value_counts().head(10))

if __name__ == "__main__":
    main()
