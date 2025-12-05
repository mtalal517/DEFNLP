"""
Enhanced cleaning script that also attempts to find actual URLs for datasets
using web search capabilities of the LLM
"""
import pandas as pd
import os
from groq import Groq
import json
import time

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def extract_and_enrich_datasets(messy_text, doc_id):
    """
    Use Groq LLM to extract datasets and provide enriched information including likely URLs
    """
    prompt = f"""You are a scientific data expert. Analyze this messy text containing data citations from a research paper.

Extract all datasets, databases, and data sources. For each one, provide:
1. Clean standardized name
2. Full official name (expand acronyms)
3. Organization/institution that maintains it
4. Likely official website URL (use your knowledge of common scientific datasets)
5. Brief description
6. Data type (e.g., genomic, survey, administrative, satellite, etc.)
7. Access type (public, restricted, requires registration, etc.)

Messy text:
{messy_text}

Return ONLY a JSON array with this exact structure:
[
  {{
    "dataset_name": "Short standardized name",
    "full_name": "Complete official name",
    "organization": "Maintaining organization",
    "url": "https://official-website.org or 'Unknown' if not sure",
    "description": "What the dataset contains",
    "data_type": "Type of data",
    "access": "public/restricted/registration",
    "confidence": "high/medium/low"
  }}
]

IMPORTANT:
- Only include actual datasets/databases, not years, methods, or concepts
- For well-known datasets (1000 Genomes, ADNI, etc.), provide accurate URLs
- If URL is uncertain, use "Unknown" - don't guess
- Be conservative with confidence ratings"""

    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=3000,
        )
        
        response_text = chat_completion.choices[0].message.content
        
        # Clean markdown code blocks if present
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        datasets = json.loads(response_text)
        return datasets
    
    except json.JSONDecodeError as e:
        print(f"  ⚠️  JSON parsing error for doc {doc_id}: {str(e)}")
        print(f"  Response was: {response_text[:200]}...")
        return []
    except Exception as e:
        print(f"  ❌ Error processing doc {doc_id}: {str(e)}")
        return []

def main():
    # Check for API key
    if not os.environ.get("GROQ_API_KEY"):
        print("❌ GROQ_API_KEY not set!")
        print("Please run: $env:GROQ_API_KEY = 'your-api-key'")
        print("Or see CLEANING_README.md for instructions")
        return
    
    # Read predictions
    print("📂 Reading predictions.csv...")
    df = pd.read_csv('output/predictions.csv')
    print(f"✅ Found {len(df)} documents to process\n")
    
    all_results = []
    
    # Process each document
    for idx, row in df.iterrows():
        doc_id = row['Id']
        messy_text = row['PredictionString']
        
        print(f"🔄 [{idx + 1}/{len(df)}] Processing: {doc_id}")
        
        # Extract and enrich
        datasets = extract_and_enrich_datasets(messy_text, doc_id)
        
        # Add document ID
        for dataset in datasets:
            dataset['document_id'] = doc_id
            all_results.append(dataset)
        
        print(f"   ✅ Extracted {len(datasets)} datasets")
        
        # Rate limiting
        time.sleep(1.5)  # Slightly longer delay for enriched requests
    
    # Create DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save outputs
    print("\n💾 Saving results...")
    
    # CSV output
    csv_path = 'output/enriched_datasets.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"   ✅ CSV saved: {csv_path}")
    
    # JSON output
    json_path = 'output/enriched_datasets.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"   ✅ JSON saved: {json_path}")
    
    # Create summary report
    print("\n" + "="*60)
    print("📊 SUMMARY REPORT")
    print("="*60)
    
    print(f"\n📄 Documents processed: {len(df)}")
    print(f"📦 Total datasets extracted: {len(results_df)}")
    print(f"🔢 Unique dataset names: {results_df['dataset_name'].nunique()}")
    
    if 'confidence' in results_df.columns:
        print(f"\n🎯 Confidence Distribution:")
        conf_dist = results_df['confidence'].value_counts()
        for conf, count in conf_dist.items():
            print(f"   {conf}: {count}")
    
    if 'access' in results_df.columns:
        print(f"\n🔓 Access Type Distribution:")
        access_dist = results_df['access'].value_counts()
        for acc, count in access_dist.items():
            print(f"   {acc}: {count}")
    
    # Datasets with known URLs
    if 'url' in results_df.columns:
        known_urls = results_df[results_df['url'] != 'Unknown']
        print(f"\n🔗 Datasets with known URLs: {len(known_urls)}")
    
    # Top datasets
    print(f"\n🏆 Top 10 Most Mentioned Datasets:")
    top_datasets = results_df['dataset_name'].value_counts().head(10)
    for i, (name, count) in enumerate(top_datasets.items(), 1):
        print(f"   {i}. {name}: {count} mentions")
    
    # High-confidence datasets with URLs
    if 'confidence' in results_df.columns and 'url' in results_df.columns:
        high_conf_urls = results_df[
            (results_df['confidence'] == 'high') & 
            (results_df['url'] != 'Unknown')
        ][['dataset_name', 'url', 'organization']].drop_duplicates()
        
        if len(high_conf_urls) > 0:
            print(f"\n🌟 High-Confidence Datasets with URLs:")
            for _, row in high_conf_urls.head(15).iterrows():
                print(f"   • {row['dataset_name']}")
                print(f"     URL: {row['url']}")
                print(f"     Org: {row['organization']}")
    
    print("\n" + "="*60)
    print("✅ Processing complete!")
    print("="*60)

if __name__ == "__main__":
    main()
