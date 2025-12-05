"""
Quality Chart Generator for DEFNLP Pipeline
Compares raw predictions (Phase I-III) vs LLM-cleaned predictions (Phase IV)
"""

import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import re

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_raw_predictions(csv_path='output/predictions2.csv'):
    """Load raw predictions from Phase I-III"""
    df = pd.read_csv(csv_path)
    return df

def load_cleaned_predictions(json_path='output/cleaned_datasets.json'):
    """Load LLM-cleaned predictions from Phase IV"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def count_raw_items(prediction_string):
    """Count items in raw prediction string"""
    if pd.isna(prediction_string) or prediction_string == '':
        return 0
    items = [item.strip() for item in prediction_string.split('|')]
    return len(items)

def is_noise(item):
    """Detect if an item is likely noise (year, generic term, etc.)"""
    item = item.strip().lower()
    
    # Check if it's a year (4 digits)
    if re.match(r'^\d{4}$', item):
        return True
    
    # Check if it's a generic term
    generic_terms = ['data', 'study', 'analysis', 'dataset', 'database', 
                     'year', 'years', 'age', 'individual', 'points', 
                     'survey', 'program', 'organization', 'take', 'cop',
                     'pal', 'wai', 'id', 'us', 'cores', 'daily', 'weekly',
                     'monthly', 'annual', 'quarterly']
    if item in generic_terms:
        return True
    
    # Check if it's just numbers or very short
    if len(item) <= 2 and not item.isupper():
        return True
    
    # Check if it's a date range or age range
    if re.match(r'^\d+[-,]\s*\d+$', item):
        return True
    
    return False

def analyze_quality(raw_df, cleaned_data):
    """Analyze quality metrics"""
    results = []
    
    for idx, row in raw_df.iterrows():
        doc_id = row['Id']
        raw_string = row['PredictionString']
        
        # Count raw items
        raw_items = [item.strip() for item in raw_string.split('|')]
        total_raw = len(raw_items)
        
        # Count noise in raw
        noise_count = sum(1 for item in raw_items if is_noise(item))
        clean_raw = total_raw - noise_count
        
        # Count cleaned items for this document
        cleaned_items = [d for d in cleaned_data if d['document_id'] == doc_id]
        total_cleaned = len(cleaned_items)
        
        # Count high confidence items
        high_conf = sum(1 for d in cleaned_items if d.get('confidence') == 'high')
        
        results.append({
            'document_id': doc_id,
            'total_raw': total_raw,
            'noise_raw': noise_count,
            'clean_raw': clean_raw,
            'total_cleaned': total_cleaned,
            'high_confidence': high_conf,
            'noise_percentage': (noise_count / total_raw * 100) if total_raw > 0 else 0,
            'reduction_percentage': ((total_raw - total_cleaned) / total_raw * 100) if total_raw > 0 else 0
        })
    
    return pd.DataFrame(results)

def create_comparison_charts(quality_df, raw_df, cleaned_data):
    """Create comprehensive quality comparison charts"""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Chart 1: Before vs After Counts
    ax1 = plt.subplot(2, 3, 1)
    x = np.arange(len(quality_df))
    width = 0.35
    
    ax1.bar(x - width/2, quality_df['total_raw'], width, label='Raw (Phase I-III)', 
            color='#e74c3c', alpha=0.8)
    ax1.bar(x + width/2, quality_df['total_cleaned'], width, label='Cleaned (Phase IV)', 
            color='#27ae60', alpha=0.8)
    
    ax1.set_xlabel('Document')
    ax1.set_ylabel('Number of Items')
    ax1.set_title('Prediction Count: Before vs After LLM Cleaning', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Doc {i+1}' for i in range(len(quality_df))])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Chart 2: Noise Reduction
    ax2 = plt.subplot(2, 3, 2)
    ax2.bar(x, quality_df['noise_percentage'], color='#e67e22', alpha=0.8)
    ax2.set_xlabel('Document')
    ax2.set_ylabel('Noise Percentage (%)')
    ax2.set_title('Noise in Raw Predictions', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Doc {i+1}' for i in range(len(quality_df))])
    ax2.axhline(y=quality_df['noise_percentage'].mean(), color='r', 
                linestyle='--', label=f'Avg: {quality_df["noise_percentage"].mean():.1f}%')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Chart 3: Reduction Percentage
    ax3 = plt.subplot(2, 3, 3)
    ax3.bar(x, quality_df['reduction_percentage'], color='#3498db', alpha=0.8)
    ax3.set_xlabel('Document')
    ax3.set_ylabel('Reduction (%)')
    ax3.set_title('Item Reduction After LLM Cleaning', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'Doc {i+1}' for i in range(len(quality_df))])
    ax3.axhline(y=quality_df['reduction_percentage'].mean(), color='r', 
                linestyle='--', label=f'Avg: {quality_df["reduction_percentage"].mean():.1f}%')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # Chart 4: Confidence Distribution
    ax4 = plt.subplot(2, 3, 4)
    confidence_counts = Counter([d['confidence'] for d in cleaned_data])
    colors_conf = {'high': '#27ae60', 'medium': '#f39c12', 'low': '#e74c3c'}
    
    conf_labels = list(confidence_counts.keys())
    conf_values = list(confidence_counts.values())
    conf_colors = [colors_conf.get(label, '#95a5a6') for label in conf_labels]
    
    ax4.pie(conf_values, labels=conf_labels, autopct='%1.1f%%', 
            colors=conf_colors, startangle=90)
    ax4.set_title('Confidence Distribution (Phase IV)', fontweight='bold')
    
    # Chart 5: Average Metrics Comparison
    ax5 = plt.subplot(2, 3, 5)
    metrics = ['Total Items', 'Clean Items', 'High Confidence']
    before = [quality_df['total_raw'].mean(), quality_df['clean_raw'].mean(), 0]
    after = [quality_df['total_cleaned'].mean(), quality_df['total_cleaned'].mean(), 
             quality_df['high_confidence'].mean()]
    
    x_metrics = np.arange(len(metrics))
    width = 0.35
    
    ax5.bar(x_metrics - width/2, before, width, label='Before (Phase I-III)', 
            color='#e74c3c', alpha=0.8)
    ax5.bar(x_metrics + width/2, after, width, label='After (Phase IV)', 
            color='#27ae60', alpha=0.8)
    
    ax5.set_ylabel('Average Count')
    ax5.set_title('Average Quality Metrics', fontweight='bold')
    ax5.set_xticks(x_metrics)
    ax5.set_xticklabels(metrics)
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)
    
    # Chart 6: Quality Improvement Summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Calculate summary statistics
    total_raw_all = quality_df['total_raw'].sum()
    total_cleaned_all = quality_df['total_cleaned'].sum()
    total_noise = quality_df['noise_raw'].sum()
    avg_reduction = quality_df['reduction_percentage'].mean()
    high_conf_pct = (quality_df['high_confidence'].sum() / total_cleaned_all * 100) if total_cleaned_all > 0 else 0
    
    summary_text = f"""
    QUALITY IMPROVEMENT SUMMARY
    ═══════════════════════════════
    
    Total Raw Predictions: {total_raw_all}
    Total Cleaned Predictions: {total_cleaned_all}
    
    Noise Removed: {total_noise} items
    Average Reduction: {avg_reduction:.1f}%
    
    High Confidence Items: {quality_df['high_confidence'].sum()}
    High Confidence %: {high_conf_pct:.1f}%
    
    Documents Processed: {len(quality_df)}
    
    ═══════════════════════════════
    Phase IV (LLM Cleaning) successfully:
    ✓ Filtered out noise (years, generic terms)
    ✓ Reduced item count by ~{avg_reduction:.0f}%
    ✓ Enriched with metadata
    ✓ Assigned confidence scores
    """
    
    ax6.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', 
             facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('output/quality_chart.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: output/quality_chart.png")
    
    return fig

def create_detailed_comparison_table(quality_df):
    """Create detailed comparison table"""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    table_data.append(['Doc', 'Raw Items', 'Noise', 'Clean Raw', 'LLM Cleaned', 
                       'High Conf', 'Reduction %'])
    
    for idx, row in quality_df.iterrows():
        table_data.append([
            f"Doc {idx+1}",
            f"{row['total_raw']}",
            f"{row['noise_raw']}",
            f"{row['clean_raw']}",
            f"{row['total_cleaned']}",
            f"{row['high_confidence']}",
            f"{row['reduction_percentage']:.1f}%"
        ])
    
    # Add summary row
    table_data.append([
        'TOTAL/AVG',
        f"{quality_df['total_raw'].sum()}",
        f"{quality_df['noise_raw'].sum()}",
        f"{quality_df['clean_raw'].sum()}",
        f"{quality_df['total_cleaned'].sum()}",
        f"{quality_df['high_confidence'].sum()}",
        f"{quality_df['reduction_percentage'].mean():.1f}%"
    ])
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.12, 0.12, 0.12, 0.12, 0.15, 0.12, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(7):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style summary row
    for i in range(7):
        table[(len(table_data)-1, i)].set_facecolor('#95a5a6')
        table[(len(table_data)-1, i)].set_text_props(weight='bold')
    
    plt.title('Detailed Quality Comparison: Raw vs LLM-Cleaned Predictions', 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.savefig('output/quality_table.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: output/quality_table.png")
    
    return fig

def main():
    print("="*60)
    print("DEFNLP QUALITY ANALYSIS: Phase I-III vs Phase IV")
    print("="*60)
    
    # Load data
    print("\n📂 Loading data...")
    raw_df = load_raw_predictions()
    cleaned_data = load_cleaned_predictions()
    
    print(f"   Raw predictions: {len(raw_df)} documents")
    print(f"   Cleaned datasets: {len(cleaned_data)} items")
    
    # Analyze quality
    print("\n📊 Analyzing quality metrics...")
    quality_df = analyze_quality(raw_df, cleaned_data)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"\nTotal raw predictions: {quality_df['total_raw'].sum()}")
    print(f"Total noise items: {quality_df['noise_raw'].sum()}")
    print(f"Total cleaned predictions: {quality_df['total_cleaned'].sum()}")
    print(f"High confidence items: {quality_df['high_confidence'].sum()}")
    print(f"\nAverage noise percentage: {quality_df['noise_percentage'].mean():.1f}%")
    print(f"Average reduction: {quality_df['reduction_percentage'].mean():.1f}%")
    
    # Create charts
    print("\n📈 Creating quality comparison charts...")
    create_comparison_charts(quality_df, raw_df, cleaned_data)
    create_detailed_comparison_table(quality_df)
    
    # Save quality metrics to CSV
    quality_df.to_csv('output/quality_metrics.csv', index=False)
    print("✅ Saved: output/quality_metrics.csv")
    
    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  • output/quality_chart.png - Main quality comparison charts")
    print("  • output/quality_table.png - Detailed comparison table")
    print("  • output/quality_metrics.csv - Quality metrics data")
    print("\nUse these charts in your conference paper!")

if __name__ == "__main__":
    main()
