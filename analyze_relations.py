#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from collections import Counter
import json

# Define paths
BASE_DIR = os.path.dirname(__file__)
RESOURCES_DIR = os.path.join(BASE_DIR, "post_traitement/ressources_completes")
MARMITON_DIR = os.path.join(RESOURCES_DIR, "marmiton")
WIKIPEDIA_DIR = os.path.join(RESOURCES_DIR, "wikipedia")
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_results")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_csv_files(directory, sample_size=None):
    """Load and combine CSV files from a directory"""
    all_files = glob.glob(os.path.join(directory, "*_merged.csv"))
    
    if sample_size and sample_size < len(all_files):
        all_files = np.random.choice(all_files, sample_size, replace=False)
    
    print(f"Loading {len(all_files)} files from {directory}...")
    
    dfs = []
    for file in all_files:
        try:
            df = pd.read_csv(file, header=0)
            # Add source filename
            df['source'] = os.path.basename(file).replace('_merged.csv', '')
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if not dfs:
        return pd.DataFrame()
    
    return pd.concat(dfs, ignore_index=True)

def analyze_relation_distribution(df, source_name):
    """Analyze and visualize relation name distribution"""
    # Filter rows with relation_name (not empty)
    relations_df = df[df['relation_name'].notna() & (df['relation_name'] != '')]
    
    if len(relations_df) == 0:
        print(f"No relations found in the {source_name} data")
        return None
    
    # Count occurrences of each relation type
    relation_counts = Counter(relations_df['relation_name'])
    
    # Get top relations
    top_n = 10
    top_relations = relation_counts.most_common(top_n)
    
    # Prepare data for plotting
    labels = [r[0] for r in top_relations]
    counts = [r[1] for r in top_relations]
    
    # Create figure
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(labels)), counts)
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
    plt.xlabel('Relation Type')
    plt.ylabel('Count')
    plt.title(f'{source_name}: Top {top_n} Relations')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(OUTPUT_DIR, f"{source_name.lower()}_relation_distribution.png"))
    plt.close()

def analyze_similarity_distribution(df, source_name):
    """Analyze and visualize similarity distribution"""
    plt.figure(figsize=(10, 6))
    sns.histplot(df['sim'].dropna(), bins=30, kde=True)
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.title(f'{source_name}: Distribution of Similarity Scores')
    
    # Add mean and median lines
    mean_sim = df['sim'].mean()
    median_sim = df['sim'].median()
    plt.axvline(mean_sim, color='r', linestyle='--', label=f'Mean: {mean_sim:.3f}')
    plt.axvline(median_sim, color='g', linestyle='-.', label=f'Median: {median_sim:.3f}')
    plt.legend()
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(OUTPUT_DIR, f"{source_name.lower()}_similarity_distribution.png"))
    plt.close()

def create_combined_image(marmiton_df, wikipedia_df):
    """Create a combined image with visualizations for both datasets"""
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Marmiton relation distribution
    marmiton_relations = marmiton_df[marmiton_df['relation_name'].notna() & (marmiton_df['relation_name'] != '')]
    if len(marmiton_relations) > 0:
        relation_counts = Counter(marmiton_relations['relation_name'])
        top_relations = relation_counts.most_common(10)
        labels = [r[0] for r in top_relations]
        counts = [r[1] for r in top_relations]
        
        axs[0, 0].bar(range(len(labels)), counts)
        axs[0, 0].set_xticks(range(len(labels)))
        axs[0, 0].set_xticklabels(labels, rotation=45, ha='right')
        axs[0, 0].set_title('Marmiton: Top Relations')
    else:
        axs[0, 0].text(0.5, 0.5, "No relations found", ha='center', va='center')
        axs[0, 0].set_title('Marmiton: Relations')
    
    # 2. Wikipedia relation distribution
    wikipedia_relations = wikipedia_df[wikipedia_df['relation_name'].notna() & (wikipedia_df['relation_name'] != '')]
    if len(wikipedia_relations) > 0:
        relation_counts = Counter(wikipedia_relations['relation_name'])
        top_relations = relation_counts.most_common(10)
        labels = [r[0] for r in top_relations]
        counts = [r[1] for r in top_relations]
        
        axs[0, 1].bar(range(len(labels)), counts)
        axs[0, 1].set_xticks(range(len(labels)))
        axs[0, 1].set_xticklabels(labels, rotation=45, ha='right')
        axs[0, 1].set_title('Wikipedia: Top Relations')
    else:
        axs[0, 1].text(0.5, 0.5, "No relations found", ha='center', va='center')
        axs[0, 1].set_title('Wikipedia: Relations')
    
    # 3. Marmiton similarity distribution
    sns.histplot(marmiton_df['sim'].dropna(), bins=30, kde=True, ax=axs[1, 0])
    axs[1, 0].set_xlabel('Similarity Score')
    axs[1, 0].set_ylabel('Frequency')
    axs[1, 0].set_title('Marmiton: Similarity Distribution')
    
    # 4. Wikipedia similarity distribution
    sns.histplot(wikipedia_df['sim'].dropna(), bins=30, kde=True, ax=axs[1, 1])
    axs[1, 1].set_xlabel('Similarity Score')
    axs[1, 1].set_ylabel('Frequency')
    axs[1, 1].set_title('Wikipedia: Similarity Distribution')
    
    plt.tight_layout()
    plt.suptitle('Relation and Similarity Analysis', fontsize=16, y=0.98)
    plt.subplots_adjust(top=0.92)
    
    # Save the figure
    plt.savefig(os.path.join(OUTPUT_DIR, "combined_analysis.png"))
    plt.close()

def main():
    """Main function to run the analysis"""
    print("Starting analysis of CSV files...")
    
    # Sample size (set to None to use all files)
    sample_size = None
    
    # Load data
    marmiton_df = load_csv_files(MARMITON_DIR, sample_size)
    wikipedia_df = load_csv_files(WIKIPEDIA_DIR, sample_size)
    
    print(f"Loaded {len(marmiton_df)} rows from Marmiton")
    print(f"Loaded {len(wikipedia_df)} rows from Wikipedia")
    
    # Analyze relation distribution
    analyze_relation_distribution(marmiton_df, "Marmiton")
    analyze_relation_distribution(wikipedia_df, "Wikipedia")
    
    # Analyze similarity distribution
    analyze_similarity_distribution(marmiton_df, "Marmiton")
    analyze_similarity_distribution(wikipedia_df, "Wikipedia")
    
    # Create combined visualization
    create_combined_image(marmiton_df, wikipedia_df)
    
    # Generate basic stats
    stats = {
        "marmiton": {
            "rows": len(marmiton_df),
            "files": len(marmiton_df['source'].unique()) if 'source' in marmiton_df.columns else 0,
            "similarity_mean": float(marmiton_df['sim'].mean()),
            "similarity_median": float(marmiton_df['sim'].median()),
            "relations_count": len(marmiton_df[marmiton_df['relation_name'].notna() & (marmiton_df['relation_name'] != '')])
        },
        "wikipedia": {
            "rows": len(wikipedia_df),
            "files": len(wikipedia_df['source'].unique()) if 'source' in wikipedia_df.columns else 0,
            "similarity_mean": float(wikipedia_df['sim'].mean()),
            "similarity_median": float(wikipedia_df['sim'].median()),
            "relations_count": len(wikipedia_df[wikipedia_df['relation_name'].notna() & (wikipedia_df['relation_name'] != '')])
        }
    }
    
    # Save stats
    with open(os.path.join(OUTPUT_DIR, "analysis_stats.json"), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Analysis complete. Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
