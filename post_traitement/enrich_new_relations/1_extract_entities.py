import os
import csv
import re
import sys

import spacy

from helpers import extract_ingredients_terms, extract_utensils_terms

# Add the current directory to the path so we can import helpers
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def extract_section(text, start_marker, end_marker):
    """Extract text between two markers"""
    pattern = rf"{start_marker}(.*?){end_marker}"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def main():
    # Directory containing the recipe files
    nlp = spacy.load("fr_core_news_lg")

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    recipe_dir = os.path.join(base_dir, 'corpus', 'marmiton')
    # Output CSV file
    output_file = os.path.join(base_dir, 'post_traitement', 'enrich_new_relations', 'recipe_entities.csv')
    
    # Create CSV file and write header
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['filename', 'entity', 'type'])
        
        # Counters for statistics
        total_ingredients = 0
        total_utensils = 0
        files_with_data = 0
        
        # Loop through all .txt files in the directory
        total_files = len([f for f in os.listdir(recipe_dir) if f.endswith('.txt')])
        print(f"Found {total_files} .txt files to process")
        
        processed_count = 0
        for filename in os.listdir(recipe_dir):
            if filename.endswith('.txt'):
                file_path = os.path.join(recipe_dir, filename)
                processed_count += 1
                
                if processed_count % 10 == 0 or processed_count == 1 or processed_count == total_files:
                    print(f"Processing file {processed_count}/{total_files}: {filename}")
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Extract ingredients section
                    ingredients_section = extract_section(content, "Ingrédients:", "Ustensiles:")
                    ingredients = extract_ingredients_terms(ingredients_section, nlp)
                    
                    # Extract utensils section
                    utensils_section = extract_section(content, "Ustensiles:", "Étapes:")
                    utensils = extract_utensils_terms(utensils_section, nlp)
                    
                    # Write ingredients to CSV
                    for ingredient in ingredients:
                        csvwriter.writerow([filename, ingredient, 'ingredient'])
                        total_ingredients += 1
                    
                    # Write utensils to CSV
                    for utensil in utensils:
                        csvwriter.writerow([filename, utensil, 'utensil'])
                        total_utensils += 1
                    
                    # Count files with data
                    if ingredients or utensils:
                        files_with_data += 1
                    
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        
        # Print summary statistics
        print(f"\nExtraction complete:")
        print(f"- Processed {total_files} files")
        print(f"- Found data in {files_with_data} files")
        print(f"- Extracted {total_ingredients} ingredients")
        print(f"- Extracted {total_utensils} utensils")
        print(f"- Results saved to {output_file}")

if __name__ == "__main__":
    main()
