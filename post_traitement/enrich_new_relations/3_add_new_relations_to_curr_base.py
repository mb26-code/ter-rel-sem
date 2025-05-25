#!/usr/bin/env python3
"""
Script to enrich semantic_relations_import.csv with data from recipe_relations_refined.csv
Adds two new columns: new_relation and new_relation_w
"""

import csv
import os
from typing import Dict, Tuple, Optional

def load_recipe_relations(file_path: str) -> Dict[Tuple[str, str], Tuple[str, str]]:
    """
    Load recipe relations into a dictionary for fast lookup.
    
    Args:
        file_path: Path to recipe_relations_refined.csv
        
    Returns:
        Dictionary mapping (node1, node2) to (relation, w)
    """
    recipe_relations = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip comment lines
            if row['node1'].startswith('//'):
                continue
                
            key = (row['node1'].strip(), row['node2'].strip())
            value = (row['relation'].strip(), row['w'].strip())
            recipe_relations[key] = value
    
    print(f"Loaded {len(recipe_relations)} recipe relations")
    return recipe_relations

def process_semantic_relations(input_file: str, output_file: str, recipe_relations: Dict[Tuple[str, str], Tuple[str, str]]):
    """
    Process semantic relations and add new columns, plus add missing recipe relations.
    
    Args:
        input_file: Path to semantic_relations_import.csv
        output_file: Path to output file
        recipe_relations: Dictionary of recipe relations
    """
    found_count = 0
    not_found_count = 0
    processed_pairs = set()
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        
        reader = csv.DictReader(infile)
        
        # Add new columns to fieldnames
        fieldnames = list(reader.fieldnames) + ['new_relation', 'new_relation_w']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        
        # Write header
        writer.writeheader()
        
        # Process existing semantic relations
        for row in reader:
            # Skip comment lines
            if row['node1'].startswith('//'):
                continue
            
            node1 = row['node1'].strip()
            node2 = row['node2'].strip()
            key = (node1, node2)
            processed_pairs.add(key)
            
            # Look up in recipe relations
            if key in recipe_relations:
                relation, w = recipe_relations[key]
                row['new_relation'] = relation
                row['new_relation_w'] = w
                found_count += 1
            else:
                # Not found, leave empty
                row['new_relation'] = ''
                row['new_relation_w'] = ''
                not_found_count += 1
            
            writer.writerow(row)
        
        # Add missing recipe relations that weren't in semantic relations
        added_count = 0
        for (node1, node2), (relation, w) in recipe_relations.items():
            key = (node1, node2)
            if key not in processed_pairs:
                # Create new row with empty columns except for node1, node2, new_relation, new_relation_w
                new_row = {field: '' for field in fieldnames}
                new_row['node1'] = node1
                new_row['node2'] = node2
                new_row['new_relation'] = relation
                new_row['new_relation_w'] = w
                writer.writerow(new_row)
                added_count += 1
    
    print(f"Processing complete:")
    print(f"  - Found matches: {found_count}")
    print(f"  - Not found: {not_found_count}")
    print(f"  - Added new entries from recipe relations: {added_count}")
    print(f"  - Total processed: {found_count + not_found_count + added_count}")

def main():
    """Main function to execute the enrichment process."""
    # Define file paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Input files
    semantic_relations_file = os.path.join(base_dir, '..', 'semantic_relations_import.csv')
    recipe_relations_file = os.path.join(base_dir, 'recipe_relations_refined.csv')
    
    # Output file
    output_file = os.path.join(base_dir, 'semantic_relations_enriched.csv')
    
    # Check if input files exist
    if not os.path.exists(semantic_relations_file):
        print(f"Error: {semantic_relations_file} not found")
        return
    
    if not os.path.exists(recipe_relations_file):
        print(f"Error: {recipe_relations_file} not found")
        return
    
    print("Starting enrichment process...")
    print(f"Input semantic relations: {semantic_relations_file}")
    print(f"Input recipe relations: {recipe_relations_file}")
    print(f"Output file: {output_file}")
    
    # Load recipe relations
    recipe_relations = load_recipe_relations(recipe_relations_file)
    
    # Process semantic relations
    process_semantic_relations(semantic_relations_file, output_file, recipe_relations)
    
    print(f"Enriched data written to: {output_file}")

if __name__ == "__main__":
    main()