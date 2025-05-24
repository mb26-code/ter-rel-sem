#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
from collections import Counter, defaultdict
from itertools import permutations

# Input and output file paths
input_file = 'recipe_entities_refined.csv'
output_file = 'recipe_relations_refined.csv'

# Dictionary of entity corrections (misspelled -> correct)
ENTITY_CORRECTIONS = {
    'pôele': 'poêle',
    # Add more corrections as needed
}

def normalize_entity(entity):
    """Normalize an entity name by applying corrections from ENTITY_CORRECTIONS."""
    return ENTITY_CORRECTIONS.get(entity.lower(), entity)

def process_entities_to_relations():
    # Dictionary to store counts for r_isa and r_hypo relations
    r_isa_counts = Counter()
    r_hypo_counts = Counter()
    
    # Dictionaries to store new relation types
    r_ingredient_bundle_counts = Counter()
    r_common_kit_counts = Counter()
    
    # Dictionaries to group ingredients and utensils by filename
    ingredients_by_recipe = defaultdict(list)
    utensils_by_recipe = defaultdict(list)
    
    # Temporary storage to track entities by recipe for deduplication
    entities_in_recipe = defaultdict(set)
    
    # Read input CSV file
    print(f"Reading input file: {input_file}")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            
            # First pass: collect all entities by recipe to check for duplicates
            rows = []
            for row in reader:
                if len(row) < 3:
                    continue  # Skip malformed rows
                
                filename, entity, entity_type = row
                
                # Skip rows with malformed types
                if '.txt"' in entity_type or entity_type not in ["ingredient", "utensil"]:
                    continue
                
                # Apply normalization
                normalized_entity = normalize_entity(entity)
                
                # Store original row with normalized entity for further processing
                rows.append((filename, entity, normalized_entity, entity_type))
                
                # Track normalized entities by recipe
                entities_in_recipe[filename].add(normalized_entity.lower())
            
            # Second pass: process rows with deduplication logic
            for filename, original_entity, normalized_entity, entity_type in rows:
                # Skip if this is a misspelled version and the correct version exists in the recipe
                if (original_entity.lower() in ENTITY_CORRECTIONS and 
                    original_entity.lower() != normalized_entity.lower() and
                    normalized_entity.lower() in entities_in_recipe[filename] and
                    original_entity.lower() != normalized_entity.lower()):
                    continue
                
                # Add entity to the appropriate collection
                if entity_type == 'ingredient':
                    r_isa_counts[(normalized_entity, "ingrédient", "r_isa")] += 1
                    r_hypo_counts[("ingrédient", normalized_entity, "r_hypo")] += 1
                    ingredients_by_recipe[filename].append(normalized_entity)
                elif entity_type == 'utensil':
                    r_isa_counts[(normalized_entity, "ustensile", "r_isa")] += 1
                    r_hypo_counts[("ustensile", normalized_entity, "r_hypo")] += 1
                    utensils_by_recipe[filename].append(normalized_entity)
        
        # Generate r_ingredient_bundle relations
        for recipe, ingredients in ingredients_by_recipe.items():
            # Create relations between all pairs of ingredients
            for ing1, ing2 in permutations(ingredients, 2):
                r_ingredient_bundle_counts[(ing1, ing2, "r_ingredient_bundle")] += 1
        
        # Generate r_common_kit relations
        for recipe, utensils in utensils_by_recipe.items():
            # Create relations between all pairs of utensils
            for ut1, ut2 in permutations(utensils, 2):
                r_common_kit_counts[(ut1, ut2, "r_common_kit")] += 1
    
        # Write output CSV file
        print(f"Writing output to: {output_file}")
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['node1', 'node2', 'relation', 'w'])
            
            # Write r_isa relations
            for key, count in r_isa_counts.items():
                node1, node2, relation = key
                writer.writerow([node1, node2, relation, count])
                
            # Write r_hypo relations
            for key, count in r_hypo_counts.items():
                node1, node2, relation = key
                writer.writerow([node1, node2, relation, count])
            
            # Write r_ingredient_bundle relations
            for key, count in r_ingredient_bundle_counts.items():
                node1, node2, relation = key
                writer.writerow([node1, node2, relation, count])
                
            # Write r_common_kit relations
            for key, count in r_common_kit_counts.items():
                node1, node2, relation = key
                writer.writerow([node1, node2, relation, count])
            
        total_relations = (len(r_isa_counts) + len(r_hypo_counts) + 
                           len(r_ingredient_bundle_counts) + len(r_common_kit_counts))
        
        print(f"Processing complete. Generated {total_relations} relations.")
        print(f"- r_isa relations: {len(r_isa_counts)}")
        print(f"- r_hypo relations: {len(r_hypo_counts)}")
        print(f"- r_ingredient_bundle relations: {len(r_ingredient_bundle_counts)}")
        print(f"- r_common_kit relations: {len(r_common_kit_counts)}")
        
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    process_entities_to_relations()
