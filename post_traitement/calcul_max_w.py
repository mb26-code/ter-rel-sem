#!/usr/bin/env python3
import json
import csv
import ast
import os
import sys

# Ce fichier est pour rajouter la colonne `relation` et `w`
# en calculant leurs `w`

def select_best_relation(relations, relation_name_map):
    """
    Select the best relation from a list of relations using a more sophisticated approach.
    
    This function implements multiple strategies for relation selection:
    1. If there are no relations, return empty values
    2. If there's only one relation, use that one
    3. If there are multiple relations, use a weighted approach that considers:
       - The weight of the relation
       - The priority of certain relation types (predefined priority map)
    
    Args:
        relations (list): List of relation dictionaries containing 'type' and 'w' keys
        relation_name_map (dict): Mapping from relation type ID to relation name
    
    Returns:
        tuple: (relation_name, weight) of the selected relation
    """
    if not relations or len(relations) == 0:
        return '', ''
    
    # If there's only one relation, use it
    if len(relations) == 1:
        rel = relations[0]
        rel_type = rel.get('type')
        weight = float(rel.get('w', 0))
        relation_name = relation_name_map.get(rel_type, f"unknown_{rel_type}")
        return relation_name, str(weight)
    
    # Define a priority multiplier for different relation types
    # Higher values mean higher priority
    relation_priority = {
        # High priority relations (examples)
        0: 1.2,      # r_associated - Associated concepts have higher priority
        1: 1.2,      # r_raff_sem - Raffinement sémantique vers un usage particulier du terme source
        6: 1.2,      # r_isa - Taxonomic relations are important
        8: 1.2,      # r_hypo - Il est demandé d'énumérer des SPECIFIQUES/hyponymes du terme. Par exemple, 'mouche', 'abeille', 'guêpe' pour 'insecte'
        9: 1.2,      # r_has_part - Part-whole relations are important
        10: 1.2,     # r_holo - Il est démandé d'énumérer des 'TOUT' (a pour holonymes)  de l'objet en question. Pour 'main', on aura 'bras', 'corps', 'personne', etc... Le tout est aussi l'ensemble comme 'classe' pour 'élève'.
        16: 1.2,     # r_instr - L'instrument est l'objet avec lequel on fait l'action. Dans - Il mange sa salade avec une fourchette -, fourchette est l'instrument. Des instruments typiques de 'tuer' peuvent être 'arme', 'pistolet', 'poison', ... (couper r_instr couteau)
        17: 1.2,     # r_carac - Characteristic relations are important
        75: 1.2,     # r_accomp - Est souvent accompagné de, se trouve avec... Par exemple : Astérix et Obelix, le pain et le fromage, les fraises et la chantilly.
        
        # Medium priority relations
        15: 1.1,     # r_lieu - Location relations have medium priority
        53: 1.1,     # r_make - Que peut PRODUIRE le terme ? (par exemple abeille -> miel, usine -> voiture, agriculteur -> blé,  moteur -> gaz carbonique ...)
        67: 1.1      # r_similar - Similaire/ressemble à ; par exemple le congre est similaire à une anguille, ...
    
        
    }
    
    # Default priority for relation types not explicitly listed
    default_priority = 1.0
    
    # Calculate a score for each relation that considers both weight and type priority
    best_score = float('-inf')
    best_type = None
    best_weight = 0
    
    # Also track the highest raw weight for comparison
    max_raw_weight = float('-inf')
    max_raw_type = None
    
    for rel in relations:
        rel_type = rel.get('type')
        weight = float(rel.get('w', 0))
        
        # Track the relation with the highest raw weight
        if weight > max_raw_weight:
            max_raw_weight = weight
            max_raw_type = rel_type
        
        # Get priority multiplier for this relation type
        priority = relation_priority.get(rel_type, default_priority)
        
        # Calculate score based on weight and relation type priority
        score = weight * priority
        
        if score > best_score:
            best_score = score
            best_type = rel_type
            best_weight = weight
    
    # Print info when the selected relation is different from the one with highest raw weight
    if best_type != max_raw_type:
        max_raw_name = relation_name_map.get(max_raw_type, f"unknown_{max_raw_type}")
        best_name = relation_name_map.get(best_type, f"unknown_{best_type}")
        print(f"Priority changed selection: {max_raw_name} (w={max_raw_weight}) -> {best_name} (w={best_weight}, score={best_score})")
    
    # Get the relation name
    relation_name = relation_name_map.get(best_type, f"unknown_{best_type}")
    
    return relation_name, str(best_weight)

def process_csv(input_csv_path, output_csv_path, relation_name_map):
    """Process a single CSV file and add relation columns"""
    print(f"Processing: {input_csv_path}")
    print(f"Output to: {output_csv_path}")
    
    # Process the CSV file
    with open(input_csv_path, 'r', encoding='utf-8') as infile, \
         open(output_csv_path, 'w', encoding='utf-8', newline='') as outfile:
        
        # Skip the comment line if it exists
        first_line = infile.readline().strip()
        if first_line.startswith('//'):
            # Read the real header next
            csv_reader = csv.reader(infile)
            header = next(csv_reader)
        else:
            # The first line was the header
            csv_reader = csv.reader([first_line])
            header = next(csv_reader)
            # Continue reading from the beginning
            infile.seek(0)
            next(infile)  # Skip header
            csv_reader = csv.reader(infile)
        
        # Create CSV writer with the new columns
        csv_writer = csv.writer(outfile)
        new_header = header + ['relation_name', 'w']
        csv_writer.writerow(new_header)
        
        # Process each row
        rows_processed = 0
        for row in csv_reader:
            rows_processed += 1
            if len(row) >= 6:  # Make sure we have enough columns
                relation_str = row[5]
                
                try:
                    # Parse the relation JSON string
                    relations = ast.literal_eval(relation_str)
                    
                    if not relations or len(relations) == 0:
                        # No relations, leave the fields blank
                        new_row = row + ['', '']
                        csv_writer.writerow(new_row)
                    else:
                        # Use the select_best_relation function
                        relation_name, max_weight = select_best_relation(relations, relation_name_map)
                        
                        # Add the new columns
                        new_row = row + [relation_name, max_weight]
                        csv_writer.writerow(new_row)
                    
                except (ValueError, SyntaxError) as e:
                    print(f"Error processing row {rows_processed} in {os.path.basename(input_csv_path)}: {e}")
                    # Write the row without the additional columns
                    csv_writer.writerow(row + ['error', 'error'])
            else:
                print(f"Row {rows_processed} in {os.path.basename(input_csv_path)} doesn't have enough columns: {row}")
                csv_writer.writerow(row + ['', ''])
        
        return rows_processed
    
def main():
    # Define file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    relations_types_path = os.path.join(script_dir, 'relations_types.json')
    
    # Define resources directory and output directory
    resources_dir = os.path.join(script_dir, 'ressources')
    output_dir = os.path.join(script_dir, 'ressources_completes/wikipedia')
    
    print(f"Script directory: {script_dir}")
    print(f"Loading relation types from: {relations_types_path}")
    print(f"Resources directory: {resources_dir}")
    print(f"Output directory: {output_dir}")
    
    # Check if resources directory exists
    if not os.path.exists(resources_dir):
        print(f"ERROR: Resources directory not found: {resources_dir}")
        return
        
    # Check if relations_types file exists
    if not os.path.exists(relations_types_path):
        print(f"ERROR: Relations types file not found: {relations_types_path}")
        return
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Load relation types mapping
    with open(relations_types_path, 'r', encoding='utf-8') as f:
        # Skip the first line if it starts with //
        content = f.read()
        if content.startswith('//'):
            content = content[content.find('\n') + 1:]
        relation_types = json.loads(content)
    
    # Create a mapping from type ID to name
    relation_name_map = {}
    for relation in relation_types:
        if isinstance(relation, dict) and 'id' in relation and 'name' in relation:
            relation_name_map[relation['id']] = relation['name']
    
    print(f"Loaded {len(relation_name_map)} relation types")
    
    # Get all CSV files from resources directory
    csv_files = [f for f in os.listdir(resources_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {resources_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Process each CSV file
    total_files_processed = 0
    total_rows_processed = 0
    
    for csv_file in csv_files:
        input_path = os.path.join(resources_dir, csv_file)
        output_path = os.path.join(output_dir, f"{csv_file}")
        
        try:
            rows_processed = process_csv(input_path, output_path, relation_name_map)
            total_rows_processed += rows_processed
            total_files_processed += 1
            print(f"Processed {rows_processed} rows in {csv_file}")
        except Exception as e:
            print(f"Error processing file {csv_file}: {e}")
    
    print(f"Processing complete!")
    print(f"Total files processed: {total_files_processed}")
    print(f"Total rows processed: {total_rows_processed}")
    print(f"Output files saved to: {output_dir}")

if __name__ == "__main__":
    main()
