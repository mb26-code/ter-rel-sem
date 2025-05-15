#!/usr/bin/env python3
import json
import csv
import ast
import os
import sys

def main():
    # Define file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    relations_types_path = os.path.join(script_dir, 'relations_types.json')
    test_csv_path = os.path.join(script_dir, 'test.csv')
    output_csv_path = os.path.join(script_dir, 'test_with_relations.csv')
    
    print(f"Script directory: {script_dir}")
    print(f"Loading relation types from: {relations_types_path}")
    print(f"Input CSV path: {test_csv_path}")
    print(f"Output CSV path: {output_csv_path}")
    
    # Check if files exist
    if not os.path.exists(relations_types_path):
        print(f"ERROR: Relations types file not found: {relations_types_path}")
        return
    if not os.path.exists(test_csv_path):
        print(f"ERROR: Test CSV file not found: {test_csv_path}")
        return
    
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
    
    # Process the CSV file
    with open(test_csv_path, 'r', encoding='utf-8') as infile, \
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
                    
                    # Find the relation with the highest weight
                    max_weight = float('-inf')
                    max_type = None
                    
                    for rel in relations:
                        weight = float(rel.get('w', 0))
                        if weight > max_weight:
                            max_weight = weight
                            max_type = rel.get('type')
                    
                    # Get the relation name
                    relation_name = relation_name_map.get(max_type, f"unknown_{max_type}")
                    
                    # Add the new columns
                    new_row = row + [relation_name, str(max_weight)]
                    csv_writer.writerow(new_row)
                    
                except (ValueError, SyntaxError) as e:
                    print(f"Error processing row {rows_processed}: {e}")
                    # Write the row without the additional columns
                    csv_writer.writerow(row + ['error', 'error'])
            else:
                print(f"Row {rows_processed} doesn't have enough columns: {row}")
                csv_writer.writerow(row + ['', ''])
    
    print(f"Processed {rows_processed} rows")
    print(f"New CSV file created: {output_csv_path}")

if __name__ == "__main__":
    main()
