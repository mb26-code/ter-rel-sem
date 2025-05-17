#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Process CSV files from ressources_completes directory and generate a single CSV file
that can be imported into a PostgreSQL table.

This script:
1. Traverses all CSV files in both marmiton and wikipedia subdirectories
2. Maps CSV column names to database column names:
   - head → node1
   - lemma → node2
   - dep → dep (unchanged)
   - pos → pos (unchanged)
   - sim → sim (unchanged)
   - relations → relations (unchanged)
   - relation_name → best_relation
   - w → best_relation_w
3. Skips duplicate entries (based on node1, node2, and dep combination)
4. Outputs a single CSV file ready for import into the PostgreSQL table

RESULTS:
Total files processed: 7441
Total entries extracted: 104536
Total duplicates skipped: 163522
"""

import os
import csv
import json
from collections import OrderedDict

# Define paths
BASE_DIR = "/Users/qmacstore/Development/python/ter-rel-sem/post_traitement"
RESOURCES_DIR = os.path.join(BASE_DIR, "ressources_completes")
MARMITON_DIR = os.path.join(RESOURCES_DIR, "marmiton")
WIKIPEDIA_DIR = os.path.join(RESOURCES_DIR, "wikipedia")
OUTPUT_FILE = os.path.join(BASE_DIR, "semantic_relations_import.csv")

# Define column mapping
COLUMN_MAPPING = {
    "head": "node1",
    "lemma": "node2",
    "dep": "dep",
    "pos": "pos",
    "sim": "sim",
    "relations": "relations",
    "relation_name": "best_relation",
    "w": "best_relation_w"
}

# Database column order
DB_COLUMNS = ["node1", "node2", "dep", "pos", "sim", "relations", "best_relation", "best_relation_w"]

def process_csv_files():
    """Process all CSV files and generate a single output CSV file."""
    processed_entries = set()  # Track unique entries to avoid duplicates
    total_files = 0
    total_entries = 0
    skipped_duplicates = 0

    # Open output file for writing
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=DB_COLUMNS)
        writer.writeheader()

        # Process Marmiton files
        for root, _, files in os.walk(MARMITON_DIR):
            for filename in files:
                if filename.endswith(".csv"):
                    file_path = os.path.join(root, filename)
                    entries, duplicates = process_file(file_path, writer, processed_entries)
                    total_files += 1
                    total_entries += entries
                    skipped_duplicates += duplicates
                    
                    # Print progress every 100 files
                    if total_files % 100 == 0:
                        print(f"Processed {total_files} files, {total_entries} entries, skipped {skipped_duplicates} duplicates")

        # Process Wikipedia files
        for root, _, files in os.walk(WIKIPEDIA_DIR):
            for filename in files:
                if filename.endswith(".csv"):
                    file_path = os.path.join(root, filename)
                    entries, duplicates = process_file(file_path, writer, processed_entries)
                    total_files += 1
                    total_entries += entries
                    skipped_duplicates += duplicates
                    
                    # Print progress every 100 files
                    if total_files % 100 == 0:
                        print(f"Processed {total_files} files, {total_entries} entries, skipped {skipped_duplicates} duplicates")

    print(f"Processing complete!")
    print(f"Total files processed: {total_files}")
    print(f"Total entries extracted: {total_entries}")
    print(f"Total duplicates skipped: {skipped_duplicates}")
    print(f"Output file: {OUTPUT_FILE}")


def process_file(file_path, writer, processed_entries):
    """Process a single CSV file and write its contents to the output CSV."""
    entries_count = 0
    duplicates_count = 0

    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as infile:
            # Skip the first line if it contains the file path comment
            first_line = infile.readline()
            if first_line.startswith('//'):
                # It's a comment, open the file again to reset the reader
                infile.close()
                infile = open(file_path, 'r', newline='', encoding='utf-8')
            else:
                # It wasn't a comment, seek back to the beginning
                infile.seek(0)
                
            reader = csv.DictReader(infile)
            
            for row in reader:
                # Map columns to database names
                db_row = {}
                for csv_col, db_col in COLUMN_MAPPING.items():
                    if csv_col in row:
                        db_row[db_col] = row[csv_col]
                    else:
                        db_row[db_col] = ""  # Use empty string for missing columns
                
                # Ensure relations is a valid JSON array
                if db_row['relations'] and db_row['relations'] != "[]":
                    try:
                        json_data = json.loads(db_row['relations'])
                        # Ensure it's properly formatted as JSON string
                        db_row['relations'] = json.dumps(json_data)
                    except json.JSONDecodeError:
                        # If not valid JSON, set to empty array
                        db_row['relations'] = "[]"
                else:
                    db_row['relations'] = "[]"
                
                # Create a unique key for deduplication (node1, node2, dep)
                entry_key = (db_row['node1'], db_row['node2'], db_row['dep'])
                
                # Skip if we've already seen this combination
                if entry_key in processed_entries:
                    duplicates_count += 1
                    continue
                
                # Add to our set of processed entries
                processed_entries.add(entry_key)
                
                # Convert values to appropriate types
                if db_row['sim']:
                    try:
                        db_row['sim'] = float(db_row['sim'])
                    except ValueError:
                        db_row['sim'] = None
                
                if db_row['best_relation_w']:
                    try:
                        db_row['best_relation_w'] = float(db_row['best_relation_w'])
                    except ValueError:
                        db_row['best_relation_w'] = None
                
                # Write to output file
                writer.writerow(db_row)
                entries_count += 1
                
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return 0, 0
    
    return entries_count, duplicates_count


if __name__ == "__main__":
    process_csv_files()
