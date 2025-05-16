import pandas as pd
import os

def merge_data():
    # Define directories with absolute paths for reliability
    base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    
    # Extract directory names to variables for easy modification
    SOURCE_DIR = "marmiton"  # Change this to target different datasets
    pairs_subdir = "pairs"
    relations_subdir = "relations"
    output_subdir = "ressources"
    output_parent_dir = "post_traitement"
    
    # Build the full paths using the variables
    pairs_dir = os.path.join(base_dir, "output", SOURCE_DIR, pairs_subdir)
    relations_dir = os.path.join(base_dir, "output", SOURCE_DIR, relations_subdir)
    output_dir = os.path.join(base_dir, output_parent_dir, output_subdir, SOURCE_DIR)
    
    print(f"Base directory: {base_dir}")
    print(f"Dataset directory: {SOURCE_DIR}")
    print(f"Looking for pairs in: {pairs_dir}")
    print(f"Looking for relations in: {relations_dir}")
    print(f"Output will be saved to: {output_dir}")
    
    # Check if directories exist
    if not os.path.exists(pairs_dir):
        print(f"Error: Pairs directory does not exist: {pairs_dir}")
        return
    if not os.path.exists(relations_dir):
        print(f"Error: Relations directory does not exist: {relations_dir}")
        return
        
    # Check if directories contain any TXT files
    pairs_files = [f for f in os.listdir(pairs_dir) if f.endswith('.txt')]
    if not pairs_files:
        print(f"Error: No TXT files found in pairs directory: {pairs_dir}")
        return
    else:
        print(f"Found {len(pairs_files)} TXT files in pairs directory")
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through files in the pairs directory
    for pair_filename in os.listdir(pairs_dir):
        if pair_filename.endswith(".txt"):
            base_name = pair_filename.replace("_pairs.txt", "")
            
            relations_filename = f"{base_name}_relations.csv"
            
            pair_filepath = os.path.join(pairs_dir, pair_filename)
            relations_filepath = os.path.join(relations_dir, relations_filename)

            if not os.path.exists(relations_filepath):
                print(f"Warning: Corresponding relations file not found for {pair_filename}")
                continue

            try:
                # Read the pairs file (.txt but in CSV format)
                with open(pair_filepath, 'r', encoding='utf-8') as file:
                    if file.readline().startswith('//'):  # Skip comment line if present
                        pairs_content = file.read()
                    else:
                        file.seek(0)  # Go back to start of file
                        pairs_content = file.read()
                
                # Parse the pairs file content as CSV
                import io
                pairs_df = pd.read_csv(io.StringIO(pairs_content), names=['head', 'lemma', 'dep', 'pos', 'sim'] if 'head' not in pairs_content.splitlines()[0] else None)
                
                # Read the relations file (standard CSV)
                relations_df = pd.read_csv(relations_filepath)

                # Rename columns in relations_df for merging
                relations_df.rename(columns={'node1': 'head', 'node2': 'lemma'}, inplace=True)
                
                # Merge the dataframes
                # Using an inner merge to only keep rows where keys are found in both dataframes
                merged_df = pd.merge(pairs_df, relations_df, on=['head', 'lemma'], how='inner')
                
                # Select and reorder columns
                output_columns = ['head', 'lemma', 'dep', 'pos', 'sim', 'relation'] 
                # Ensure 'relation' column exists, it might be 'relations' from the relations_df
                if 'relations' in merged_df.columns and 'relation' not in merged_df.columns:
                    merged_df.rename(columns={'relations': 'relation'}, inplace=True)
                
                # Filter for existing columns to avoid errors if a column is missing
                final_columns = [col for col in output_columns if col in merged_df.columns]
                merged_df = merged_df[final_columns]

                # Define the output file path
                output_filename = f"{base_name}_merged.csv"
                output_filepath = os.path.join(output_dir, output_filename)

                # Save the merged dataframe to a new CSV
                merged_df.to_csv(output_filepath, index=False)
                print(f"Successfully merged and saved to {output_filepath}")

            except Exception as e:
                print(f"Error processing files {pair_filename} and {relations_filename}: {e}")

if __name__ == "__main__":
    merge_data()