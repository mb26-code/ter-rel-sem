import pandas as pd
import re

def process_entities():
    # Load the CSV file
    df = pd.read_csv('recipe_entities_final.csv')
    
    # Create a new list to hold the processed entries
    processed_entries = []
    
    # Pattern to match for "et" and other common separators
    pattern = r'\s+et\s+|\s*,\s*'
    
    # Process each row
    for _, row in df.iterrows():
        filename = row['filename']
        entity = row['entity']
        entity_type = row['type']
        
        # Skip entities containing ® symbol
        if '®' in entity:
            continue
            
        # Check if entity contains separators
        if re.search(pattern, entity):
            # Split the entity
            splits = re.split(pattern, entity)
            
            # Add each split as a separate entry
            for split in splits:
                if split.strip():  # Ensure we don't add empty strings
                    # Skip splits containing ® symbol
                    if '®' not in split:
                        processed_entries.append({
                            'filename': filename,
                            'entity': split.strip(),
                            'type': entity_type
                        })
        else:
            # Keep the original entry if no separator
            processed_entries.append({
                'filename': filename,
                'entity': entity,
                'type': entity_type
            })
    
    # Create a new DataFrame with the processed entries
    processed_df = pd.DataFrame(processed_entries)
    
    # Optional: Remove duplicates (same filename, entity, and type)
    processed_df = processed_df.drop_duplicates()
    
    # Save to a new CSV file
    processed_df.to_csv('recipe_entities_refined.csv', index=False)
    
    print(f"Processing complete. Found and processed {len(processed_df) - len(df)} additional entities.")

if __name__ == "__main__":
    process_entities()