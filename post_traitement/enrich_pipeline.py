import pandas as pd
import json
import time
import os
import datetime
import concurrent.futures
import threading
try:
    # Try to import the new recommended package first
    from langchain_ollama import OllamaLLM as Ollama
except ImportError:
    # Fall back to the deprecated version if the new package is not installed
    from langchain_community.llms import Ollama

# List of relations we care about with descriptions
RELATIONS = {
    "r_associated": "Les concepts associés ont une priorité plus élevée",
    "r_raff_sem": "Raffinement sémantique vers un usage particulier du terme source",
    "r_isa": "Les relations taxonomiques sont importantes",
    "r_hypo": "Il est demandé d'énumérer des SPECIFIQUES/hyponymes du terme. Par exemple, 'mouche', 'abeille', 'guêpe' pour 'insecte'",
    "r_has_part": "Les relations partie-tout sont importantes",
    "r_holo": "Il est démandé d'énumérer des 'TOUT' (a pour holonymes) de l'objet en question. Pour 'main', on aura 'bras', 'corps', 'personne', etc... Le tout est aussi l'ensemble comme 'classe' pour 'élève'",
    "r_instr": "L'instrument est l'objet avec lequel on fait l'action. Dans - Il mange sa salade avec une fourchette -, fourchette est l'instrument",
    "r_carac": "Les relations caractéristiques sont importantes",
    "r_accomp": "Est souvent accompagné de, se trouve avec... Par exemple : Astérix et Obelix, le pain et le fromage, les fraises et la chantilly",
    "r_lieu": "Les relations de lieu ont une priorité moyenne",
    "r_make": "Que peut PRODUIRE le terme ? (par exemple abeille -> miel, usine -> voiture, agriculteur -> blé, moteur -> gaz carbonique ...)",
    "r_similar": "Similaire/ressemble à ; par exemple le congre est similaire à une anguille, ..."
}

# Mapping from relation name to type ID based on your code
RELATION_TYPE_MAP = {
    "r_associated": 0,
    "r_raff_sem": 1,
    "r_isa": 6,
    "r_hypo": 8,
    "r_has_part": 9,
    "r_holo": 10,
    "r_instr": 16,
    "r_carac": 17,
    "r_lieu": 15,
    "r_make": 53,
    "r_similar": 67,
    "r_accomp": 75
}

# Thread-local storage to create one model instance per thread
thread_local = threading.local()

def get_thread_model():
    """Get a thread-specific model instance"""
    if not hasattr(thread_local, "model"):
        # For Ollama, we need to use the model name as registered in Ollama
        # Using the model "llama3-relations" that's already registered in Ollama
        thread_local.model = Ollama(
            model="llama3:instruct", # llama3-relations
            # Configure for better deterministic outputs and to handle specific responses
            # temperature=0.1,  # Lower temperature for more deterministic outputs
        )
    return thread_local.model

def get_relation_from_model(head, lemma, pos, dependency, context="cuisine"):
    """Ask the model for a relation between two words."""
    prompt = f"""
    Determine le type de relation sémantique entre deux mots dans un contexte de {context}:
    
    Mot 1: {head}
    Mot 2: {lemma}(POS: {pos})
    Relation syntaxique: {dependency}
    
    Choisir UNE relation parmi les suivantes qui décrit le mieux comment ces mots sont liés:
    
    - r_associated: Concepts généralement associés ensemble
    - r_raff_sem: Raffinement sémantique vers un usage particulier du terme source
    - r_isa: Relations taxonomiques (X est un Y)
    - r_hypo: Terme spécifique (hyponyme) du premier mot (X est un type de Y)
    - r_has_part: Relation partie-tout (X a Y comme partie)
    - r_holo: Le tout dont le premier mot fait partie (X fait partie de Y)
    - r_instr: Relation d'instrument (X est utilisé pour faire Y)
    - r_carac: Relation de caractéristique (X a la caractéristique Y)
    - r_accomp: Souvent accompagné de (X est souvent avec Y)
    - r_lieu: Relation de lieu (X se trouve à/dans Y)
    - r_make: X produit Y
    - r_similar: X est similaire à Y
    
    Réponds uniquement avec le nom de la relation (par exemple "r_carac") sans explication. Si aucune relation ne s'applique, réponds "none".
    """
    
    # Get thread-local model
    local_model = get_thread_model()
    
    # Call the model
    try:
        response = local_model.invoke(prompt).strip().lower()
        
        # Extract relation from potential JSON response
        # Look specifically for relation patterns to avoid parsing JSON incorrectly
        for rel in RELATIONS:
            if rel.lower() in response:
                return rel
                
        if "none" in response.lower():
            return None
            
        return None  # Default if no relation is detected
    except Exception as e:
        print(f"Error querying model: {e}")
        return None

def get_relations_batch(pairs, context="cuisine", batch_size=5):
    """Process a batch of word pairs at once to improve efficiency."""
    # Split pairs into smaller batches to avoid context length limits
    all_relations = []
    
    for i in range(0, len(pairs), batch_size):
        current_batch = pairs[i:i+batch_size]
        
        prompt = f"""
        Determine les types de relations sémantiques entre les paires de mots suivantes dans un contexte de {context}.
        
        Pour chaque paire, choisis UNE relation parmi les suivantes qui décrit le mieux comment ces mots sont liés:
        
        - r_associated: Concepts généralement associés ensemble
        - r_raff_sem: Raffinement sémantique vers un usage particulier du terme source
        - r_isa: Relations taxonomiques (X est un Y)
        - r_hypo: Terme spécifique (hyponyme) du premier mot (X est un type de Y)
        - r_has_part: Relation partie-tout (X a Y comme partie)
        - r_holo: Le tout dont le premier mot fait partie (X fait partie de Y)
        - r_instr: Relation d'instrument (X est utilisé pour faire Y)
        - r_carac: Relation de caractéristique (X a la caractéristique Y)
        - r_accomp: Souvent accompagné de (X est souvent avec Y)
        - r_lieu: Relation de lieu (X se trouve à/dans Y)
        - r_make: X produit Y
        - r_similar: X est similaire à Y
        
        Voici les paires de mots:
        """
        
        # Add each pair to the prompt
        for j, (head, lemma, pos, dep) in enumerate(current_batch):
            prompt += f"\nPaire {j+1}: Mot 1: {head}, Mot 2: {lemma} (POS: {pos}), Relation syntaxique: {dep}"
        
        prompt += "\n\nPour chaque paire (dans l'ordre), réponds uniquement avec le nom de la relation (exemple: \"r_carac\") ou \"none\" si aucune relation ne s'applique. Réponds avec une relation par ligne, sans numérotation ni explication."
        
        try:
            # Use thread-local model
            local_model = get_thread_model()
            response = local_model.invoke(prompt).strip()
            
            # Process response line by line
            relations = []
            
            # Split the response into lines and clean them
            lines = [line.strip().lower() for line in response.split('\n') if line.strip()]
            
            for line in lines:
                if not line:
                    continue
                    
                # Check if line contains a relation
                found_relation = None
                for rel in RELATIONS:
                    if rel.lower() in line:
                        found_relation = rel
                        break
                
                relations.append(found_relation)
                
                # If we've collected enough relations for this batch, stop
                if len(relations) == len(current_batch):
                    break
            
            # If we didn't get enough relations, pad with None
            while len(relations) < len(current_batch):
                relations.append(None)
                
            all_relations.extend(relations)
            
        except Exception as e:
            print(f"Error querying model for batch: {e}")
            # If an error occurs, return None for each pair in the batch
            all_relations.extend([None] * len(current_batch))
        
        # Add a small delay between batches
        time.sleep(1)
        
    return all_relations

def save_enhancement_log(enhancements_list, dataset_name):
    """Save the enhancement log to a CSV file.
    
    Args:
        enhancements_list: List of dictionaries with enhancement data
        dataset_name: Name of the dataset (e.g. 'marmiton' or 'wikipedia')
    """
    if not enhancements_list:
        print("No enhancements to log")
        return
    
    # Create log dataframe
    log_df = pd.DataFrame(enhancements_list)
    
    # Add timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save to CSV
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(script_dir, f"enhance_log_{dataset_name}.csv")
    
    log_df.to_csv(log_path, index=False)
    print(f"Enhancement log saved to: {log_path}")
    
    # Print summary of relations added
    relation_counts = log_df['relation_name'].value_counts()
    print("\nSummary of relations added:")
    for rel, count in relation_counts.items():
        print(f"  {rel}: {count}")

def process_file(file_path, output_path=None, context="cuisine", batch_size=5, enhancement_list=None, sim_threshold=0.55, skip_existing=True):
    """Process a CSV file to enhance relations using batch processing."""
    if output_path is None:
        output_path = file_path.replace(".csv", "_enhanced.csv")
    
    # Check if output file already exists and skip_existing is True
    if skip_existing and os.path.exists(output_path):
        print(f"Skipping file (already processed): {file_path}")
        return 0, True  # Return 0 enhancements and True for skipped
    
    print(f"Processing file: {file_path}")
    
    try:
        # Load the data
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return 0, False  # Return 0 enhancements and False for skipped
    
    # Track changes
    enhancements_count = 0
    
    # Process rows with empty relations and sim > threshold
    empty_relations = df[(df['relation'] == '[]') & (df['sim'] > sim_threshold)]
    total_to_enhance = len(empty_relations)
    
    print(f"Found {total_to_enhance} pairs with empty relations and similarity > {sim_threshold}")
    
    if total_to_enhance == 0:
        print("No pairs to enhance in this file.")
        return 0
    
    # Prepare pairs for batch processing
    pairs = []
    indices = []
    
    for idx in empty_relations.index:
        row = df.loc[idx]
        pairs.append((row['head'], row['lemma'], row['pos'], row['dep']))
        indices.append(idx)
    
    # Process pairs in batches
    print(f"Processing {len(pairs)} pairs in batches of {batch_size}...")
    batch_count = (len(pairs) + batch_size - 1) // batch_size
    print(f"Total batches: {batch_count}")
    
    # Start time for batch processing
    batch_start_time = time.time()
    
    relations = get_relations_batch(pairs, context, batch_size)
    
    # Calculate batch processing time
    batch_time = time.time() - batch_start_time
    if batch_count > 0:
        per_batch_time = batch_time/batch_count
        per_pair_time = batch_time/len(pairs)
        print(f"Batch processing finished in {batch_time:.1f}s:")
        print(f"- {per_batch_time:.2f}s per batch of {batch_size} pairs")
        print(f"- {per_pair_time:.2f}s per word pair")
    
    # Get filename for logging
    filename = os.path.basename(file_path)
    
    # Update dataframe with results
    for i, (idx, relation) in enumerate(zip(indices, relations)):
        if relation:
            # Get type ID for this relation
            type_id = RELATION_TYPE_MAP.get(relation, 0)
            
            # Create new relation object
            new_relation = [{
                "id": 52000 + idx,  # ID starting from 52000
                "node1": 0,  # Placeholder
                "node2": 0,  # Placeholder
                "type": type_id,
                "w": 0.0    # Default weight set to 0.0
            }]
            
            # Update the row
            df.at[idx, 'relation'] = json.dumps(new_relation)
            df.at[idx, 'relation_name'] = relation
            df.at[idx, 'w'] = 0.0
            enhancements_count += 1
            
            # Add to enhancement log
            if enhancement_list is not None:
                enhancement_list.append({
                    'file': filename,
                    'head': df.at[idx, 'head'],
                    'lemma': df.at[idx, 'lemma'],
                    'pos': df.at[idx, 'pos'],
                    'dep': df.at[idx, 'dep'],
                    'sim': df.at[idx, 'sim'],
                    'relation_name': relation,
                    'relation_type': type_id,
                    'weight': 0.0
                })
            
            # Log progress 
            if enhancements_count % 10 == 0:
                print(f"Enhanced {enhancements_count}/{total_to_enhance} relations...")
    
    # Save the enhanced data
    df.to_csv(output_path, index=False)
    print(f"Enhanced {enhancements_count} relations. Saved to {output_path}")
    return enhancements_count, False  # Return enhancements count and False for not skipped

def process_directory(directory_path, context="cuisine", batch_size=5, workers=1, sim_threshold=0.55, skip_existing=True):
    """Process all CSV files in a directory."""
    if workers <= 1:
        # If only one worker, use the sequential version
        return process_directory_sequential(directory_path, context, batch_size, sim_threshold, skip_existing)
    else:
        # Use parallel processing with multiple workers
        return process_directory_parallel(directory_path, context, batch_size, workers, sim_threshold, skip_existing)

def process_directory_sequential(directory_path, context="cuisine", batch_size=5, sim_threshold=0.55, skip_existing=True):
    """Process all CSV files in a directory sequentially."""
    if not os.path.exists(directory_path):
        print(f"Directory not found: {directory_path}")
        return
    
    # Determine if we're processing marmiton or wikipedia based on the path
    dataset_name = None
    if "marmiton" in directory_path.lower():
        dataset_name = "marmiton"
    elif "wikipedia" in directory_path.lower():
        dataset_name = "wikipedia"
    else:
        # Use the directory name if we can't determine
        dataset_name = os.path.basename(directory_path)
    
    # Create output directory in ressources_enriched with the corresponding subfolder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_output_dir = os.path.join(script_dir, "ressources_enriched")
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
        
    output_dir = os.path.join(base_output_dir, dataset_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Output directory set to: {output_dir}")
    
    # Get all CSV files
    csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
    print(f"Found {len(csv_files)} CSV files in {directory_path}")
    
    # List to collect all enhancements
    enhancement_list = []
    
    # For progress tracking
    start_time = time.time()
    total_files = len(csv_files)
    total_enhancements = 0
    skipped_files = 0
    
    for i, file_name in enumerate(csv_files):
        input_path = os.path.join(directory_path, file_name)
        output_path = os.path.join(output_dir, file_name)
        
        print(f"Processing file {i+1}/{total_files}: {file_name}")
        
        enhancements, was_skipped = process_file(input_path, output_path, context, batch_size, enhancement_list, sim_threshold, skip_existing)
        total_enhancements += enhancements
        if was_skipped:
            skipped_files += 1
        
        # Calculate and show progress
        elapsed_time = time.time() - start_time
        files_completed = i + 1
        files_remaining = total_files - files_completed
        
        if files_completed > 0:
            avg_time_per_file = elapsed_time / files_completed
            est_time_remaining = avg_time_per_file * files_remaining
            
            # Convert to minutes and seconds
            elapsed_min, elapsed_sec = divmod(int(elapsed_time), 60)
            remaining_min, remaining_sec = divmod(int(est_time_remaining), 60)
            
            print(f"Progress: {files_completed}/{total_files} files ({(files_completed/total_files)*100:.1f}%)")
            print(f"Files skipped: {skipped_files}, Files processed: {files_completed - skipped_files}")
            print(f"Total enhancements so far: {total_enhancements}")
            print(f"Time elapsed: {elapsed_min}m {elapsed_sec}s, Estimated time remaining: {remaining_min}m {remaining_sec}s")
            print("-" * 50)
    
    # Save enhancement log
    if enhancement_list:
        save_enhancement_log(enhancement_list, dataset_name)
    
    # Calculate total time
    total_time = time.time() - start_time
    print_processing_summary(total_files, total_enhancements, total_time, workers=1, skipped_files=skipped_files)
    
    return total_enhancements, enhancement_list, skipped_files

def process_directory_parallel(directory_path, context="cuisine", batch_size=5, max_workers=4, sim_threshold=0.55, skip_existing=True):
    """Process all CSV files in a directory using multiple threads."""
    if not os.path.exists(directory_path):
        print(f"Directory not found: {directory_path}")
        return
    
    # Determine if we're processing marmiton or wikipedia based on the path
    dataset_name = None
    if "marmiton" in directory_path.lower():
        dataset_name = "marmiton"
    elif "wikipedia" in directory_path.lower():
        dataset_name = "wikipedia"
    else:
        # Use the directory name if we can't determine
        dataset_name = os.path.basename(directory_path)
    
    # Create output directory in ressources_enriched with the corresponding subfolder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_output_dir = os.path.join(script_dir, "ressources_enriched")
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
        
    output_dir = os.path.join(base_output_dir, dataset_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Output directory set to: {output_dir}")
    
    # Get all CSV files
    csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
    print(f"Found {len(csv_files)} CSV files in {directory_path}")
    
    # Thread-safe list for enhancements
    enhancement_list = []
    enhancement_lock = threading.Lock()
    
    # For progress tracking
    start_time = time.time()
    total_files = len(csv_files)
    
    # Counter for completed files and total enhancements
    completed_files = 0
    total_enhancements = 0
    counter_lock = threading.Lock()
    
    # Add a counter for skipped files
    skipped_files = 0
    
    def process_file_with_tracking(input_path, output_path):
        nonlocal completed_files, total_enhancements, skipped_files
        
        # Process the file
        local_enhancements, was_skipped = process_file(input_path, output_path, context, batch_size, None, sim_threshold, skip_existing)
        
        # If there were enhancements, collect them separately to avoid thread conflicts
        if local_enhancements > 0:
            # Re-open the file and extract the enhanced pairs for the log
            try:
                df = pd.read_csv(output_path)
                enhanced_rows = df[df['relation_name'].notna()]
                
                filename = os.path.basename(input_path)
                
                local_enhancements_list = []
                for _, row in enhanced_rows.iterrows():
                    if row['relation_name'] in RELATION_TYPE_MAP:
                        type_id = RELATION_TYPE_MAP[row['relation_name']]
                        local_enhancements_list.append({
                            'file': filename,
                            'head': row['head'],
                            'lemma': row['lemma'],
                            'pos': row['pos'],
                            'dep': row['dep'],
                            'sim': row['sim'],
                            'relation_name': row['relation_name'],
                            'relation_type': type_id,
                            'weight': row['w']
                        })
                
                # Add to the global enhancement list with lock protection
                with enhancement_lock:
                    enhancement_list.extend(local_enhancements_list)
            except Exception as e:
                print(f"Error collecting enhancements from {output_path}: {e}")
        
        # Update counters with lock protection
        with counter_lock:
            nonlocal completed_files, total_enhancements, skipped_files
            completed_files += 1
            total_enhancements += local_enhancements
            if was_skipped:
                skipped_files += 1
            
            # Calculate and show progress
            elapsed_time = time.time() - start_time
            files_remaining = total_files - completed_files
            
            if completed_files > 0:
                avg_time_per_file = elapsed_time / completed_files
                est_time_remaining = avg_time_per_file * files_remaining
                
                # Convert to minutes and seconds
                elapsed_min, elapsed_sec = divmod(int(elapsed_time), 60)
                remaining_min, remaining_sec = divmod(int(est_time_remaining), 60)
                
                print(f"Progress: {completed_files}/{total_files} files ({completed_files/total_files*100:.1f}%)")
                print(f"Files skipped: {skipped_files}, Files processed: {completed_files - skipped_files}")
                print(f"Total enhancements so far: {total_enhancements}")
                print(f"Time elapsed: {elapsed_min}m {elapsed_sec}s, Estimated time remaining: {remaining_min}m {remaining_sec}s")
                print(f"Active workers: {min(max_workers, total_files - completed_files)}")
                print("-" * 50)
    
    # Process files in parallel
    print(f"Starting parallel processing with {max_workers} worker threads...")
    print(f"Using similarity threshold: {sim_threshold}, batch size: {batch_size}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for file_name in csv_files:
            input_path = os.path.join(directory_path, file_name)
            output_path = os.path.join(output_dir, file_name)
            futures.append(executor.submit(process_file_with_tracking, input_path, output_path))
        
        # Wait for all tasks to complete
        concurrent.futures.wait(futures)
    
    # Save enhancement log
    if enhancement_list:
        save_enhancement_log(enhancement_list, dataset_name)
    
    # Calculate total time
    total_time = time.time() - start_time
    print_processing_summary(total_files, total_enhancements, total_time, workers=max_workers, skipped_files=skipped_files)
    
    return total_enhancements, enhancement_list, skipped_files

def print_processing_summary(total_files, total_enhancements, total_time, workers=1, skipped_files=0):
    """Print a summary of the processing with performance metrics."""
    # Calculate total time
    total_min, total_sec = divmod(int(total_time), 60)
    
    # Calculate files per second (only counting actually processed files)
    processed_files = total_files - skipped_files
    files_per_sec = processed_files / total_time if total_time > 0 and processed_files > 0 else 0
    
    print("\n" + "="*60)
    print(f"PROCESSING SUMMARY")
    print("="*60)
    print(f"Worker threads:         {workers}")
    print(f"Total files:            {total_files}")
    print(f"Files skipped:          {skipped_files}")
    print(f"Files processed:        {processed_files}")
    print(f"Total enhancements:     {total_enhancements}")
    print(f"Total processing time:  {total_min}m {total_sec}s")
    print(f"Files per second:       {files_per_sec:.3f}")
    if workers > 1:
        print(f"Parallelization gain:   {files_per_sec * 1.0:.1f}x (vs. estimated sequential)")
    print("="*60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhance semantic relations in CSV files using Mistral AI')
    parser.add_argument('--file', type=str, help='Path to a single CSV file to enhance')
    parser.add_argument('--dir', type=str, help='Path to directory containing CSV files')
    parser.add_argument('--context', type=str, default='gastronomie', help='Context domain (default: gastronomie)')
    parser.add_argument('--batch', type=int, default=5, help='Number of word pairs to process in each batch (default: 5)')
    parser.add_argument('--workers', type=int, default=1, help='Number of worker threads for parallel processing (default: 1)')
    parser.add_argument('--sim-threshold', type=float, default=0.55, help='Similarity threshold for word pairs (default: 0.55)')
    parser.add_argument('--no-skip', action='store_false', dest='skip_existing', help='Process all files even if they already exist in the output directory')
    parser.set_defaults(skip_existing=True)
    
    args = parser.parse_args()
    
    if args.file:
        process_file(args.file, context=args.context, batch_size=args.batch, sim_threshold=args.sim_threshold, skip_existing=args.skip_existing)
    elif args.dir:
        process_directory(args.dir, context=args.context, batch_size=args.batch, workers=args.workers, sim_threshold=args.sim_threshold, skip_existing=args.skip_existing)
    else:
        # Default example usage
        file_path = "/Users/qmacstore/Development/python/ter-rel-sem/post_traitement/ressources_completes/marmiton/Accras de morue_merged.csv"
        print(f"No file or directory specified. Using example file: {file_path}")
        process_file(file_path, context='gastronomie', batch_size=5, sim_threshold=0.55, skip_existing=args.skip_existing)