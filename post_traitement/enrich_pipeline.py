from transformers import pipeline
import os
import csv
import json
import ast
import pandas as pd
from tqdm import tqdm
import sys

# Initialize the zero-shot classification model
classifier = pipeline("zero-shot-classification",
                      model="joeddav/xlm-roberta-large-xnli")

# Define the relation priority map
relation_priority = {
    0: 1.2,      # r_associated - Concepts associés qui ont une priorité plus élevée
    1: 1.2,      # r_raff_sem - Raffinement sémantique vers un usage particulier du terme source
    6: 1.2,      # r_isa - Relations taxonomiques qui sont importantes ("est un")
    8: 1.2,      # r_hypo - Il est demandé d'énumérer des SPECIFIQUES/hyponymes du terme. Par exemple, 'mouche', 'abeille', 'guêpe' pour 'insecte'
    9: 1.2,      # r_has_part - Relations partie-tout qui sont importantes ("a pour partie")
    10: 1.2,     # r_holo - Il est démandé d'énumérer des 'TOUT' (a pour holonymes) de l'objet en question. Pour 'main', on aura 'bras', 'corps', 'personne', etc... Le tout est aussi l'ensemble comme 'classe' pour 'élève'.
    16: 1.2,     # r_instr - L'instrument est l'objet avec lequel on fait l'action. Dans - Il mange sa salade avec une fourchette -, fourchette est l'instrument. Des instruments typiques de 'tuer' peuvent être 'arme', 'pistolet', 'poison', ... (couper r_instr couteau)
    17: 1.2,     # r_carac - Relations caractéristiques qui sont importantes
    75: 1.2,     # r_accomp - Est souvent accompagné de, se trouve avec... Par exemple : Astérix et Obelix, le pain et le fromage, les fraises et la chantilly.
    
    # Medium priority relations
    15: 1.1,     # r_lieu - Relations de lieu qui ont une priorité moyenne
    53: 1.1,     # r_make - Que peut PRODUIRE le terme ? (par exemple abeille -> miel, usine -> voiture, agriculteur -> blé, moteur -> gaz carbonique ...)
    67: 1.1      # r_similar - Similaire/ressemble à ; par exemple le congre est similaire à une anguille, ...
}

# Load relation types to get descriptions and names
def load_relation_types(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        if content.startswith('//'):
            content = content[content.find('\n') + 1:]
        relation_types = json.loads(content)
    
    relation_dict = {}
    for relation in relation_types:
        if isinstance(relation, dict) and 'id' in relation and 'name' in relation:
            relation_dict[relation['id']] = {
                'name': relation['name'],
                'help': relation.get('help', ''),
                'gpname': relation.get('gpname', '')
            }
    
    return relation_dict

# Function to classify a pair of words and determine their relation
def classify_word_pair(word1, word2, relation_types=None):
    """
    Classify the semantic relation between two words using the zero-shot classification model.
    
    Args:
        word1 (str): The first word (source)
        word2 (str): The second word (target)
        relation_types (dict, optional): Dictionary of relation types. If None, will be loaded from file.
        
    Returns:
        tuple: (relation_id, relation_name, relation_description, confidence_score)
    """
    # If relation_types is not provided, load it
    if relation_types is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        relation_types_path = os.path.join(script_dir, 'relations_types.json')
        relation_types = load_relation_types(relation_types_path)

    # Create a description of the pair for classification
    sentence = f"Relation sémantique entre '{word1}' et '{word2}'"
    
    # Create candidate labels for classification from all priority relations
    candidate_relations = []
    relation_ids = []
    descriptions = []
    
    # Include all relations with descriptions from the priority map
    for rel_id in relation_priority.keys():
        if rel_id in relation_types:
            relation = relation_types[rel_id]
            label = f"{relation['name']} ({relation.get('gpname', '')})"
            description = relation.get('help', '')
            if description:
                candidate_relations.append(label)
                relation_ids.append(rel_id)
                descriptions.append(description)        # Run the classification if we have candidates
    if candidate_relations:
        try:
            result = classifier(sentence, candidate_relations)
            
            # Get the top 3 most probable relations
            top_labels = result['labels'][:3]
            top_scores = result['scores'][:3]
            
            # Get the details for the top relations
            top_relations = []
            for i, label in enumerate(top_labels):
                idx = candidate_relations.index(label)
                rel_id = relation_ids[idx]
                rel_name = relation_types[rel_id]['name']
                rel_desc = descriptions[idx]
                conf_score = top_scores[i]
                
                # Add relation to results
                top_relations.append({
                    'id': rel_id,
                    'name': rel_name,
                    'description': rel_desc,
                    'confidence': round(conf_score * 100, 1),
                    'passes_threshold': conf_score >= 0.7  # Mark if the relation passes the 70% threshold
                })
            
            return top_relations
        
        except Exception as e:
            print(f"Classification error: {e}")
            return []
    else:
        print("No candidate relations with descriptions were found.")
        return []

# Function to predict relation type using zero-shot classification
def predict_relation(head, lemma, dep, pos, relation_types):
    # Create a description of the pair
    sentence = f"Relation sémantique entre '{head}' et '{lemma}' ({dep}, {pos})"
    
    # Select most relevant relation types based on POS and dependency
    candidate_relations = []
    relation_ids = []
    
    # For verb -> noun with obj dependency, consider patient, instrument, location relations
    if pos == 'NOUN' and dep == 'obj' and head.endswith(('er', 'ir', 're')):
        relevant_types = [14, 16, 15, 53]  # r_patient, r_instr, r_lieu, r_make
    # For noun -> adj with amod dependency, consider characteristic relations
    elif pos == 'ADJ' and dep == 'amod':
        relevant_types = [17, 67]  # r_carac, r_similar
    # For noun -> noun with nmod dependency, consider part-whole, taxonomic relations
    elif pos == 'NOUN' and dep == 'nmod':
        relevant_types = [9, 10, 6, 8]  # r_has_part, r_holo, r_isa, r_hypo
    # Default: consider associated, characteristic, similar relations
    else:
        relevant_types = [0, 17, 67]  # r_associated, r_carac, r_similar
    
    # Add high priority relations that should be considered in all cases
    relevant_types = list(set(relevant_types + [0, 1, 6]))  # Always consider r_associated, r_raff_sem, r_isa
    
    # Create candidate labels for classification
    for rel_id in relevant_types:
        if rel_id in relation_types:
            relation = relation_types[rel_id]
            label = f"{relation['name']} ({relation['gpname']})"
            description = relation.get('help', '')
            if description:
                candidate_relations.append(label)
                relation_ids.append(rel_id)
    
    # If we don't have enough candidate relations, add some defaults
    if len(candidate_relations) < 3:
        for rel_id in relation_priority.keys():
            if rel_id in relation_types and rel_id not in relevant_types:
                relation = relation_types[rel_id]
                label = f"{relation['name']} ({relation['gpname']})"
                candidate_relations.append(label)
                relation_ids.append(rel_id)
                if len(candidate_relations) >= 5:
                    break
    
    # Run the classification
    if candidate_relations:
        try:
            result = classifier(sentence, candidate_relations)
            
            # Get the most probable relation
            top_label = result['labels'][0]
            top_score = result['scores'][0]
            
            # Find the relation ID
            idx = candidate_relations.index(top_label)
            rel_id = relation_ids[idx]
            
            # Return the predicted relation
            return rel_id, relation_types[rel_id]['name'], top_score
        except Exception as e:
            print(f"Classification error: {e}")
            return None, None, 0.0
    else:
        return None, None, 0.0

def get_classification_template(word1, word2, pos1=None, pos2=None):
    """Generate specific templates in French based on word types"""
    
    # Default template in French
    template = f"Déterminer la relation sémantique entre '{word1}' et '{word2}'."
    
    # For noun -> noun relationships (taxonomic, part-whole)
    if pos1 == 'NOUN' and pos2 == 'NOUN':
        template = f"Déterminer la relation sémantique entre les noms '{word1}' et '{word2}'. " \
                  f"Considérer si l'un est un type de l'autre, une partie de l'autre, ou communément associé."
    
    # For verb -> noun relationships (patient, instrument)
    elif pos1 == 'VERB' and pos2 == 'NOUN':
        template = f"Déterminer comment le nom '{word2}' se rapporte à l'action '{word1}'. " \
                  f"Est-ce qu'il subit l'action? Est-ce un instrument utilisé pour l'action? Est-ce un lieu ou un résultat?"
    
    return template

def process_csv_file(input_path, output_path, relation_types):
    print(f"Processing file: {os.path.basename(input_path)}")
    
    # Load the CSV data
    df = pd.read_csv(input_path)
    
    # Initialize counters
    total_pairs = len(df)
    no_relation_pairs = 0
    enriched_pairs = 0
    skipped_pairs = 0  # Track pairs skipped due to low confidence
    
    # Process each row
    for i, row in tqdm(df.iterrows(), total=total_pairs):
        # Check if there's no relation
        try:
            relations = ast.literal_eval(row['relations']) if row['relations'] and row['relations'] != '[]' else []
        except:
            relations = []
        
        if not relations:
            no_relation_pairs += 1
            
            # Predict relation
            rel_id, rel_name, score = predict_relation(
                row['head'], row['lemma'], row['dep'], row['pos'], relation_types
            )
            
            # Only save relations with confidence score above 70%
            confidence_threshold = 0.7  # 70% confidence threshold
            if rel_id is not None and score >= confidence_threshold:
                # Create a new synthetic relation
                new_relation = {
                    "id": f"predicted_{i}",
                    "node1": -1,  # Placeholder
                    "node2": -1,  # Placeholder
                    "type": rel_id,
                    "w": round(score * 100, 1)  # Convert score to a weight similar to the original data
                }
                
                # Update the row
                df.at[i, 'relations'] = str([new_relation])
                df.at[i, 'relation_name'] = rel_name
                df.at[i, 'w'] = new_relation['w']
                enriched_pairs += 1
                
                # Print detailed information about the enriched pair
                print(f"Enriched: '{row['head']}' -> '{row['lemma']}' ({rel_name}) with confidence {round(score * 100, 1)}%")
            elif rel_id is not None:
                print(f"Skipped relation with low confidence ({round(score * 100, 1)}%): {row['head']} -> {row['lemma']} ({rel_name})")
                skipped_pairs += 1
    
    # Save the enriched data
    df.to_csv(output_path, index=False)
    
    print(f"File processed: {os.path.basename(input_path)}")
    print(f"Total pairs: {total_pairs}")
    print(f"Pairs without relations: {no_relation_pairs}")
    print(f"Pairs below confidence threshold (70%): {skipped_pairs}")
    print(f"Enriched pairs: {enriched_pairs}")
    
    return no_relation_pairs, enriched_pairs, skipped_pairs

def main():
    # Define file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_data_dir = os.path.join(script_dir, 'test_data', 'wikipedia')
    output_dir = os.path.join(script_dir, 'test_output', 'wikipedia')
    relation_types_path = os.path.join(script_dir, 'relations_types.json')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load relation types
    relation_types = load_relation_types(relation_types_path)
    print(f"Loaded {len(relation_types)} relation types")
    
    # Process all CSV files in the test_data directory
    csv_files = [f for f in os.listdir(test_data_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {test_data_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Process each CSV file
    total_files_processed = 0
    total_no_relation_pairs = 0
    total_enriched_pairs = 0
    total_skipped_pairs = 0
    
    for csv_file in csv_files:
        input_path = os.path.join(test_data_dir, csv_file)
        output_path = os.path.join(output_dir, f"enriched_{csv_file}")
        
        try:
            no_relation, enriched, skipped = process_csv_file(input_path, output_path, relation_types)
            total_no_relation_pairs += no_relation
            total_enriched_pairs += enriched
            total_skipped_pairs += skipped
            total_files_processed += 1
            print(f"Processed {csv_file}")
            print("=" * 50)
        except Exception as e:
            print(f"Error processing file {csv_file}: {e}")
    
    print(f"Processing complete!")
    print(f"Total files processed: {total_files_processed}")
    print(f"Total pairs without relations: {total_no_relation_pairs}")
    print(f"Total pairs below confidence threshold (70%): {total_skipped_pairs}")
    print(f"Total pairs enriched: {total_enriched_pairs}")
    print(f"Acceptance rate: {round(total_enriched_pairs / (total_enriched_pairs + total_skipped_pairs) * 100, 1)}% of predicted relations")
    print(f"Output files saved to: {output_dir}")

# Additional CLI for word pair classification
if __name__ == "__main__":
    # Check if being run in word pair classification mode
    if len(sys.argv) == 3 and sys.argv[1] != "--file":
        word1 = sys.argv[1]
        word2 = sys.argv[2]
        
        print(f"\nClassifying relation between '{word1}' and '{word2}'...")
        
        # Load relation types
        script_dir = os.path.dirname(os.path.abspath(__file__))
        relation_types_path = os.path.join(script_dir, 'relations_types.json')
        relation_types = load_relation_types(relation_types_path)
        
        # Classify the word pair
        results = classify_word_pair(word1, word2, relation_types)
        
        if results:
            print("\nPredicted relations:")
            print("-" * 100)
            print(f"{'Relation':<15} {'Confidence':<10} {'Status':<15} {'Description'}")
            print("-" * 100)
            
            for result in results:
                status = "✓ Accepted" if result['passes_threshold'] else "✗ Rejected"
                print(f"{result['name']:<15} {str(result['confidence']) + '%':<10} {status:<15} {result['description'][:60]}...")
            
            accepted_relations = [r for r in results if r['passes_threshold']]
            print("\nSummary:")
            print(f"Total relations found: {len(results)}")
            print(f"Relations above 70% threshold: {len(accepted_relations)}")
            
            if accepted_relations:
                print(f"\nBest relation: {accepted_relations[0]['name']} ({accepted_relations[0]['confidence']}%)")
            else:
                print("\nNo relations meet the confidence threshold of 70%.")
        else:
            print("Could not classify the relation between these words.")
    
    # Regular file processing mode
    elif len(sys.argv) > 1 and sys.argv[1] == "--file":
        # Process specific file if provided
        if len(sys.argv) > 2:
            # Logic for processing a specific file
            script_dir = os.path.dirname(os.path.abspath(__file__))
            test_data_dir = os.path.join(script_dir, 'test_data', 'wikipedia')
            output_dir = os.path.join(script_dir, 'test_output', 'wikipedia')
            relation_types_path = os.path.join(script_dir, 'relations_types.json')
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Load relation types
            relation_types = load_relation_types(relation_types_path)
            
            file_name = sys.argv[2]
            input_path = os.path.join(test_data_dir, file_name)
            output_path = os.path.join(output_dir, f"enriched_{file_name}")
            
            if os.path.exists(input_path):
                try:
                    no_relation, enriched, skipped = process_csv_file(input_path, output_path, relation_types)
                    print(f"Processing complete for {file_name}!")
                    print(f"Pairs without relations: {no_relation}")
                    print(f"Pairs below confidence threshold (70%): {skipped}")
                    print(f"Pairs enriched: {enriched}")
                    if enriched + skipped > 0:
                        print(f"Acceptance rate: {round(enriched / (enriched + skipped) * 100, 1)}% of predicted relations")
                except Exception as e:
                    print(f"Error processing file {file_name}: {e}")
            else:
                print(f"File not found: {input_path}")
        else:
            print("Please provide a file name after --file")
    else:
        # Run the normal main function for batch processing
        main()

