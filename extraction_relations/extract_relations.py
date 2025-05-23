#!/usr/bin/env python3
# extract_from_string.py - Extract semantic relations from an input string

import os
import sys
# Fix scipy/numpy compatibility issues at import time
try:
    import scipy
    import numpy as np
    if not hasattr(scipy.linalg, 'triu'):
        # Add triu implementation if it's missing
        from scipy import sparse
        def triu(m, k=0):
            if sparse.issparse(m):
                return sparse.triu(m, k)
            else:
                return np.triu(m, k)
        scipy.linalg.triu = triu
except ImportError as e:
    print(f"Error importing scipy/numpy: {e}")
    print("Try installing compatible versions: pip install scipy==1.10.1 numpy==1.24.3")
    sys.exit(1)

import spacy
import numpy as np
from gensim.models import KeyedVectors
from numpy.linalg import norm
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import requests
import json
import string
import argparse
import csv
import ast
from tqdm import tqdm
import logging

# Constants and configuration
BASE_URL = "https://jdm-api.demo.lirmm.fr/v0/relations"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_RELATIONS_TYPES = os.path.join(BASE_DIR, "../post_traitement/relations_types.json") # Liste des relations

# Load relation types
def load_relation_types():
    """
    Load relation types from JSON file
    """
    try:
        with open(PATH_RELATIONS_TYPES, 'r', encoding='utf-8') as f:
            relation_types = json.load(f)
            # Create a mapping from id to name
            # The API response uses 'type' field for the relation type number
            relation_map = {r['id']: r['name'] for r in relation_types}
            print(f"Successfully loaded {len(relation_map)} relation types")
            return relation_map
    except Exception as e:
        logging.error(f"Error loading relation types: {e}")
        return {}

# Setup logging
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)

# Load spaCy model
try:
    nlp = spacy.load("fr_core_news_lg")
    print("Successfully loaded the 'fr_core_news_lg' model")
except OSError:
    print("Error: The 'fr_core_news_lg' model is not installed.")
    print("Please install it with: python -m spacy download fr_core_news_lg")
    sys.exit(1)

# List of relations that are capable of providing interesting entities
INTERESTING_DEPS = {
    # verbal relations: subject → verb, verb → object
    "nsubj", "nsubj:pass", "obj", "iobj",
    # noun modifiers: noun → noun (prep-headed), noun → adjective
    "nmod", "obl", "amod",    # e.g. "recette de cuisine", "bouillon aromatique"
    # multi-word names & appositions
    "compound", "flat:name", "appos",
    # clausal modifiers (relative/participial clauses)
    "acl", "acl:relcl", "advcl",
}

# Words to create a vector representing gastronomy (our domain of interest)
pivots = [
    # domain & practice
    "cuisine", "gastronomie", "cuisinier",

    # recipe structure
    "recette", "ingrédient",

    # cooking techniques
    "cuisson", "mijoter", "bouillir", "rôtir", "griller",
    "sauter", "frire", "braiser", "cuire",

    # utensils & containers
    "marmite", "casserole", "poêle", "four", "ustensile",

    # spices & seasonings
    "épice", "condiment", "assaisonnement", "aromate", "bouquet_garni",

    # food categories
    "viande", "poisson", "légume", "fruit", "fruit_de_mer",
    "produit_laitier", "farine", "sucre", "sel", "poivre",

    # dish types
    "entrée", "plat_principal", "accompagnement", "dessert",
    "apéritif", "sauce",

    # nutrition & dietetics
    "nutrition", "calorie", "diététique"
]

# Similarity threshold with the "gastronomy" vector
THRESHOLD = 0.4

@dataclass
class TokenAnnotation:
    text: str           # surface form
    lemma: str          # canonical form
    pos: str            # part-of-speech tag
    dep: str            # dependency label
    head: int           # index of the head token in its sentence
    sent_id: Optional[int] = None  # optional: which sentence

def annotate(text: str) -> List[List[TokenAnnotation]]:
    """
    Annotate text with spaCy for linguistic analysis
    """
    doc = nlp(text)

    all_sents: List[List[TokenAnnotation]] = []
    for sent_id, sent in enumerate(doc.sents):
        tokens = []
        for token in sent:
            tokens.append(TokenAnnotation(
                text=token.text,
                lemma=token.lemma_,
                pos=token.pos_,
                dep=token.dep_,
                head=token.head.i - sent.start,  # index *within* this sentence
                sent_id=sent_id
            ))
        all_sents.append(tokens)
    return all_sents

def extract_pairs(tokens):
    """
    Extract interesting pairs based on dependency relations
    """
    pairs = []
    for idx, tok in enumerate(tokens):
        if tok.dep in INTERESTING_DEPS:
            head = tokens[tok.head]
            if tok.lemma in string.punctuation or head.lemma in string.punctuation:
                continue
            pairs.append((head.lemma, tok.lemma, tok.dep, tok.pos))
    return pairs

def load_word2vec_model(model_path='cc.fr.300.vec.gz', limit=50000):
    """
    Load the Word2Vec model
    """
    try:
        model_path = os.path.join(BASE_DIR, model_path)
        if not os.path.exists(model_path):
            print(f"Error: Word2Vec model not found at {model_path}")
            print("Please download it with: wget -c \"https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.vec.gz\"")
            return None
            
        print(f"Loading Word2Vec model from {model_path}...")
        model = KeyedVectors.load_word2vec_format(model_path, binary=False, limit=limit)
        print(f"Successfully loaded Word2Vec model with {len(model.key_to_index)} terms")
        return model
    except Exception as e:
        print(f"Error loading Word2Vec model: {e}")
        return None

def pair_vector(model, h, lemma):
    """Average the two word vectors."""
    return (model[h] + model[lemma]) / 2

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between vectors."""
    return float(a.dot(b) / (norm(a) * norm(b)))

def select_best_relation(relations, relation_types):
    """
    Sélectionne la meilleure relation à partir d'une liste de relations.
    Adapted from calcul_max_w.py
    
    Args:
        relations (list): Liste des dictionnaires de relations contenant les clés 'type' et 'w'
        relation_types (dict): Correspondance entre l'ID du type de relation et le nom de la relation
    
    Returns:
        tuple: (relation_name, weight)
    """
    if not relations or len(relations) == 0:
        return '', ''
    
    # S'il n'y a qu'une seule relation, on l'utilise
    if len(relations) == 1:
        rel = relations[0]
        rel_type = rel.get('type')
        weight = float(rel.get('w', 0))
        relation_name = relation_types.get(rel_type, f"unknown_{rel_type}")
        return relation_name, str(weight)
    
    # Définit un multiplicateur de priorité pour différents types de relations
    relation_priority = {
        1: 1.05,      # r_raff_sem - Raffinement sémantique vers un usage particulier du terme source
        6: 1.05,      # r_isa - Les relations taxonomiques sont importantes
        8: 1.05,      # r_hypo - Hyponymes
        9: 1.05,      # r_has_part - Relations partie-tout
        10: 1.05,     # r_holo - Holonymes
        16: 1.05,     # r_instr - Relation d'instrument
        17: 1.05,     # r_carac - Relations caractéristiques
        75: 1.05,     # r_accomp - Est souvent accompagné de
        15: 1.05,     # r_lieu - Relations de lieu
        53: 1.05,     # r_make - Productions
        67: 1.05      # r_similar - Similarités
    }
    
    # Priorité par défaut
    default_priority = 1.0
    
    # Calcule un score pour chaque relation
    best_score = float('-inf')
    best_type = None
    best_weight = 0
    
    for rel in relations:
        rel_type = rel.get('type')
        weight = float(rel.get('w', 0))
        
        # Obtient le multiplicateur de priorité pour ce type de relation
        priority = relation_priority.get(rel_type, default_priority)
        
        # Calcule le score basé sur le poids et la priorité du type de relation
        score = weight * priority
        
        if score > best_score:
            best_score = score
            best_type = rel_type
            best_weight = weight
    
    # Obtient le nom de la meilleure relation
    relation_name = relation_types.get(best_type, f"unknown_{best_type}")
    
    return relation_name, str(best_weight)

def process_text(text, model=None, relation_types=None):
    """
    Process input text and extract relations
    """
    # Annotate the text
    sents = annotate(text)
    
    # Extract pairs from sentences
    all_pairs = []
    for sent in sents:
        p = extract_pairs(sent)
        all_pairs.append(p)

    # Flatten all_pairs for easier processing and printing
    flat_pairs = []
    for pairs in all_pairs:
        flat_pairs.extend(pairs)
    
    print(f"Extracted {len(flat_pairs)} linguistic pairs")
    
    # If model is available, filter pairs by similarity to domain
    filtered = []
    
    if model is not None:
        # Compute domain centroid
        pivot_vecs = [model[p] for p in pivots if p in model]
        if not pivot_vecs:
            raise ValueError("None of your pivots were in the model!")
        domain_centroid = np.mean(pivot_vecs, axis=0)
        
        # Filter pairs based on similarity
        for h, lemma, dep, pos in flat_pairs:
            if h in model and lemma in model:
                vec = pair_vector(model, h, lemma)
                sim = cosine_sim(domain_centroid, vec)
                filtered.append((h, lemma, dep, pos, sim))
        
        print(f"Filtered to {len(filtered)} pairs with words in model")
    
    # Query JDM API for relations
    results = []
    
    if model is not None:
        pairs_to_query = [(h, lemma) for h, lemma, dep, pos, sim in filtered if sim >= THRESHOLD]
        print(f"Querying JDM API for {len(pairs_to_query)} pairs (similarity ≥ {THRESHOLD})")
    else:
        pairs_to_query = [(h, lemma) for h, lemma, _, _ in flat_pairs]
        print(f"Querying JDM API for all {len(pairs_to_query)} pairs (no filtering)")
    
    # Remove duplicates before querying API
    seen_pairs = set()
    unique_pairs = []
    for pair in pairs_to_query:
        if pair not in seen_pairs:
            unique_pairs.append(pair)
            seen_pairs.add(pair)
    
    # Query API for each unique pair
    pair_to_dep_pos_sim = {}
    if model is not None:
        # Create a lookup dictionary for dependency, POS, and similarity
        for h, lemma, dep, pos, sim in filtered:
            pair_to_dep_pos_sim[(h, lemma)] = (dep, pos, sim)
    else:
        # If no model, use just dependency and POS
        for h, lemma, dep, pos in flat_pairs:
            if (h, lemma) not in pair_to_dep_pos_sim:
                pair_to_dep_pos_sim[(h, lemma)] = (dep, pos, 0.0)
    
    all_pairs_with_info = []
    
    for h, lemma in tqdm(unique_pairs, desc="Querying JDM API"):
        try:
            resp = requests.get(f"{BASE_URL}/from/{h}/to/{lemma}")
            resp.raise_for_status()
            rels = resp.json().get("relations", [])
            
            # Get dependency, POS, and similarity
            dep, pos, sim = pair_to_dep_pos_sim.get((h, lemma), ('', '', 0.0))
            
            # Add to results regardless of whether there are relations or not
            pair_info = {
                "head": h,
                "lemma": lemma,
                "dep": dep,
                "pos": pos,
                "sim": sim,
                "relations": rels
            }
            all_pairs_with_info.append(pair_info)
            
            # Also add to the relations results if there are relations
            if rels:
                results.append({
                    "node1": h,
                    "node2": lemma,
                    "relations": rels,
                })
        except Exception as e:
            logging.error(f"Error processing pair ({h}, {lemma}): {e}")
    
    return results, all_pairs_with_info

def main():
    global THRESHOLD
    
    parser = argparse.ArgumentParser(description='Extract semantic relations from an input text.')
    parser.add_argument('--text', '-t', type=str, help='Text to process')
    parser.add_argument('--file', '-f', type=str, help='File containing text to process')
    parser.add_argument('--no-model', action='store_true', help='Skip Word2Vec filtering (faster but less accurate)')
    parser.add_argument('--threshold', type=float, default=THRESHOLD, help=f'Similarity threshold (default: {THRESHOLD})')
    parser.add_argument('--output', '-o', type=str, help='Output CSV file path')
    parser.add_argument('--output-dir', '-d', default='output', type=str, help='Output directory for CSV files (filename derived from input file)')
    
    args = parser.parse_args()
    
    # Get input text
    input_file_name = None
    if args.text:
        input_text = args.text
        input_file_name = "input_text"
    elif args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                input_text = f.read()
            input_file_name = os.path.basename(args.file).replace('.txt', '')
        except Exception as e:
            print(f"Error reading file: {e}")
            return
    else:
        print("Please provide input text with --text or --file")
        parser.print_help()
        return
    
    # Determine output file path
    output_file = None
    if args.output:
        output_file = args.output
    elif args.output_dir:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        output_file = os.path.join(args.output_dir, f"{input_file_name}.csv")
    else:
        print("No output specified. Results will only be printed to console.")
        print("Use --output or --output-dir to save results to a CSV file.")

    # Load relation types
    relation_types = load_relation_types()
    
    # Load model if needed
    model = None
    if not args.no_model:
        model = load_word2vec_model()
        if model is None:
            print("Warning: Running without Word2Vec filtering")

    # Update threshold if specified
    if args.threshold != THRESHOLD:
        THRESHOLD = args.threshold
    
    # Process text
    results, all_pairs_with_info = process_text(input_text, model, relation_types)
    
    # Apply the calcul_max_w logic to select the best relation for each pair
    for pair in all_pairs_with_info:
        relations = pair.get('relations', [])
        relation_name, weight = select_best_relation(relations, relation_types)
        pair['relation_name'] = relation_name
        pair['w'] = weight
    
    # Write results to CSV file if output is specified
    if output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8', newline='') as csvfile:
            # Write the CSV data
            fieldnames = ['head', 'lemma', 'dep', 'pos', 'sim', 'relations', 'relation_name', 'w']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for pair in all_pairs_with_info:
                writer.writerow({
                    'head': pair['head'],
                    'lemma': pair['lemma'],
                    'dep': pair['dep'],
                    'pos': pair['pos'],
                    'sim': pair['sim'],
                    'relations': json.dumps(pair['relations']),
                    'relation_name': pair['relation_name'],
                    'w': pair['w']
                })
        
        print(f"\nRelations saved to {output_file}")
    
    # Print results
    if results:
        print("\n===== SEMANTIC RELATIONS =====")
        print(f"Found {len(results)} pairs with relations out of {len(all_pairs_with_info)} total pairs")
        
        # Print only the best relation for each pair (limited to 10 to avoid excessive output)
        pairs_with_relations = [p for p in all_pairs_with_info if p.get('relations')]
        for i, pair in enumerate(pairs_with_relations[:10]):
            head = pair['head']
            lemma = pair['lemma']
            relation_name = pair['relation_name']
            weight = pair['w']
            sim = pair['sim']
            
            if relation_name:  # Only print if a relation was found
                print(f"{i+1}. {head} → {lemma}: {relation_name} (weight: {weight}, sim: {sim:.4f})")
        
        if len(results) > 10:
            print(f"\n... and {len(results) - 10} more pairs with relations")
    else:
        print("\nNo semantic relations found.")

if __name__ == "__main__":
    main()
