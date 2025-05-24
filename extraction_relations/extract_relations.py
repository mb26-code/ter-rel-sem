#!/usr/bin/env python3
# extract_from_string.py - Extrait les relations sémantiques d'une chaîne d'entrée

import os
import sys
import warnings
warnings.filterwarnings('ignore', module='urllib3')
import os
from supabase import create_client, Client

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

print("\n===== EXTRACTION DES RELATIONS SÉMANTIQUES =====")

# Corrige les problèmes de compatibilité scipy/numpy au moment de l'importation
try:
    import scipy
    import numpy as np
    if not hasattr(scipy.linalg, 'triu'):
        # Ajoute l'implémentation de triu si elle est manquante
        from scipy import sparse
        def triu(m, k=0):
            if sparse.issparse(m):
                return sparse.triu(m, k)
            else:
                return np.triu(m, k)
        scipy.linalg.triu = triu
except ImportError as e:
    print(f"Erreur lors de l'importation de scipy/numpy : {e}")
    print("Essayez d'installer des versions compatibles : pip install scipy==1.10.1 numpy==1.24.3")
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

# Constantes et configuration
BASE_URL = "https://jdm-api.demo.lirmm.fr/v0/relations"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_RELATIONS_TYPES = os.path.join(BASE_DIR, "../post_traitement/relations_types.json") # Liste des relations

# Charger les types de relations
def load_relation_types():
    """
    Charger les types de relations depuis un fichier JSON
    """
    try:
        with open(PATH_RELATIONS_TYPES, 'r', encoding='utf-8') as f:
            relation_types = json.load(f)
            # Créer une correspondance de l'id vers le nom
            # La réponse de l'API utilise le champ 'type' pour le numéro du type de relation
            relation_map = {r['id']: r['name'] for r in relation_types}
            print(f"Types de relations chargés avec succès : {len(relation_map)}")
            return relation_map
    except Exception as e:
        logging.error(f"Erreur lors du chargement des types de relations : {e}")
        return {}

# Configuration du logging
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)

# Charger le modèle spaCy
try:
    nlp = spacy.load("fr_core_news_lg")
    print("Modèle 'fr_core_news_lg' chargé avec succès")
except OSError:
    print("Erreur : Le modèle 'fr_core_news_lg' n'est pas installé.")
    print("Veuillez l'installer avec : python -m spacy download fr_core_news_lg")
    sys.exit(1)

# Liste des relations capables de fournir des entités intéressantes
INTERESTING_DEPS = {
    # relations verbales : sujet → verbe, verbe → objet
    "nsubj", "nsubj:pass", "obj", "iobj",
    # modificateurs de noms : nom → nom (dirigé par une préposition), nom → adjectif
    "nmod", "obl", "amod",    # ex. "recette de cuisine", "bouillon aromatique"
    # noms composés et appositions
    "compound", "flat:name", "appos",
    # modificateurs clausaux (propositions relatives/participiales)
    "acl", "acl:relcl", "advcl",
}

# Mots pour créer un vecteur représentant la gastronomie (notre domaine d'intérêt)
pivots = [
    # domaine et pratique
    "cuisine", "gastronomie", "cuisinier",

    # structure de recette
    "recette", "ingrédient",

    # techniques de cuisson
    "cuisson", "mijoter", "bouillir", "rôtir", "griller",
    "sauter", "frire", "braiser", "cuire",

    # ustensiles et contenants
    "marmite", "casserole", "poêle", "four", "ustensile",

    # épices et assaisonnements
    "épice", "condiment", "assaisonnement", "aromate", "bouquet_garni",

    # catégories d'aliments
    "viande", "poisson", "légume", "fruit", "fruit_de_mer",
    "produit_laitier", "farine", "sucre", "sel", "poivre",

    # types de plats
    "entrée", "plat_principal", "accompagnement", "dessert",
    "apéritif", "sauce",

    # nutrition et diététique
    "nutrition", "calorie", "diététique"
]

# Seuil de similarité avec le vecteur "gastronomie"
THRESHOLD = 0.5

# Limite de mots pour le modèle Word2Vec
LIMIT_WORD2VEC = 50000  # Limite pour éviter de charger trop de mots

@dataclass
class TokenAnnotation:
    text: str           # forme de surface
    lemma: str          # forme canonique
    pos: str            # étiquette morpho-syntaxique
    dep: str            # étiquette de dépendance
    head: int           # index du token tête dans sa phrase
    sent_id: Optional[int] = None  # optionnel : quelle phrase

def annotate(text: str) -> List[List[TokenAnnotation]]:
    """
    Annoter le texte avec spaCy pour l'analyse linguistique
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
                head=token.head.i - sent.start,  # index *dans* cette phrase
                sent_id=sent_id
            ))
        all_sents.append(tokens)
    return all_sents

def extract_pairs(tokens):
    """
    Extraire les paires intéressantes basées sur les relations de dépendance
    """
    pairs = []
    for idx, tok in enumerate(tokens):
        if tok.dep in INTERESTING_DEPS:
            head = tokens[tok.head]
            if tok.lemma in string.punctuation or head.lemma in string.punctuation:
                continue
            pairs.append((head.lemma, tok.lemma, tok.dep, tok.pos))
    return pairs

def load_word2vec_model(model_path='cc.fr.300.vec.gz', limit=LIMIT_WORD2VEC):
    """
    Charger le modèle Word2Vec
    """
    try:
        model_path = os.path.join(BASE_DIR, model_path)
        if not os.path.exists(model_path):
            print(f"Erreur : Modèle Word2Vec non trouvé à {model_path}")
            print("Veuillez le télécharger avec : wget -c \"https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.vec.gz\"")
            return None
            
        print(f"Chargement du modèle Word2Vec depuis {model_path}...")
        model = KeyedVectors.load_word2vec_format(model_path, binary=False, limit=limit)
        print(f"Modèle Word2Vec chargé avec succès avec {len(model.key_to_index)} termes")
        return model
    except Exception as e:
        print(f"Erreur lors du chargement du modèle Word2Vec : {e}")
        return None

def pair_vector(model, h, lemma):
    """Moyenne des deux vecteurs de mots."""
    return (model[h] + model[lemma]) / 2

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Calculer la similarité cosinus entre vecteurs."""
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
    Traiter le texte d'entrée et extraire les relations
    """
    # Annoter le texte
    sents = annotate(text)
    
    # Extraire les paires des phrases
    all_pairs = []
    for sent in sents:
        p = extract_pairs(sent)
        all_pairs.append(p)

    # Aplatir all_pairs pour un traitement et un affichage plus faciles
    flat_pairs = []
    for pairs in all_pairs:
        flat_pairs.extend(pairs)
    
    print(f"Extrait {len(flat_pairs)} paires linguistiques")
    
    # Si le modèle est disponible, filtrer les paires par similarité au domaine
    filtered = []
    
    if model is not None:
        # Calculer le centroïde du domaine
        pivot_vecs = [model[p] for p in pivots if p in model]
        if not pivot_vecs:
            raise ValueError("Aucun de vos pivots n'était dans le modèle !")
        domain_centroid = np.mean(pivot_vecs, axis=0)
        
        # Filtrer les paires basées sur la similarité
        for h, lemma, dep, pos in flat_pairs:
            if h in model and lemma in model:
                vec = pair_vector(model, h, lemma)
                sim = cosine_sim(domain_centroid, vec)
                filtered.append((h, lemma, dep, pos, sim))
        
        print(f"Filtré à {len(filtered)} paires avec des mots dans le modèle")
    
    # Interroger l'API JDM pour les relations
    results = []
    
    if model is not None:
        pairs_to_query = [(h, lemma) for h, lemma, dep, pos, sim in filtered if sim >= THRESHOLD]
        print(f"Interrogation de l'API JDM pour {len(pairs_to_query)} paires (similarité ≥ {THRESHOLD})")
    else:
        pairs_to_query = [(h, lemma) for h, lemma, _, _ in flat_pairs]
        print(f"Interrogation de l'API JDM pour toutes les {len(pairs_to_query)} paires (pas de filtrage)")
    
    # Supprimer les doublons avant d'interroger l'API
    seen_pairs = set()
    unique_pairs = []
    for pair in pairs_to_query:
        if pair not in seen_pairs:
            unique_pairs.append(pair)
            seen_pairs.add(pair)
    
    # Interroger l'API pour chaque paire unique
    pair_to_dep_pos_sim = {}
    if model is not None:
        # Créer un dictionnaire de recherche pour la dépendance, POS et similarité
        for h, lemma, dep, pos, sim in filtered:
            pair_to_dep_pos_sim[(h, lemma)] = (dep, pos, sim)
    else:
        # Si pas de modèle, utiliser seulement la dépendance et POS
        for h, lemma, dep, pos in flat_pairs:
            if (h, lemma) not in pair_to_dep_pos_sim:
                pair_to_dep_pos_sim[(h, lemma)] = (dep, pos, 0.0)
    
    all_pairs_with_info = []
    
    for h, lemma in tqdm(unique_pairs, desc="Interrogation de l'API JDM"):
        try:
            resp = requests.get(f"{BASE_URL}/from/{h}/to/{lemma}")
            resp.raise_for_status()
            rels = resp.json().get("relations", [])
            
            # Obtenir la dépendance, POS et similarité
            dep, pos, sim = pair_to_dep_pos_sim.get((h, lemma), ('', '', 0.0))
            
            # Ajouter aux résultats qu'il y ait des relations ou non
            pair_info = {
                "head": h,
                "lemma": lemma,
                "dep": dep,
                "pos": pos,
                "sim": sim,
                "relations": rels
            }
            all_pairs_with_info.append(pair_info)
            
            # Ajouter aussi aux résultats des relations s'il y a des relations
            if rels:
                results.append({
                    "node1": h,
                    "node2": lemma,
                    "relations": rels,
                })
        except Exception as e:
            logging.error(f"Erreur lors du traitement de la paire ({h}, {lemma}) : {e}")
    
    return results, all_pairs_with_info

def main():
    global THRESHOLD
    
    parser = argparse.ArgumentParser(description='Extraire les relations sémantiques d\'un texte d\'entrée.')
    parser.add_argument('--text', '-t', type=str, help='Texte à traiter')
    parser.add_argument('--file', '-f', type=str, help='Fichier contenant le texte à traiter')
    parser.add_argument('--no-model', action='store_true', help='Ignorer le filtrage Word2Vec (plus rapide mais moins précis)')
    parser.add_argument('--threshold', type=float, default=THRESHOLD, help=f'Seuil de similarité (défaut : {THRESHOLD})')
    parser.add_argument('--output', '-o', type=str, help='Chemin du fichier CSV de sortie')
    parser.add_argument('--output-dir', '-d', default='output', type=str, help='Répertoire de sortie pour les fichiers CSV (nom de fichier dérivé du fichier d\'entrée)')
    
    args = parser.parse_args()
    
    # Obtenir le texte d'entrée
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
            print(f"Erreur lors de la lecture du fichier : {e}")
            return
    else:
        print("Veuillez fournir un texte d'entrée avec --text ou --file")
        parser.print_help()
        return
    
    # Déterminer le chemin du fichier de sortie
    output_file = None
    if args.output:
        output_file = args.output
    elif args.output_dir:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        output_file = os.path.join(args.output_dir, f"{input_file_name}.csv")
    else:
        print("Aucune sortie spécifiée. Les résultats ne seront affichés que dans la console.")
        print("Utilisez --output ou --output-dir pour sauvegarder les résultats dans un fichier CSV.")

    # Charger les types de relations
    relation_types = load_relation_types()
    
    # Charger le modèle si nécessaire
    model = None
    if not args.no_model:
        model = load_word2vec_model()
        if model is None:
            print("Avertissement : Exécution sans filtrage Word2Vec")

    # Mettre à jour le seuil si spécifié
    if args.threshold != THRESHOLD:
        THRESHOLD = args.threshold
    
    # Traiter le texte
    results, all_pairs_with_info = process_text(input_text, model, relation_types)
    
    # Appliquer la logique calcul_max_w pour sélectionner la meilleure relation pour chaque paire
    for pair in all_pairs_with_info:
        relations = pair.get('relations', [])
        relation_name, weight = select_best_relation(relations, relation_types)
        pair['relation_name'] = relation_name
        pair['w'] = weight
    
    # Écrire les résultats dans un fichier CSV si la sortie est spécifiée
    if output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8', newline='') as csvfile:
            # Écrire les données CSV
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
        
        print(f"\nRelations sauvegardées dans {output_file}")
    
    # Afficher les résultats
    if results:
        print("\n===== RELATIONS SÉMANTIQUES =====")
        print(f"Trouvé {len(results)} paires avec relations sur {len(all_pairs_with_info)} paires totales")
        
        # Afficher seulement la meilleure relation pour chaque paire (limité à 10 pour éviter une sortie excessive)
        pairs_with_relations = [p for p in all_pairs_with_info if p.get('relations')]
        for i, pair in enumerate(pairs_with_relations):
            head = pair['head']
            lemma = pair['lemma']
            relation_name = pair['relation_name']
            weight = pair['w']
            
            if relation_name:  # Afficher seulement si une relation a été trouvée
                print(f"{i+1}. {head} → {relation_name} → {lemma}  (poids : {weight})")
        
    else:
        print("\nAucune relation sémantique trouvée.")

if __name__ == "__main__":
    main()
