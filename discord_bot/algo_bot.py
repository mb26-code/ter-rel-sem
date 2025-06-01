print("\n===== EXTRACTION DES RELATIONS SÉMANTIQUES =====")

import warnings

warnings.filterwarnings('ignore', module='urllib3')

import os
import sys
from dotenv import load_dotenv
from supabase import create_client, Client
import spacy
import numpy as np
from gensim.models import KeyedVectors
from numpy.linalg import norm
from dataclasses import dataclass
from typing import List, Optional
import json
import string
import argparse
import csv
from tqdm import tqdm
import logging
import requests

# Import configurations
from configs import (
    BASE_JDM_URL, BASE_DIR, PATH_RELATIONS_TYPES, SPACY_MODEL,
    WORD2VEC_MODEL_PATH, LIMIT_WORD2VEC, THRESHOLD, INTERESTING_DEPS,
    PIVOTS, RELATION_PRIORITY, DEFAULT_RELATION_PRIORITY, LOGGING_CONFIG, Colors
)

# Load environment variables from .env file
load_dotenv()

url: str = os.getenv("SUPABASE_URL")
key: str = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)


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
logging.basicConfig(**LOGGING_CONFIG)

# Charger le modèle spaCy
try:
    nlp = spacy.load(SPACY_MODEL)
    print(f"Modèle '{SPACY_MODEL}' chargé avec succès")
except OSError:
    print(f"Erreur : Le modèle '{SPACY_MODEL}' n'est pas installé.")
    print(f"Veuillez l'installer avec : python -m spacy download {SPACY_MODEL}")
    sys.exit(1)


@dataclass
class TokenAnnotation:
    text: str  # forme de surface
    lemma: str  # forme canonique
    pos: str  # étiquette morpho-syntaxique
    dep: str  # étiquette de dépendance
    head: int  # index du token tête dans sa phrase
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


def load_word2vec_model(model_path=WORD2VEC_MODEL_PATH, limit=LIMIT_WORD2VEC):
    """
    Charger le modèle Word2Vec
    """
    try:
        model_path = os.path.join(BASE_DIR, model_path)
        if not os.path.exists(model_path):
            print("Word2Vec inactif : fichier introuvable.")
            print(f"Erreur : modèle Word2Vec non trouvé à l’emplacement : {model_path}")
            print("Téléchargez-le avec la commande suivante si nécessaire :")
            print('wget -c "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.vec.gz"')
            return None

        print(f"Chargement du modèle Word2Vec depuis : {model_path}")
        model = KeyedVectors.load_word2vec_format(model_path, binary=False, limit=limit)
        print(f"Modèle Word2Vec chargé avec succès ({len(model.key_to_index)} termes)")
        print("Word2Vec activé : filtrage par similarité contextuelle en cours")
        return model
    except Exception as e:
        print("Word2Vec inactif : erreur lors du chargement")
        print(f"Erreur : {e}")
        return None



def pair_vector(model, h, lemma):
    """Moyenne des deux vecteurs de mots."""
    return (model[h] + model[lemma]) / 2


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Calculer la similarité cosinus entre vecteurs."""
    return float(a.dot(b) / (norm(a) * norm(b)))


def batch_query_database(pairs):
    """
    Requête par lots pour optimiser les accès à la base de données (lecture seule)
    """
    results = {}

    # Créer une requête avec plusieurs conditions OR
    if not pairs:
        return results

    try:
        # Pour l'instant, nous faisons des requêtes individuelles
        # car Supabase ne supporte pas facilement les requêtes OR complexes avec des tuples
        for node1, node2 in pairs:
            response = supabase.table('semantic_relations').select('*') \
                .eq('node1', node1) \
                .eq('node2', node2) \
                .limit(1) \
                .execute()

            if response.data:
                results[(node1, node2)] = response.data[0]
    except Exception as e:
        logging.error(f"Erreur lors de la requête par lot : {e}")

    return results


def query_jdm_api(head, lemma):
    """
    Interroge l'API JdM pour récupérer les relations entre deux termes.

    Args:
        head (str): Premier terme de la paire
        lemma (str): Deuxième terme de la paire

    Returns:
        list: Liste des relations ou liste vide en cas d'erreur
    """
    try:
        resp = requests.get(f"{BASE_JDM_URL}/from/{head}/to/{lemma}")
        resp.raise_for_status()
        relations = resp.json().get("relations", [])
        return relations
    except requests.exceptions.RequestException as e:
        logging.error(f"Erreur lors de l'appel API JdM pour {head} -> {lemma}: {e}")
        return []
    except Exception as e:
        logging.error(f"Erreur inattendue lors de l'appel API JdM pour {head} -> {lemma}: {e}")
        return []


def select_best_jdm_relation(relations, relation_types):
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
    relation_priority = RELATION_PRIORITY

    # Priorité par défaut
    default_priority = DEFAULT_RELATION_PRIORITY

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


def select_best_relation(db_entry):
    """
    Sélectionne la meilleure relation entre best_relation et new_relation.

    Args:
        db_entry (dict): Entrée de la base de données contenant best_relation, best_relation_w,
                        new_relation, new_relation_w

    Returns:
        tuple: (relation_name, weight)
    """
    if not db_entry:
        return '', ''

    # Récupérer les valeurs de la base de données
    best_relation = db_entry.get('best_relation', '') or ''
    best_relation_w = db_entry.get('best_relation_w')
    new_relation = db_entry.get('new_relation', '') or ''
    new_relation_w = db_entry.get('new_relation_w')

    # Convertir les poids en float si ils existent
    best_w = float(best_relation_w) if best_relation_w is not None else None
    new_w = float(new_relation_w) if new_relation_w is not None else None

    # Si aucune relation n'est disponible
    if not best_relation and not new_relation:
        return '', ''

    # Si seulement best_relation existe
    if best_relation and not new_relation:
        return best_relation, str(best_w) if best_w is not None else ''

    # Si seulement new_relation existe
    if new_relation and not best_relation:
        return new_relation, str(new_w) if new_w is not None else ''

    # Si les deux relations existent, prendre celle avec le poids le plus élevé
    if best_relation and new_relation and best_w is not None and new_w is not None:
        if new_w > best_w:
            print(
                f"Choix de new_relation: {new_relation} (poids: {new_w}) > best_relation: {best_relation} (poids: {best_w})")
            return new_relation, str(new_w)
        else:
            print(
                f"Choix de best_relation: {best_relation} (poids: {best_w}) >= new_relation: {new_relation} (poids: {new_w})")
            return best_relation, str(best_w)

    # Fallback: retourner best_relation s'il existe
    if best_relation:
        return best_relation, str(best_w) if best_w is not None else ''

    return new_relation, str(new_w) if new_w is not None else ''


def process_text(text, model=None, relation_types=None):
    """
    Traiter le texte d'entrée et extraire les relations (lecture seule)
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

    print(f"Extrait {len(flat_pairs)} paires linguistiques \n")

    # Si le modèle est disponible, filtrer les paires par similarité au domaine
    filtered = []

    if model is not None:
        # Calculer le centroïde du domaine
        pivot_vecs = [model[p] for p in PIVOTS if p in model]
        if not pivot_vecs:
            raise ValueError("Aucun de vos pivots n'était dans le modèle !")
        domain_centroid = np.mean(pivot_vecs, axis=0)

        # Filtrer les paires basées sur la similarité
        for h, lemma, dep, pos in flat_pairs:
            if h in model and lemma in model:
                vec = pair_vector(model, h, lemma)
                sim = cosine_sim(domain_centroid, vec)
                filtered.append((h, lemma, dep, pos, sim))

        print("--------------")

        print(f"Filtré à {len(filtered)} paires avec des mots dans le modèle")

    # Interroger la base de données Supabase pour les relations
    results = []

    if model is not None:
        pairs_to_query = [(h, lemma) for h, lemma, dep, pos, sim in filtered if sim >= THRESHOLD]
        print(f"Interrogation de la base de données pour {len(pairs_to_query)} paires (similarité ≥ {THRESHOLD})")
        print("------------  \n")
    else:
        pairs_to_query = [(h, lemma) for h, lemma, _, _ in flat_pairs]
        print(f"Interrogation de la base de données pour toutes les {len(pairs_to_query)} paires (pas de filtrage)")

        print("------------  \n")

    # Supprimer les doublons avant d'interroger la base
    seen_pairs = set()
    unique_pairs = []
    for pair in pairs_to_query:
        if pair not in seen_pairs:
            unique_pairs.append(pair)
            seen_pairs.add(pair)

    # Créer un dictionnaire de recherche pour la dépendance, POS et similarité
    pair_to_dep_pos_sim = {}
    if model is not None:
        for h, lemma, dep, pos, sim in filtered:
            pair_to_dep_pos_sim[(h, lemma)] = (dep, pos, sim)
    else:
        # Si pas de modèle, utiliser seulement la dépendance et POS
        for h, lemma, dep, pos in flat_pairs:
            if (h, lemma) not in pair_to_dep_pos_sim:
                pair_to_dep_pos_sim[(h, lemma)] = (dep, pos, 0.0)

    all_pairs_with_info = []

    # Compteurs pour les statistiques
    db_found_count = 0
    jdm_found_count = 0
    no_relation_count = 0

    # Listes pour tracker les paires par source
    db_found_pairs = []
    jdm_found_pairs = []
    no_relation_pairs = []

    for h, lemma in tqdm(unique_pairs, desc="Interrogation de la base de données"):
        try:
            # Interroger Supabase pour cette paire
            response = supabase.table('semantic_relations').select('*').eq('node1', h).eq('node2', lemma).execute()

            # Obtenir la dépendance, POS et similarité
            dep, pos, sim = pair_to_dep_pos_sim.get((h, lemma), ('', '', 0.0))

            # Traiter les résultats de la base de données
            rels = []
            best_relation = ""
            best_relation_w = ""
            new_relation = ""
            new_relation_w = ""
            source = ""  # Pour tracker la source de la relation

            if response.data:
                # Si on trouve une entrée existante, récupérer les relations
                db_entry = response.data[0]
                if db_entry.get('relations'):
                    rels = db_entry['relations'] if isinstance(db_entry['relations'], list) else []

                # Récupérer new_relation et new_relation_w depuis la base de données
                new_relation = db_entry.get('new_relation', '') or ''
                new_relation_w = db_entry.get('new_relation_w', '') or ''

                # Récupérer les valeurs originales de la base de données pour le CSV
                original_best_relation = db_entry.get('best_relation', '') or ''
                original_best_relation_w = db_entry.get('best_relation_w', '') or ''

                # Calculer la meilleure relation seulement pour l'affichage console
                calculated_best_relation, calculated_best_relation_w = select_best_relation(db_entry)

                if calculated_best_relation:
                    source = "database"
                    db_found_count += 1
                    db_found_pairs.append((h, lemma, calculated_best_relation, calculated_best_relation_w))
                    print(f"✓ Base: {h} → {calculated_best_relation} → {lemma} (poids: {calculated_best_relation_w}) \n")
                    # Utiliser les valeurs calculées pour l'affichage
                    best_relation = calculated_best_relation
                    best_relation_w = calculated_best_relation_w
                else:
                    source = "database_no_relation"
                    no_relation_count += 1
                    no_relation_pairs.append((h, lemma, "database"))
                    print(f"✗ Base: {h} → {lemma} (entrée trouvée mais aucune relation) \n")
                    best_relation = ""
                    best_relation_w = ""
            else:
                # Si la paire n'existe pas dans la base, interroger l'API JdM
                print(f"? Paire non trouvée dans la base: {h} → {lemma}. Interrogation de l'API JdM... \n")

                # Initialiser new_relation et new_relation_w comme vides pour les données JdM
                new_relation = ""
                new_relation_w = ""

                # Pour JdM, pas de valeurs originales de base de données
                original_best_relation = ""
                original_best_relation_w = ""

                # Interroger l'API JdM pour cette paire
                jdm_relations = query_jdm_api(h, lemma)
                rels = jdm_relations

                # Sélectionner la meilleure relation JdM si des relations ont été trouvées
                if jdm_relations and relation_types:
                    best_relation, best_relation_w = select_best_jdm_relation(jdm_relations, relation_types)
                    source = "jdm"
                    jdm_found_count += 1
                    jdm_found_pairs.append((h, lemma, best_relation, best_relation_w))
                    print(f"✓ JdM: {h} → {best_relation} → {lemma} (poids: {best_relation_w})")
                else:
                    best_relation, best_relation_w = "", ""
                    source = "no_relation"
                    no_relation_count += 1
                    no_relation_pairs.append((h, lemma, "jdm"))
                    if not jdm_relations:
                        print(f"✗ JdM: {h} → {lemma} (aucune relation trouvée) \n")
                    else:
                        print(f"✗ JdM: {h} → {lemma} (relations trouvées mais aucun type de relation valide) \n")

            # Ajouter aux résultats qu'il y ait des relations ou non
            pair_info = {
                "head": h,
                "lemma": lemma,
                "dep": dep,
                "pos": pos,
                "sim": sim,
                "relations": rels,
                "relation_name": best_relation,  # Pour l'affichage console
                "w": best_relation_w,  # Pour l'affichage console
                "original_best_relation": original_best_relation,  # Pour le CSV
                "original_best_relation_w": original_best_relation_w,  # Pour le CSV
                "new_relation": new_relation,
                "new_relation_w": new_relation_w,
                "source": source  # Ajouter la source de la relation
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

    # Afficher les statistiques de sourcing
    print(f"\n===== STATISTIQUES DE SOURCING =====")
    print(f"Total des paires traitées: {len(unique_pairs)}")
    print(f"Relations trouvées dans la base de données: {db_found_count}")
    print(f"Relations trouvées dans JdM: {jdm_found_count}")
    print(f"Paires sans relation: {no_relation_count}")
    print(f"Taux de succès global: {((db_found_count + jdm_found_count) / len(unique_pairs) * 100):.1f}%") if len(unique_pairs) > 0 else print("Aucune paire unique trouvée.")

    # Affichage détaillé optionnel des paires sans relation (limité pour éviter le spam)
    if no_relation_pairs and len(no_relation_pairs) <= 20:
        print(f"\nPaires sans relation ({len(no_relation_pairs)}):")
        for h, lemma, attempted_source in no_relation_pairs:
            print(f"  - {h} → {lemma} (testé dans {attempted_source})")
    elif len(no_relation_pairs) > 20:
        print(f"\nPaires sans relation: {len(no_relation_pairs)} (trop nombreuses pour affichage détaillé)")

    print("=" * 50)

    return results, all_pairs_with_info


def main():
    global THRESHOLD

    parser = argparse.ArgumentParser(description='Extraire les relations sémantiques d\'un texte d\'entrée.')
    parser.add_argument('--text', '-t', type=str, help='Texte à traiter')
    parser.add_argument('--file', '-f', type=str, help='Fichier contenant le texte à traiter')
    parser.add_argument('--no-model', action='store_true',
                        help='Ignorer le filtrage Word2Vec (plus rapide mais moins précis)')
    parser.add_argument('--threshold', type=float, default=THRESHOLD,
                        help=f'Seuil de similarité (défaut : {THRESHOLD})')
    parser.add_argument('--output', '-o', type=str, help='Chemin du fichier CSV de sortie')
    parser.add_argument('--output-dir', '-d', default='output', type=str,
                        help='Répertoire de sortie pour les fichiers CSV (nom de fichier dérivé du fichier d\'entrée)')

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

    # Les relations et poids sont déjà récupérés de la base de données
    # Pas besoin d'appliquer select_best_relation car les données viennent de la base

    # Écrire les résultats dans un fichier CSV si la sortie est spécifiée
    if output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

        with open(output_file, 'w', encoding='utf-8', newline='') as csvfile:
            # Écrire les données CSV avec les noms de colonnes de la base de données
            fieldnames = [
                'node1', 'node2', 'dep', 'pos', 'sim',
                'relations', 'best_relation', 'best_relation_w',
                'new_relation', 'new_relation_w', 'source',
                'filtrage'
            ]

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for pair in all_pairs_with_info:
                writer.writerow({
                    'node1': pair['head'],
                    'node2': pair['lemma'],
                    'dep': pair['dep'],
                    'pos': pair['pos'],
                    'sim': pair['sim'],
                    'relations': json.dumps(pair['relations']),
                    'best_relation': pair['original_best_relation'],  # Utiliser les valeurs originales de la base
                    'best_relation_w': pair['original_best_relation_w'],  # Utiliser les valeurs originales de la base
                    'new_relation': pair.get('new_relation', ''),
                    'new_relation_w': pair.get('new_relation_w', ''),
                    'source': pair.get('source', ''),
                    'filtrage': 'oui' if model is not None else 'non'

                })

        #print(f"\nRelations sauvegardées dans {output_file}")

    # Afficher les résultats
    if results:
        print("\n===== RELATIONS SÉMANTIQUES =====")
        print(f"Trouvé {len(results)} paires avec relations sur {len(all_pairs_with_info)} paires totales")

        # Calculer les statistiques de source pour l'affichage final
        pairs_with_relations = [p for p in all_pairs_with_info if p.get('relation_name')]
        db_relations = [p for p in pairs_with_relations if p.get('source') == 'database']
        jdm_relations = [p for p in pairs_with_relations if p.get('source') == 'jdm']

        print(f"  - Relations de la base de données: {len(db_relations)}")
        print(f"  - Relations de JdM: {len(jdm_relations)} \n")

        # Afficher seulement la meilleure relation pour chaque paire
        for i, pair in enumerate(pairs_with_relations):
            head = pair['head']
            lemma = pair['lemma']
            relation_name = pair['relation_name']
            weight = pair['w']
            source = pair.get('source', 'unknown')

            if relation_name:  # Afficher seulement si une relation a été trouvée
                # Mettre la relation en couleur pour la rendre plus visible
                colored_relation = f"{Colors.MAGENTA}{Colors.BOLD}{relation_name}{Colors.RESET}"
                source_indicator = f"{Colors.CYAN}[{source}]{Colors.RESET}"
                print(f"{i + 1}. {head} → {colored_relation} → {lemma}  (poids : {weight}) {source_indicator}")
        print("\n")

    else:
        print("\nAucune relation sémantique trouvée.")


if __name__ == "__main__":
    main()
