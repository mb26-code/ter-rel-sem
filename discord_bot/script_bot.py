
import warnings
warnings.filterwarnings('ignore', module='urllib3')

import signal
import time
import os
import sys
from contextlib import contextmanager
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

import csv
from tqdm import tqdm
import logging
import discord
from discord.ext import commands
import requests
from bs4 import BeautifulSoup
import io
import asyncio

from algo_bot import process_text, load_word2vec_model, load_relation_types
import subprocess
import datetime


# Import configurations
from configs import (
    BASE_JDM_URL, BASE_DIR, PATH_RELATIONS_TYPES, SPACY_MODEL,
    WORD2VEC_MODEL_PATH, LIMIT_WORD2VEC, THRESHOLD, INTERESTING_DEPS,
    PIVOTS, RELATION_PRIORITY, DEFAULT_RELATION_PRIORITY, LOGGING_CONFIG, Colors
)

load_dotenv()

intents = discord.Intents.default()
bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    print(f"Connecté en tant que {bot.user}")

@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound):
        await ctx.send("Commande non reconnue.")
    else:
        raise error 

@bot.command(name='analyse_web')
async def analyse_url(ctx, *, url: str):
    output_path = f"relations.csv"

    await ctx.send("Recherche du lien et traitement du texte ... \n") 

    texte = get_clean_text_from_url(url)

    # ======= AJOUT : sauvegarder l'URL dans un fichier .txt avec date et heure =======
    now = datetime.datetime.now()
    horodatage = now.strftime("%d/%m/%Y %H:%M:%S")

    with open("liens_utilisateurs.txt", "a", encoding="utf-8") as f:
        f.write(f"Utilisateur : {ctx.author} | Date : {horodatage}\n")
        f.write(f"URL : {url}\n")
        f.write("="*70 + "\n")
    # =================================================================================

    await ctx.send("Texte traité ... \n") 

    try:
        await ctx.send("Recherche de relations ... (peut prendre 1 à 2 min) \n") 

        results, all_pairs_with_info = analyse_texte(texte)

        if results:
            await ctx.send("\n===== RELATIONS SÉMANTIQUES =====")
            await ctx.send(
                f"Trouvé {len(results)} paires avec relations sur "
                f"{'plus de 200' if len(all_pairs_with_info) >= 200 else len(all_pairs_with_info)} paires totales"
            )

            pairs_with_relations = [p for p in all_pairs_with_info if p.get('relation_name')]
            db_relations = [p for p in pairs_with_relations if p.get('source') == 'database']
            jdm_relations = [p for p in pairs_with_relations if p.get('source') == 'jdm']

            await ctx.send(f"  - Relations de la base de données: {len(db_relations)}")
            await ctx.send(f"  - Relations de JdM: {len(jdm_relations)} \n")

            # === TRI PAR POIDS DÉCROISSANT ===
            pairs_with_relations_sorted = extraire_relations_ordre_par_w(pairs_with_relations)

            for i, pair in enumerate(pairs_with_relations_sorted):
                head = pair['head']
                lemma = pair['lemma']
                relation_name = pair['relation_name']
                weight = pair['w']
                source = pair.get('source', 'unknown')

                if relation_name:
                    colored_relation = f"**__{relation_name}__**"
                    source_indicator = f"`[{source}]`"
                    await ctx.send(
                        f"> **{i + 1}.** `{head}` → {colored_relation} → `{lemma}`  *(poids : {weight})* {source_indicator}"
                    )
        else:
            await ctx.send("\nAucune relation sémantique trouvée. Le site bloque probablement les requêtes automatisées.")

        if os.path.exists(output_path):
            await ctx.send("Voici le fichier CSV généré :", file=discord.File(output_path))
        else:
            await ctx.send("Erreur : aucun fichier CSV n’a été généré.")

        await ctx.send("FIN")

    except Exception as e:
        await ctx.send(f"Erreur lors de l'exécution : {e}")


@bot.command(name='analyse_texte')
async def analyse_texte_command(ctx):
    await ctx.send("Veuillez coller le texte : ")

    def check(m):
        return m.author == ctx.author and m.channel == ctx.channel

    try:
        msg = await bot.wait_for('message', check=check, timeout=120)  # 2 min pour répondre
        texte = msg.content

        # ======= AJOUT : sauvegarder le texte dans un fichier .txt avec date et heure =======
        now = datetime.datetime.now()
        horodatage = now.strftime("%d/%m/%Y %H:%M:%S")

        with open("textes_utilisateurs.txt", "a", encoding="utf-8") as f:
            f.write(f"Utilisateur : {ctx.author} | Date : {horodatage}\n")
            f.write(texte + "\n")
            f.write("="*70 + "\n")
        # =====================================================================================

        await ctx.send("Recherche de relations ... (peut prendre 1 à 2 min) \n")

        results, all_pairs_with_info = analyse_texte(texte)
        output_path = "relations.csv"

        if results:
            await ctx.send("\n===== RELATIONS SÉMANTIQUES =====")
            await ctx.send(
                f"Trouvé {len(results)} paires avec relations sur "
                f"{'plus de 200' if len(all_pairs_with_info) >= 200 else len(all_pairs_with_info)} paires totales"
            )

            pairs_with_relations = [p for p in all_pairs_with_info if p.get('relation_name')]
            db_relations = [p for p in pairs_with_relations if p.get('source') == 'database']
            jdm_relations = [p for p in pairs_with_relations if p.get('source') == 'jdm']

            await ctx.send(f"  - Relations de la base de données: {len(db_relations)}")
            await ctx.send(f"  - Relations de JdM: {len(jdm_relations)} \n")

            # TRI PAR POIDS DÉCROISSANT
            pairs_with_relations_sorted = extraire_relations_ordre_par_w(pairs_with_relations)
            

            for i, pair in enumerate(pairs_with_relations_sorted):
                head = pair['head']
                lemma = pair['lemma']
                relation_name = pair['relation_name']
                weight = pair['w']
                source = pair.get('source', 'unknown')

                if relation_name:
                    colored_relation = f"**__{relation_name}__**"
                    source_indicator = f"`[{source}]`"
                    await ctx.send(
                        f"> **{i + 1}.** `{head}` → {colored_relation} → `{lemma}`  *(poids : {weight})* {source_indicator}"
                    )
        else:
            await ctx.send("Aucune relation sémantique trouvée. Le texte est sûrement trop court")

        if os.path.exists(output_path):
            await ctx.send("Voici le fichier CSV généré :", file=discord.File(output_path))
        else:
            await ctx.send("Erreur : aucun fichier CSV n’a été généré.")

        await ctx.send("FIN")

    except asyncio.TimeoutError:
        await ctx.send("Temps écoulé. Veuillez relancer la commande.")
    except Exception as e:
        await ctx.send(f"Erreur lors de l'exécution : {e}")




@bot.command(name='aide')
async def afficher_aide_embed(ctx):
    embed = discord.Embed(
        title="Aide du Bot de Relations Sémantiques",
        description="Commandes disponibles pour analyser un mot ou un texte dans le domaine de la gastronomie.",
        color=discord.Color.blue()
    )

    embed.add_field(
        name="ATTENTION",
        value="Une recherche complète peut durer 2 min maximum // ET pour savoir si un message est fini, il se termine par FIN",
        inline=False
    )

    embed.add_field(
        name="1. !analyse_web <url>",
        value="Analyse un mot à partir du contenu HTML d’une page web.\n"
              "**Exemple :** `!analyse_web https://fr.wikipedia.org/wiki/Chocolat`",
        inline=False
    )

    embed.add_field(
        name="2. !analyse_texte",
        value="Analyse uniquement les relations contenant le mot donné dans un court texte.\n"
              "Vous serez invité à coller le texte dans un second temps\n"
              "**Exemple :** `!analyse_texte ***** ET ENSUITE ON COLLE ****** \"Ajoutez du sel au plat.\"`",
        inline=False
    )

    embed.set_footer(text="Projet académique – Extraction de relations sémantiques à partir de textes culinaires.")
    await ctx.send(embed=embed)



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
            return relation_map
    except Exception as e:
        logging.error(f"Erreur lors du chargement des types de relations : {e}")
        return {}


# Configuration du logging
logging.basicConfig(**LOGGING_CONFIG)

# Charger le modèle spaCy
try:
    nlp = spacy.load(SPACY_MODEL)
except OSError:
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
            return None

        model = KeyedVectors.load_word2vec_format(model_path, binary=False, limit=limit)
        return model
    except Exception as e:
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
            return new_relation, str(new_w)
        else:
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


    # Interroger la base de données Supabase pour les relations
    results = []

    if model is not None:
        pairs_to_query = [(h, lemma) for h, lemma, dep, pos, sim in filtered if sim >= THRESHOLD]
    else:
        pairs_to_query = [(h, lemma) for h, lemma, _, _ in flat_pairs]

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


    max_iterations = 200
    limited_unique_pairs = unique_pairs[:max_iterations]

    for h, lemma in tqdm(limited_unique_pairs):
        start_time = time.time()
        
        try:
            with timeout(1):  # Timeout d'1 seconde pour toute l'itération
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
                        # Utiliser les valeurs calculées pour l'affichage
                        best_relation = calculated_best_relation
                        best_relation_w = calculated_best_relation_w
                    else:
                        source = "database_no_relation"
                        no_relation_count += 1
                        no_relation_pairs.append((h, lemma, "database"))
                        best_relation = ""
                        best_relation_w = ""
                else:
                    # Si la paire n'existe pas dans la base, interroger l'API JdM

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
                    else:
                        best_relation, best_relation_w = "", ""
                        source = "no_relation"
                        no_relation_count += 1
                        no_relation_pairs.append((h, lemma, "jdm"))

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

        except TimeoutError:
            # En cas de timeout, ajouter une entrée vide avec source timeout
            logging.warning(f"Timeout pour la paire ({h}, {lemma}) après 1 seconde")
            
            dep, pos, sim = pair_to_dep_pos_sim.get((h, lemma), ('', '', 0.0))
            pair_info = {
                "head": h,
                "lemma": lemma,
                "dep": dep,
                "pos": pos,
                "sim": sim,
                "relations": [],
                "relation_name": "",
                "w": "",
                "original_best_relation": "",
                "original_best_relation_w": "",
                "new_relation": "",
                "new_relation_w": "",
                "source": "timeout"
            }
            all_pairs_with_info.append(pair_info)
            
        except Exception as e:
            logging.error(f"Erreur lors du traitement de la paire ({h}, {lemma}) : {e}")
            # Optionnel : ajouter une entrée d'erreur
            dep, pos, sim = pair_to_dep_pos_sim.get((h, lemma), ('', '', 0.0))
            pair_info = {
                "head": h,
                "lemma": lemma,
                "dep": dep,
                "pos": pos,
                "sim": sim,
                "relations": [],
                "relation_name": "",
                "w": "",
                "original_best_relation": "",
                "original_best_relation_w": "",
                "new_relation": "",
                "new_relation_w": "",
                "source": "error"
            }
            all_pairs_with_info.append(pair_info)
        
        # Afficher le temps d'exécution pour cette itération
        elapsed = time.time() - start_time




    return results, all_pairs_with_info


def analyse_texte(string):
    

    output_file = "relations.csv"

    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)


    # Charger les types de relations
    relation_types = load_relation_types()

    # Charger le modèle si nécessaire
    model = load_word2vec_model()

    # Traiter le texte
    results, all_pairs_with_info = process_text(string, model, relation_types)

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

    

    return results, all_pairs_with_info

@contextmanager
def timeout(duration):
    """Context manager pour limiter le temps d'exécution"""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Opération timeout après {duration} seconde(s)")
    
    # Configurer le signal d'alarme
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    
    try:
        yield
    finally:
        # Restaurer l'ancien handler et annuler l'alarme
        signal.signal(signal.SIGALRM, old_handler)
        signal.alarm(0)


def get_clean_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Supprimer les balises script et style
        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()

        # Si c'est une page Wikipédia, ne prendre que les <p>
        if 'wikipedia.org' in url:
            paragraphs = soup.find_all('p')
            text = '\n'.join(p.get_text() for p in paragraphs)
        else:
            text = soup.get_text(separator='\n')

        # Nettoyage final du texte
        lines = [line.strip() for line in text.splitlines()]
        clean_text = '\n'.join(line for line in lines if line)

        return clean_text
    except Exception as e:
        return f"Erreur : {e}"

def extraire_relations_ordre_par_w(data):
    relations = []
    for item in data:
        relation = {
            "head": item["head"],
            "lemma": item["lemma"],
            "relation_name": item.get("relation_name", ""),
            "w": float(item.get("w", 0)),
            "source": item.get("source", "")
        }
        relations.append(relation)
    
    relations_triees = sorted(relations, key=lambda r: r["w"], reverse=True)
    return relations_triees


load_dotenv()
bot.run(os.getenv("DISCORD_TOKEN"))