
import os
import sys
import logging

# Base directory configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# API and URL configuration
BASE_JDM_URL = "https://jdm-api.demo.lirmm.fr/v0/relations"

# File paths
PATH_RELATIONS_TYPES = os.path.join(BASE_DIR, "relations_types.json")


# spaCy model configuration
SPACY_MODEL = "fr_core_news_lg"

# Word2Vec configuration
WORD2VEC_MODEL_PATH = 'cc.fr.300.vec.gz'
LIMIT_WORD2VEC = 50000  # Limite pour éviter de charger trop de mots

# Similarity threshold
THRESHOLD = 0.5

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
PIVOTS = [
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

# Relation priorities for JdM API response selection
RELATION_PRIORITY = {
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

# Default priority for relations not in the priority list
DEFAULT_RELATION_PRIORITY = 1.0

# Configuration du logging
LOGGING_CONFIG = {
    'level': logging.ERROR,
    'format': '%(asctime)s - %(levelname)s - %(message)s',
    'handlers': [logging.StreamHandler(sys.stderr)]
}

# Codes couleur ANSI pour la sortie colorée
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'