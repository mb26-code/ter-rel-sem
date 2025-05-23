"""
Tentative d'algorithme par patrons prédéfinis...
"""

print()

import os
import sys

from dotenv import load_dotenv
import psycopg2

import csv

import spacy
from spacy.matcher import Matcher

from itertools import combinations
from collections import Counter

import requests
from json import dumps
def print_JSON(obj):
    print(dumps(obj, indent = 4, ensure_ascii = False))

import time


#jolies couleurs affichage console
ANSI_RED = "\033[31m"
ANSI_GREEN = "\033[32m"
ANSI_YELLOW = "\033[93m"
ANSI_BLUE = "\033[94m"

ANSI_RESET = "\033[0m"

if len(sys.argv) > 1:
    try:
        nombre_max_documents_a_traiter = int(sys.argv[1])
        if nombre_max_documents_a_traiter <= 0:
            print(f"{ANSI_RED}Le nombre maximum de documents doit être un entier positif.{ANSI_RESET}")
            sys.exit(1)
    except ValueError:
        print(f"{ANSI_RED}L'argument doit être un entier.{ANSI_RESET}")
        sys.exit(1)
else:
    nombre_max_documents_a_traiter = None

#où trouver le corpus
chemin_dossier_corpus = os.path.join("./", "corpus")
#quels dossiers du corpus traiter
sous_dossiers_corpus = ["wikipedia", "marmiton"]

nlp = spacy.load("fr_core_news_lg")

DUREE_PAUSE_ENTRE_APPEL_JDM_SECONDES = 0.10
URL_API_JDM = "https://jdm-api.demo.lirmm.fr/"


patrons = {
    # r_ingredient (420), r_preparation (421)
    "[plat] est à base de [aliment]" : [420, -421],
    "[plat] contient [aliment]" : [420, -421],
    "[plat] est composé de [aliment]" : [420, -421],
    "[plat] à [aliment]" : [420, -421],
    "[aliment] entrer dans la composition de [plat]" : [-420, 421],
    "[aliment] est utilisé dans [plat]" : [-420, 421],

    # r_agent (13)
    "[agent] préparer [plat]" : [13],
    "[agent] cuisiner [plat]" : [13],
    "[agent] réaliser [plat]" : [13],

    # r_patient (14)
    "[verbe] [plat]" : [14],  # à combiner avec des verbes culinaires
    "préparer [plat]" : [14],
    "faire cuire [plat]" : [14],

    # r_instr (16)
    "[action] avec [instrument]" : [16],
    "utiliser [instrument] pour [action]" : [-16],
    "se servir de [instrument] pour [action]" : [-16],

    # r_lieu (15)
    "[objet] se trouve dans [lieu]" : [15],
    "[objet] est conservé dans [lieu]" : [15],
    "[lieu] contient [objet]" : [-15],

    # r_carac (17), r_has_prop (153)
    "[aliment] est [adjectif]" : [17, 153],
    "[aliment] a un goût [adjectif]" : [153],
    "[plat] est typiquement [adjectif]" : [17],

    # r_syn (5)
    "[terme] ou [synonyme]" : [5],
    "[terme], aussi appelé [synonyme]" : [5],
    "[synonyme] est un synonyme de [terme]" : [-5],

    # r_isa (6)
    "[terme] est un type de [hyperonyme]" : [6],
    "[terme] est une sorte de [hyperonyme]" : [6],
    "[hyperonyme] comme [terme]" : [-6],

    # r_domain (3)
    "[terme] dans le domaine de [domaine]" : [3],
    "[terme] est utilisé en [domaine]" : [3],

    # r_has_part (9)
    "[plat] contient [partie]" : [9],
    "[partie] est un élément de [plat]" : [-9],

    # r_holo (10)
    "[partie] fait partie de [tout]" : [-10],
    "[tout] inclut [partie]" : [10],

    # r_lieu_action (30)
    "[lieu] permet de [action]" : [30],
    "[action] peut être faite dans [lieu]" : [-30],

    # r_action_lieu (31)
    "[action] se fait dans [lieu]" : [31],
    "[lieu] est utilisé pour [action]" : [-31],

    # r_manner (34)
    "[action] à [manière]" : [34],
    "[action] de manière [adjectif]" : [34],
    "[action] doucement" : [34],

    # r_accompagnement (422)
    "[plat] accompagné de [aliment]" : [422],
    "[aliment] se mange avec [plat]" : [-422],
    "[plat] servi avec [aliment]" : [422],

    # r_substitut (423)
    "[aliment1] peut remplacer [aliment2]" : [-423, 423],
    "[aliment1] comme alternative à [aliment2]" : [-423, 423],

    # r_cuisson (424)
    "[plat] cuit [mode]" : [424],
    "[cuisson] de [plat]" : [-424],
    "[plat] préparé [mode]" : [424],

    # r_duree_cuisson (425)
    "[plat] cuire pendant [durée]" : [425],
    "[durée] de cuisson pour [plat]" : [-425],

    # r_origine (426)
    "[plat] est originaire de [lieu]" : [426],
    "[plat] vient de [lieu]" : [426],
    "[lieu] est l'origine de [plat]" : [-426],
}


correspondance_types_relation_id_nom = {}
chemin_fichier_types_relation = "./tentative_patrons_predefinis/types_de_relations.csv"

with open(chemin_fichier_types_relation, mode = "r", encoding = "utf-8") as fichier_types_relation_CSV:
    lecteur_csv = csv.DictReader(fichier_types_relation_CSV, delimiter = ';')
    for ligne_type_relation in lecteur_csv:
        try:
            id_type_relation = int(ligne_type_relation["id"])
            nom_type_relation = ligne_type_relation["nom"]
            correspondance_types_relation_id_nom[id_type_relation] = nom_type_relation
        except (ValueError, KeyError) as e:
            print(f"Erreur lors de la lecture de la ligne : \"{ligne_type_relation}\" : {e}")


print("Correspondance id/nom des types de relation:")
for id_type_relation, nom_type_relation in correspondance_types_relation_id_nom.items():
    print(f"    {id_type_relation} : {nom_type_relation}")

print()
relations = {}
for ids_types_relation_patron in patrons.values():
    for id_type_relation in ids_types_relation_patron:
        abs_id = abs(id_type_relation)
        if abs_id not in relations:
            relations[abs_id] = []
print(relations)


print(
    """
    --------------------------------
    |          TRAITEMENT          |
    --------------------------------
    """
)

categorie_to_motif_Matcher = {
    "plat": {"POS": "NOUN"},
    "aliment": {"POS": "NOUN"},
    "partie": {"POS": "NOUN"},
    "instrument": {"POS": "NOUN"},
    "objet": {"POS": "NOUN"},
    "agent": {"POS": "NOUN"},
    "verbe": {"POS": "VERB"},
    "action": {"POS": "VERB"},
    "adjectif": {"POS": "ADJ"},
    "manière": {"POS": "ADJ"},
    "lieu": {"POS": "NOUN"},
    "domaine": {"POS": "NOUN"},
    "mode": {"POS": "NOUN"},
    "durée": {"LIKE_NUM": True},
    "synonyme": {"POS": "NOUN"},
    "terme": {"POS": "NOUN"},
    "cuisson": {"POS": "NOUN"},
    "hyperonyme": {"POS": "NOUN"},
}

def traiter_document(chemin_fichier_texte):
    print(f"{ANSI_GREEN}        Traitement du document \"{chemin_fichier_texte}\":{ANSI_RESET}")
    with open(chemin_fichier_texte, "r", encoding="utf-8") as f:
        texte = f.read()
        doc = nlp(texte)

        matcher = Matcher(nlp.vocab)
        corr_match_rule_name_types_relation = {}

        for ind_patron, (patron, types_relation_patron) in enumerate(patrons.items()):
            tokens_patron = patron.split()

            #on crée un motif pour chaque patron
            motif_lexico_syntaxique = []

            for token in tokens_patron:
                if token.startswith("[") and token.endswith("]"):
                    #le token représente et décrit un des mots en relation
                    categorie = token[1:-1]
                    partie_motif_mot = categorie_to_motif_Matcher.get(categorie, {"POS": "NOUN"})
                    motif_lexico_syntaxique.append(partie_motif_mot)
                else:
                    motif_lexico_syntaxique.append({"LOWER": token.lower()})

            match_rule_name = f"patron_{ind_patron}"
            matcher.add(match_rule_name, [motif_lexico_syntaxique])
            corr_match_rule_name_types_relation[match_rule_name] = types_relation_patron

        matchs = matcher(doc)

        for match_id, match_start, match_end in matchs:
            match_rule_name = nlp.vocab.strings[match_id]
            span_match = doc[match_start:match_end]

            types_relation_patron = corr_match_rule_name_types_relation.get(match_rule_name, [])
            for type_relation_rel_id in types_relation_patron:
                type_relation_id = abs(type_relation_rel_id)
                relation_mot1, relation_mot2 = (span_match[0].text, span_match[-1].text) if type_relation_rel_id > 0 else (span_match[-1].text, span_match[0].text)
                relations[type_relation_id].append((relation_mot1, relation_mot2))

debut_traitement = time.time()
compteur_documents_corpus = 0
nombre_documents_traites = 0

print(f"{ANSI_BLUE}Parcours du corpus ({chemin_dossier_corpus}):{ANSI_RESET}")
for sous_dossier_corpus in sous_dossiers_corpus:
    chemin_sous_dossier_corpus = os.path.join(chemin_dossier_corpus, sous_dossier_corpus)
    if not os.path.isdir(chemin_sous_dossier_corpus):
        print(f"    Le dossier {chemin_sous_dossier_corpus} n'existe pas.")
        continue

    compteur_documents_sous_dossier_corpus = 0
    print(f"    {ANSI_YELLOW}Parcours du dossier \"{sous_dossier_corpus}\" ({chemin_sous_dossier_corpus}):{ANSI_RESET}")
    for fichier in os.listdir(chemin_sous_dossier_corpus):
        chemin_fichier = os.path.join(chemin_sous_dossier_corpus, fichier)
        if os.path.isfile(chemin_fichier) and fichier.endswith(".txt"):
            compteur_documents_corpus += 1
            compteur_documents_sous_dossier_corpus += 1

            if nombre_max_documents_a_traiter is None or nombre_documents_traites < nombre_max_documents_a_traiter:
                traiter_document(chemin_fichier)
                nombre_documents_traites += 1
            
    print(f"    {ANSI_YELLOW}{compteur_documents_sous_dossier_corpus} documents rencontrés.{ANSI_RESET}")

print(f"{ANSI_GREEN}{nombre_documents_traites} documents traité(s) {ANSI_RESET}/{ANSI_YELLOW} {compteur_documents_corpus} documents rencontrés.{ANSI_RESET}", end="\n")

fin_traitement = time.time()
print(f"\n{ANSI_BLUE}Durée du traitement: {fin_traitement - debut_traitement:.2f} secondes{ANSI_RESET}")


print(f"{ANSI_GREEN}Relations extraites:{ANSI_RESET}")
for id_type_relation, liste_mots_en_relation in relations.items():
    print(f"   Relations de type '{correspondance_types_relation_id_nom[id_type_relation]}' ({len(liste_mots_en_relation)}):")
    nom_type_relation = correspondance_types_relation_id_nom[id_type_relation]
    for tuple_mots_en_relation in liste_mots_en_relation:
        print(f"        {ANSI_YELLOW}\"{tuple_mots_en_relation[0]}\" {nom_type_relation} \"{tuple_mots_en_relation[1]}\" {ANSI_RESET}")
