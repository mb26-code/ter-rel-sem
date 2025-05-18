"""
Ce script permet d'effectuer un apprentissage sur le corpus.
C'est la première moitié de l'algorithme d'extraction de relations sémantiques.

Une fois lancé, ce script remplit la base de données associée à l'algorithme: 
il s'agit de la "mémoire" du processus d'extraction.
Pour cela, l'entièreté du corpus est traitée en faisant appel à l'API JDM pour déduire des 
patrons lexico-sémantiques permettant de trouver des relations.

Notes :
- Ce script suppose que le schéma de la base de données a déjà été défini.
- À chaque exécution, il efface tous les tuples de la BDD pour la remplir à partir de zéro.
- Il n'augmente pas la mémoire de l'algorithme, mais la construit entièrement.
- Une fois cette mémoire établie, on peut extraire des relations à partir de nouveaux textes.
"""

import os
import sys
from dotenv import load_dotenv

import psycopg2

import spacy
from itertools import combinations

import requests

from json import dumps
def print_JSON(obj):
    print(dumps(obj, indent = 4, ensure_ascii = False))

import time

ANSI_RED = "\033[31m"
ANSI_GREEN = "\033[32m"
ANSI_YELLOW = "\033[93m"
ANSI_BLUE = "\033[94m"
ANSI_RESET = "\033[0m"

print()

load_dotenv()
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

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

try:
    connexion_bdd = psycopg2.connect(
        user = DB_USER,
        password = DB_PASSWORD,
        host = DB_HOST,
        port = DB_PORT,
        dbname = DB_NAME
    )

    cursor = connexion_bdd.cursor()
    
    def bdd(instruction_SQL):
        try:
            cursor.execute(instruction_SQL)
            instruction_minuscules = instruction_SQL.strip().lower()
            
            renvoie_resultat = instruction_minuscules.startswith("select") or "returning" in instruction_minuscules
            resultat = cursor.fetchall() if renvoie_resultat else None
            
            connexion_bdd.commit()
            return resultat
        except Exception as e:
            raise e
    
    print(f"{ANSI_GREEN}Connecté à la BDD.{ANSI_RESET}", end="\n\n")

    #########################################################

    chemin_dossier_corpus = os.path.join(".", "corpus")
    sous_dossiers_corpus = ["wikipedia", "marmiton"]

    nlp = spacy.load("fr_core_news_lg")
    VALEURS_ATTRIBUTS_POS_TOKENS_CANDIDATS = {"ADJ", "NOUN", "PROPN"}
    DISTANCE_MAX_PAIRE_CANDIDATS = 4
    DEPS_SYNTAXIQUES_VALIDES_PAIRE_CANDIDATS = {"amod", "nmod", "compound", "obj", "obl", "attr", "appos"}
    DISTANCE_MAX_PREFIXE_SUFFIXE = 3

    DUREE_ATTENTE_APPEL_JDM_SECONDES = 0.11
    URL_API_JDM = "https://jdm-api.demo.lirmm.fr/"
    ENDPOINT_JDM_TYPES_RELATION = "v0/relations_types"
    ENDPOINT_PARAMETRE_JDM_RELATIONS_NOEUDS = "v0/relations/from/{}/to/{}"

    ###################

    print(f"{ANSI_YELLOW}Réinitialisation des tables de la BDD...{ANSI_RESET}")
    instructions_reinitialisation_BDD = """
    DELETE FROM CORRESPONDANCE_PATRONS;
    DELETE FROM OCCURRENCES_PATRONS;
    DELETE FROM RELATIONS_CORPUS;
    ALTER SEQUENCE public.relations_corpus_id_seq RESTART WITH 1;

    DELETE FROM PATRONS;
    ALTER SEQUENCE public.patrons_id_seq RESTART WITH 1;
    DELETE FROM TYPES_RELATION;
    """
    bdd(instructions_reinitialisation_BDD)
    print(f"{ANSI_GREEN}Tables réinitialisées.{ANSI_RESET}", end="\n\n")
    
    
    ###################

    def JDM_types_de_relation():
        requete_JDM = URL_API_JDM + ENDPOINT_JDM_TYPES_RELATION
        print(f"    Envoi d'une requête à l'API JDM : {requete_JDM}")
        reponse_JDM = requests.get(requete_JDM)

        if reponse_JDM.status_code == 200:
            relations_JDM = reponse_JDM.json()
            #print_JSON(relations_JDM)
            return relations_JDM
        else:
            print(f"Erreur #{reponse_JDM.status_code}: {reponse_JDM.text}")
            return None
    
    def charger_bdd_avec_types_de_relations():
        print(f"{ANSI_YELLOW}Remplissage de la table TYPES_RELATION...{ANSI_RESET}")
        instruction_SQL_insertion_types_relation = "INSERT INTO TYPES_RELATION (id, nom, description_) VALUES \n"
        nombre_types = 0
        template_tuple_SQL_type_relation = "({}, '{}', '{}'), \n"
        for type_de_relation in JDM_types_de_relation():
            nombre_types += 1
            id_type = type_de_relation["id"]
            nom_type = type_de_relation["name"].replace("'", "''")
            description_type = type_de_relation["help"].replace("'", "''")

            instruction_SQL_insertion_types_relation += template_tuple_SQL_type_relation.format(
                id_type, nom_type, description_type
            )
        instruction_SQL_insertion_types_relation = instruction_SQL_insertion_types_relation[:-3] + ";"
        
        #print(instruction_SQL_insertion_types)
        bdd(instruction_SQL_insertion_types_relation)
        print(f"{ANSI_GREEN}Table TYPES_RELATION remplie avec {nombre_types} types de relation.{ANSI_RESET}")
    
    charger_bdd_avec_types_de_relations()

    def JDM_relations_entre_mots(mot_1, mot_2):
        requete_JDM = URL_API_JDM + ENDPOINT_PARAMETRE_JDM_RELATIONS_NOEUDS.format(mot_1, mot_2)
        #print(f"    Envoi d'une requête à l'API JDM : {requete_JDM}")
        reponse_JDM = requests.get(requete_JDM)

        if reponse_JDM.status_code == 200:
            resultat_JDM = reponse_JDM.json()
            #print_JSON(resultat_JDM)
            return resultat_JDM["relations"]
        else:
            #print(reponse_JDM.text)
            return None

    #print_JSON(JDM_relations_entre_mots("eau", "lait"))

    template_expression_patron = "{} [{}] {} [{}] {}"

    #######
    template_instruction_SQL_existence_patron = """
    SELECT id FROM PATRONS
    WHERE prefixe = '{}' AND
          type_entite1 = '{}' AND
          infixe = '{}' AND
          type_entite2 = '{}' AND
          suffixe = '{}';
    """

    template_instruction_SQL_insertion_patron = """
    INSERT INTO PATRONS (prefixe, type_entite1, infixe, type_entite2, suffixe, expression_complete) 
    VALUES ('{}', '{}', '{}', '{}', '{}', '{}') RETURNING id;
    """

    debut_instruction_SQL_insertion_relation_corpus = """
    INSERT INTO RELATIONS_CORPUS (
        id_type_relation, entite1, entite2, id_patron, extrait, reference
    ) VALUES \n
    """
    template_tuple_SQL_relation_corpus = "({}, '{}', '{}', {}, '{}', '{}'), \n"

    template_instruction_SQL_existence_occurence_patron = """
    SELECT occurences FROM OCCURRENCES_PATRONS
    WHERE id_patron = {} AND id_type_relation = {};
    """

    template_instruction_SQL_incrementation_occurence_patron = """
    UPDATE OCCURRENCES_PATRONS SET occurences = occurences + 1
    WHERE id_patron = {} AND id_type_relation = {};
    """

    template_instruction_SQL_insertion_occurence_patron = """
    INSERT INTO OCCURRENCES_PATRONS (id_patron, id_type_relation, occurences)
        VALUES ({}, {}, 1);
    """

    def extraire_prefixe(phrase, token_1):
        tokens = []
        for tok in reversed(phrase[:token_1.i - phrase.start]):
            if tok.is_punct or tok.pos_ in {"DET", "CCONJ", "SCONJ"}:
                break
            tokens.insert(0, tok.lemma_)
            if len(tokens) >= DISTANCE_MAX_PREFIXE_SUFFIXE:
                break
        return " ".join(tokens).strip()

    def extraire_infixe(phrase, token_1, token_2):
        tokens = phrase[token_1.i - phrase.start + 1 : token_2.i - phrase.start]
        return " ".join([tok.lemma_ for tok in tokens if not tok.is_punct]).strip()

    def extraire_suffixe(phrase, token_2):
        tokens = []
        for tok in phrase[token_2.i - phrase.start + 1 :]:
            if tok.is_punct or tok.pos_ in {"VERB", "AUX", "DET", "CCONJ", "SCONJ"}:
                break
            tokens.append(tok.lemma_)
            if len(tokens) >= DISTANCE_MAX_PREFIXE_SUFFIXE:
                break
        return " ".join(tokens).strip()


    def traiter_document(chemin_fichier_texte):
        print(f"{ANSI_YELLOW}        Traitement du document \"{chemin_fichier_texte}\":{ANSI_RESET}")
        with open(chemin_fichier_texte, "r", encoding="utf-8") as f:
            texte = f.read()
            print(f"        \"{texte}\"", end = "\n\n")
            doc = nlp(texte)

            for phrase in doc.sents:
                print(f"            Phrase: \"{phrase}\"")

                tokens_candidats = [token for token in phrase if token.pos_ in VALEURS_ATTRIBUTS_POS_TOKENS_CANDIDATS]

                #######################

                for candidat_1, candidat_2 in combinations(tokens_candidats, 2):
                    
                    if abs(candidat_1.i - candidat_2.i) > DISTANCE_MAX_PAIRE_CANDIDATS:
                        continue

                    #lien syntaxique (direct)?
                    liaison_syntaxique_existante = (
                        (candidat_2 in candidat_1.children and candidat_2.dep_ in DEPS_SYNTAXIQUES_VALIDES_PAIRE_CANDIDATS) or
                        (candidat_1 in candidat_2.children and candidat_1.dep_ in DEPS_SYNTAXIQUES_VALIDES_PAIRE_CANDIDATS) or
                        (candidat_1.head == candidat_2 and candidat_1.dep_ in DEPS_SYNTAXIQUES_VALIDES_PAIRE_CANDIDATS) or
                        (candidat_2.head == candidat_1 and candidat_2.dep_ in DEPS_SYNTAXIQUES_VALIDES_PAIRE_CANDIDATS)
                    )
                    if not liaison_syntaxique_existante:
                        continue

                    print(f"                Paire de tokens candidats considérée: {ANSI_YELLOW}{candidat_1, candidat_2}{ANSI_RESET}")

                    #vérifier si il existe des relations connues par JDM entre les mots candidats
                    time.sleep(DUREE_ATTENTE_APPEL_JDM_SECONDES)
                    relations_candidats = JDM_relations_entre_mots(candidat_1.lemma_, candidat_2.lemma_)

                    if not relations_candidats:
                        print(f"                    Pas de relation connue.")
                    else:
                        relations_connues_JDM = [(rel["type"], rel["w"]) for rel in relations_candidats]
                        print(f"                    Relations connue(s): {relations_connues_JDM}")
                        #si il y a des relations connues entre candidats, alors on relève le patron qui lie les candidats
                        
                        
                        pos_candidat_1 = candidat_1.pos_
                        pos_candidat_2 = candidat_2.pos_

                        prefixe = extraire_prefixe(phrase, candidat_1)
                        infixe = extraire_infixe(phrase, candidat_1, candidat_2)
                        suffixe = extraire_suffixe(phrase, candidat_2)



                        expression_complete = template_expression_patron.format(
                            prefixe, pos_candidat_1, infixe, pos_candidat_2, suffixe
                        )
                        print(f"                    Patron relevé: {ANSI_BLUE}\"{expression_complete}\"{ANSI_RESET} ", end="")

                        prefixe_sql = prefixe.replace("'", "''")
                        infixe_sql = infixe.replace("'", "''")
                        suffixe_sql = suffixe.replace("'", "''")
                        expression_complete_sql = expression_complete.replace("'", "''")
                            
                        instruction_SQL_existence_patron = template_instruction_SQL_existence_patron.format(
                            prefixe_sql, pos_candidat_1, infixe_sql, pos_candidat_2, suffixe_sql, expression_complete_sql
                        )

                        resultat_existence_patron = bdd(instruction_SQL_existence_patron)

                        if resultat_existence_patron:
                            id_patron = resultat_existence_patron[0][0]
                            print(f"({ANSI_YELLOW}déjà connu{ANSI_RESET} : {ANSI_BLUE}id = {id_patron}{ANSI_RESET})")
                        else:
                            instruction_SQL_insertion_patron = template_instruction_SQL_insertion_patron.format(
                                prefixe_sql, pos_candidat_1, infixe_sql, pos_candidat_2, suffixe_sql, expression_complete_sql
                            )
                            #print(instruction_SQL_insertion_patron)
                            resultat_insertion = bdd(instruction_SQL_insertion_patron)
                            
                            id_patron = resultat_insertion[0][0]
                            print(f"({ANSI_GREEN}nouveau{ANSI_RESET} : {ANSI_BLUE}id = {id_patron}{ANSI_RESET})")

                        #########

                        instruction_SQL_insertion_relation_corpus = debut_instruction_SQL_insertion_relation_corpus
                        for relation in relations_candidats:
                            ## pour remplir la table RELATIONS_CORPUS
                            extrait = doc[candidat_1.i - DISTANCE_MAX_PREFIXE_SUFFIXE:candidat_2.i + DISTANCE_MAX_PREFIXE_SUFFIXE].text
                            instruction_SQL_insertion_relation_corpus += template_tuple_SQL_relation_corpus.format(
                                relation["type"], candidat_1.lemma_.replace("'", "''"), candidat_2.lemma_.replace("'", "''"), 
                                id_patron, extrait.replace("'", "''"), chemin_fichier_texte.replace("'", "''") + f" #{candidat_1.i}"
                            )
                            ## pour remplir la table OCCURRENCES_PATRONS
                            instruction_SQL_existence_occurence_patron = template_instruction_SQL_existence_occurence_patron.format(
                                id_patron, relation["type"]
                            )
                            resultat_occurence_patron = bdd(instruction_SQL_existence_occurence_patron)

                            if resultat_occurence_patron:
                                instruction_SQL_incrementation_occurence_patron = template_instruction_SQL_incrementation_occurence_patron.format(
                                    id_patron, relation["type"]
                                )
                                bdd(instruction_SQL_incrementation_occurence_patron)
                            else:
                                instruction_SQL_insertion_occurence_patron = template_instruction_SQL_insertion_occurence_patron.format(
                                    id_patron, relation["type"]
                                )
                                bdd(instruction_SQL_insertion_occurence_patron)
                            ##
                        instruction_SQL_insertion_relation_corpus = instruction_SQL_insertion_relation_corpus[:-3] + ";"
                        
                        #print(instruction_SQL_insertion_relation_corpus)
                        bdd(instruction_SQL_insertion_relation_corpus)
                
    print(
        """
        --------------------------------
        |          TRAITEMENT          |
        --------------------------------
        """
    )
    
    debut_traitement = time.time()
    #Test sur 1 document
    #traiter_document("./corpus/wikipedia/Poulet au citron.txt")

    compteur_documents_corpus = 0
    nombre_documents_traites = 0
    print(f"\nParcours du corpus ({chemin_dossier_corpus}) :")
    for sous_dossier_corpus in sous_dossiers_corpus:
        chemin_sous_dossier_corpus = os.path.join(chemin_dossier_corpus, sous_dossier_corpus)
        if not os.path.isdir(chemin_sous_dossier_corpus):
            print(f"    Le dossier {chemin_sous_dossier_corpus} n'existe pas.")
            continue

        compteur_documents_sous_dossier_corpus = 0
        print(f"    Parcours du dossier \"{sous_dossier_corpus}\" ({chemin_sous_dossier_corpus}) :")
        for fichier in os.listdir(chemin_sous_dossier_corpus):
            chemin_fichier = os.path.join(chemin_sous_dossier_corpus, fichier)
            if os.path.isfile(chemin_fichier) and fichier.endswith(".txt"):
                compteur_documents_corpus += 1
                compteur_documents_sous_dossier_corpus += 1

                if nombre_max_documents_a_traiter is None or nombre_documents_traites < nombre_max_documents_a_traiter:
                    traiter_document(chemin_fichier)
                    nombre_documents_traites += 1
                
        print(f"\n    {ANSI_YELLOW}{compteur_documents_sous_dossier_corpus} documents rencontrés.{ANSI_RESET}")
    print(f"{ANSI_GREEN}{nombre_documents_traites} documents traité(s) {ANSI_RESET}/{ANSI_YELLOW} {compteur_documents_corpus} documents rencontrés.{ANSI_RESET}", end="\n")

    fin_traitement = time.time()
    print(f"\n{ANSI_BLUE}Durée du traitement: {fin_traitement - debut_traitement:.2f} secondes{ANSI_RESET}")

    ########################################################

    requete_SQL_paires_patron_type_relation_frequents = """
    SELECT 
        P.expression_complete,
        T.nom AS type_relation,
        OP.occurences
    FROM 
        OCCURRENCES_PATRONS OP
    JOIN 
        PATRONS P ON OP.id_patron = P.id
    JOIN 
        TYPES_RELATION T ON OP.id_type_relation = T.id
    WHERE 
        OP.occurences >= 10
    ORDER BY 
        OP.occurences DESC;
    """

    resultat_paires_patron_type_relation_frequents = bdd(requete_SQL_paires_patron_type_relation_frequents)

    print("\nPaires (patron, type de relation) les plus fréquentes dans le(s) document(s) traité(s):")
    for ligne in resultat_paires_patron_type_relation_frequents:
        expression_complete, type_relation, occurences = ligne
        print(f"    - {ANSI_BLUE}\"{expression_complete}\"{ANSI_RESET} | {ANSI_YELLOW}\"{type_relation}\"{ANSI_RESET} | Occurrences: {occurences}")

    ##
    requete_SQL_patrons_frequents = """
    SELECT 
        P.expression_complete,
        SUM(OP.occurences) AS total_occurences
    FROM 
        OCCURRENCES_PATRONS OP
    JOIN 
        PATRONS P ON OP.id_patron = P.id
    GROUP BY 
        P.expression_complete
    ORDER BY 
        total_occurences DESC
    LIMIT 100;
    """

    resultat_patrons_frequents = bdd(requete_SQL_patrons_frequents)

    print("\nLes 100 patrons les plus fréquents (tous types de relation confondus) dans le(s) document(s) traité(s):")
    for ligne in resultat_patrons_frequents:
        expression_complete, total_occurences = ligne
        print(f"    - {ANSI_BLUE}\"{expression_complete}\"{ANSI_RESET} | Occurrences: {total_occurences}")

    
    cursor.close()
    connexion_bdd.close()
    print("\nDéconnecté de la BDD.")

except Exception as e:
    raise e

print()