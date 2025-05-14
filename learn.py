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
from dotenv import load_dotenv

import psycopg2

import spacy

import requests

from json import dumps
def print_JSON(obj):
    print(dumps(obj, indent = 4, ensure_ascii = False))

import time

print()

load_dotenv()
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

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
    
    print("Connecté à la BDD.", end="\n\n")

    ####################################################################

    chemin_dossier_corpus = os.path.join(".", "corpus")
    sous_dossiers_corpus = ["marmiton", "wikipedia"]

    nlp = spacy.load("fr_core_news_lg")
    VALEURS_ATTRIBUTS_POS_TOKENS_CANDIDATS = {"ADJ", "NOUN", "NUM", "PROPN", "VERB", "X"}
    DISTANCE_MAX_PAIRE_TOKENS_CANDIDATS = 3
    DISTANCE_MAX_PREFIXE_SUFFIXE = 2

    URL_API_JDM = "https://jdm-api.demo.lirmm.fr/"
    ENDPOINT_JDM_TYPES_RELATION = "v0/relations_types"
    ENDPOINT_PARAMETRE_JDM_RELATIONS_NOEUDS = "v0/relations/from/{}/to/{}"

    ###################

    print("Réinitialisation des tables de la BDD...")
    instructions_suppression_tuples = """
    DELETE FROM CORRESPONDANCE_PATRONS;
    DELETE FROM OCCURRENCES_PATRONS;
    DELETE FROM RELATIONS_CORPUS;
    ALTER SEQUENCE public.relations_corpus_id_seq RESTART WITH 1;

    DELETE FROM PATRONS;
    ALTER SEQUENCE public.patrons_id_seq RESTART WITH 1;
    DELETE FROM TYPES_RELATION;
    """
    bdd(instructions_suppression_tuples)
    print("Tables réinitialisées.")
    print()
    
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
            print(reponse_JDM.text)
            return None
    
    def charger_bdd_avec_types_de_relations():
        print("Remplissage de la table TYPES_RELATION...")
        instruction_SQL_insertion_types = "INSERT INTO TYPES_RELATION (id, nom, description_) VALUES \n"
        nombre_types = 0
        tuple_SQL_type = "({}, '{}', '{}'), \n"
        for type_de_relation in JDM_types_de_relation():
            nombre_types += 1
            id_type = type_de_relation["id"]
            nom_type = type_de_relation["name"].replace("'", "''")
            description_type = type_de_relation["help"].replace("'", "''")

            instruction_SQL_insertion_types += tuple_SQL_type.format(id_type, nom_type, description_type)
        instruction_SQL_insertion_types = instruction_SQL_insertion_types[:-3] + ";"
        
        #print(instruction_SQL_insertion_types)
        bdd(instruction_SQL_insertion_types)
        print(f"Table TYPES_RELATION remplie avec {nombre_types} types de relation.")
    
    charger_bdd_avec_types_de_relations()
    print()

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
    print()

    modele_expression_patron = "{} [{}] {} [{}] {}"
    modele_instruction_SQL_existence_patron = """
    SELECT id FROM PATRONS
    WHERE prefixe = '{}' AND
          type_entite1 = '{}' AND
          infixe = '{}' AND
          type_entite2 = '{}' AND
          suffixe = '{}' AND
          expression_complete = '{}';
    """
    modele_instruction_SQL_insertion_patron = """
    INSERT INTO PATRONS (prefixe, type_entite1, infixe, type_entite2, suffixe, expression_complete) 
    VALUES ('{}', '{}', '{}', '{}', '{}', '{}') RETURNING id;
    """

    def traiter_document(chemin_fichier_texte):
        
        with open(chemin_fichier_texte, "r", encoding="utf-8") as f:
            texte = f.read()
            print(f"\"{texte}\"", end = "\n\n")
            doc = nlp(texte)

            for phrase in doc.sents:
                tokens_candidats = [token for token in phrase if token.pos_ in VALEURS_ATTRIBUTS_POS_TOKENS_CANDIDATS]

                print(f" - PHRASE: \"{phrase}\"")
                for i, token_candidat_1 in enumerate(tokens_candidats):
                    for j in range(i + 1, len(tokens_candidats)):
                        token_candidat_2 = tokens_candidats[j]

                        if abs(token_candidat_1.i - token_candidat_2.i) > DISTANCE_MAX_PAIRE_TOKENS_CANDIDATS:
                            break  #si les mots sont trops éloignés, on arrête d'itérer sur le second token possible
                        
                        print(f"{token_candidat_1, token_candidat_2} / {token_candidat_1.lemma_, token_candidat_2.lemma_}")
                        #Vérifier si il existe des relations connues par JDM entre les mots candidats
                        time.sleep(0.2)
                        relations_candidats = JDM_relations_entre_mots(token_candidat_1.lemma_, token_candidat_2.lemma_)

                        if relations_candidats:
                            #si il y a des relations connues entre candidats, alors on extrait le patron rencontré
                            
                            #valeurs d'attributs POS
                            pos_candidat_1 = token_candidat_1.pos_
                            pos_candidat_2 = token_candidat_2.pos_

                            #préfixe = chaîne composée des tokens lemmatisés entre le début de la phrase et token_candidat_1 (exclus)
                            debut_prefixe = max(0, token_candidat_1.i - phrase.start - DISTANCE_MAX_PREFIXE_SUFFIXE)
                            tokens_prefixe = phrase[debut_prefixe : token_candidat_1.i - phrase.start]
                            prefixe = " ".join(token.lemma_ for token in tokens_prefixe).strip()

                            #infixe = chaîne composée des tokens lemmatisés entre les deux candidats
                            tokens_infixe = phrase[token_candidat_1.i - phrase.start + 1 : token_candidat_2.i - phrase.start]
                            infixe = " ".join(token.lemma_ for token in tokens_infixe).strip()

                            #suffixe = chaîne composée des tokens lemmatisés après token_candidat_2 jusqu'à la fin de la phrase
                            fin_suffixe = min(len(phrase) - 1, token_candidat_2.i - phrase.start + 1 + DISTANCE_MAX_PREFIXE_SUFFIXE)
                            tokens_suffixe = phrase[token_candidat_2.i - phrase.start + 1 : fin_suffixe]
                            suffixe = " ".join(token.lemma_ for token in tokens_suffixe).strip()


                            expression_complete = modele_expression_patron.format(
                                prefixe, pos_candidat_1, infixe, pos_candidat_2, suffixe
                            )
                            print(expression_complete)

                            prefixe_sql = prefixe.replace("'", "''")
                            infixe_sql = infixe.replace("'", "''")
                            suffixe_sql = suffixe.replace("'", "''")
                            expression_complete_sql = expression_complete.replace("'", "''")
                            
                            instruction_SQL_existence_patron = modele_instruction_SQL_existence_patron.format(
                                prefixe_sql, pos_candidat_1, infixe_sql, pos_candidat_2, suffixe_sql, expression_complete_sql
                            )

                            resultat_existence_patron = bdd(instruction_SQL_existence_patron)

                            if resultat_existence_patron:
                                id_patron = resultat_existence_patron[0][0]
                                print(f"Patron déjà existant: id = {id_patron}")
                            else:
                                instruction_SQL_insertion_patron = modele_instruction_SQL_insertion_patron.format(
                                    prefixe_sql, pos_candidat_1, infixe_sql, pos_candidat_2, suffixe_sql, expression_complete_sql
                                )
                                #print(instruction_SQL_insertion_patron)
                                resultat_insertion = bdd(instruction_SQL_insertion_patron)
                                #print(resultat_insertion)
                                id_patron = resultat_insertion[0][0]
                                print(f"Nouveau patron inséré: id = {id_patron}")

    
    #traiter_document("./document_test.txt")
    traiter_document("./corpus/wikipedia/Poulet au citron.txt")

    compteur_documents_corpus = 0
    print(f"Parcours du corpus ({chemin_dossier_corpus}) :")
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
                #traiter_document(chemin_fichier)
                compteur_documents_corpus += 1
                compteur_documents_sous_dossier_corpus += 1
        print(f"    {compteur_documents_sous_dossier_corpus} documents traités.")
    print(f"{compteur_documents_corpus} documents traités.", end="\n\n")


    ####################################################################
    
    cursor.close()
    connexion_bdd.close()
    print("Connexion BDD terminée.")

except Exception as e:
    raise e

print()