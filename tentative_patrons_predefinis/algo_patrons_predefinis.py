from collections import Counter
import re
import time
import sys
import os

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

chemin_dossier_corpus = os.path.join("./", "corpus")
sous_dossiers_corpus = ["wikipedia", "marmiton"]

def extraire_relations_typees(textes):
    relations = {
        'ingredient_de': [],
        'partie_de': [],
        'type_de': [],
        'utilise_pour': []
    }
    
    patterns = { }

    patterns['ingredient_de'] = [
    r"([\w\s\-']+?) est un ingrédient(?: principal)? de ([\w\s\-']+)",
    r"([\w\s\-']+?) entre dans la composition de ([\w\s\-']+)",
    r"([\w\s\-']+?) est utilisé(?:e)? dans ([\w\s\-']+)",
    r"([\w\s\-']+?) (?:entre|entrent) dans la préparation de ([\w\s\-']+)",
    r"([\w\s\-']+?) est nécessaire pour préparer ([\w\s\-']+)",
    r"([\w\s\-']+?) est ajouté(?:e)? à ([\w\s\-']+)",
    r"([\w\s\-']+?) est l'un des ingrédients de ([\w\s\-']+)",
    r"([lL]es [\w\s\-']+?) sont utilisés dans ([\w\s\-']+)",
    r"([lL]e [\w\s\-']+?) est présent dans ([\w\s\-']+)",
    r"([\w\s\-']+?) contient ([\w\s\-']+)",  # l'inverse
    ]

    patterns['partie_de'] = [
    r"([\w\s\-']+?) fait partie de ([\w\s\-']+)",
    r"([\w\s\-']+?) est une étape de ([\w\s\-']+)",
    r"([\w\s\-']+?) constitue une partie de ([\w\s\-']+)",
    r"([\w\s\-']+?) est inclus(?:e)? dans ([\w\s\-']+)",
    r"([\w\s\-']+?) est l'une des étapes de ([\w\s\-']+)",
    r"([\w\s\-']+?) appartient à ([\w\s\-']+)",
    ]

    patterns['type_de'] = [
    r"([\w\s\-']+?) est un type de ([\w\s\-']+)",
    r"([\w\s\-']+?) est une variété de ([\w\s\-']+)",
    r"([\w\s\-']+?) est un genre de ([\w\s\-']+)",
    r"([\w\s\-']+?) est une sorte de ([\w\s\-']+)",
    r"([\w\s\-']+?) est une spécialité de ([\w\s\-']+)",
    r"([\w\s\-']+?) est une forme de ([\w\s\-']+)",
    r"([\w\s\-']+?) est une déclinaison de ([\w\s\-']+)",
    r"([\w\s\-']+?) est une recette de ([\w\s\-']+)",
    ]

    patterns['utilise_pour'] = [
    r"([\w\s\-']+?) est utilisé(?:e)? pour ([\w\s\-']+)",
    r"([\w\s\-']+?) sert à ([\w\s\-']+)",
    r"([\w\s\-']+?) permet de ([\w\s\-']+)",
    r"([\w\s\-']+?) est nécessaire pour ([\w\s\-']+)",
    r"([\w\s\-']+?) est employé(?:e)? pour ([\w\s\-']+)",
    r"([\w\s\-']+?) est utile pour ([\w\s\-']+)",
    r"([\w\s\-']+?) est destiné(?:e)? à ([\w\s\-']+)",
    ]



    
    for texte in textes:
        for type_rel, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.findall(pattern, texte)
                for match in matches:
                    if len(match) == 3:
                        relations[type_rel].append((match[0].strip(), match[2].strip()))
                    else:
                        relations[type_rel].append((match[0].strip(), match[1].strip()))
    
    for type_rel in relations:
        compteur = Counter(relations[type_rel])
        relations[type_rel] = [(pair[0], pair[1]) for pair, count in compteur.items() if count >= 2]
        
    return relations

print(
    """
    --------------------------------
    |          TRAITEMENT          |
    --------------------------------
    """
)

debut_traitement = time.time()
compteur_documents_corpus = 0
nombre_documents_traites = 0
textes_a_traiter = []

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
                with open(chemin_fichier, "r", encoding="utf-8") as f:
                    texte = f.read()
                    textes_a_traiter.append(texte)
                nombre_documents_traites += 1
            
    print(f"\n    {ANSI_YELLOW}{compteur_documents_sous_dossier_corpus} documents rencontrés.{ANSI_RESET}")

print(f"{ANSI_GREEN}{nombre_documents_traites} documents traité(s) {ANSI_RESET}/{ANSI_YELLOW} {compteur_documents_corpus} documents rencontrés.{ANSI_RESET}", end="\n")

for i, texte in enumerate(textes_a_traiter):
    print(f"\n--- Texte {i+1} (longueur: {len(texte)}) ---")
    print(texte[:300])  # Affiche les 300 premiers caractères



# Appel à la fonction de traitement des relations
relations_extraites = extraire_relations_typees(textes_a_traiter)

# Affichage des relations trouvées
for type_rel, liste_rel in relations_extraites.items():
    print(f"\n{ANSI_BLUE}Relations de type '{type_rel}' ({len(liste_rel)} occurrences) :{ANSI_RESET}")
    for e1, e2 in liste_rel:
        print(f"  - {e1} --> {e2}")

fin_traitement = time.time()
print(f"\n{ANSI_BLUE}Durée du traitement: {fin_traitement - debut_traitement:.2f} secondes{ANSI_RESET}")
