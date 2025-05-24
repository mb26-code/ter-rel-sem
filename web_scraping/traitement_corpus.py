import os
import re

def supprimer_dates_parentheses(dossier):
    pattern = re.compile(
        r'\(\s*(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+\d{4}\s*\)',
        flags=re.IGNORECASE
    )

    for nom_fichier in os.listdir(dossier):
        if nom_fichier.endswith('.txt'):
            chemin_fichier = os.path.join(dossier, nom_fichier)

            with open(chemin_fichier, 'r', encoding='utf-8') as f:
                contenu = f.read()

            contenu_modifie = re.sub(pattern, '', contenu)

            with open(chemin_fichier, 'w', encoding='utf-8') as f:
                f.write(contenu_modifie)

def nettoyer_fichiers_txt(dossier):
    termes_a_exclure = [
        "cet article",
        "wikipédia",
        "mise en forme",
        "points d'amélioration",
        "merci de consulter",
        "wikification",
        "articles homonymes",
        "homonyme",
        "homophone",
        "wikidata",
        "cette page est",
        "page d’aide",
        "cette page contient",
        "youtube",
        "consulté le",
        "proposition de fusion",
        "venez d'apposer",
        "bandeau",
        "utilisez ce texte",
        "à fusionner",
        "important :",
        "créer la section",
        "{{",
        "compléter l'article",
        "contributeurs principaux",
        "aider à l'améliorer",
        "modifier -",
        "modifier"
    ]
    
    # Regex pour exclure les caractères non-latins (arabe, chinois, russe, etc.), mais garder les accents latins.
    non_latin_pattern = re.compile(r'[^\x00-\x7F\u00C0-\u00FF\u0100-\u017F]+')

    for nom_fichier in os.listdir(dossier):
        if nom_fichier.endswith('.txt'):
            chemin_fichier = os.path.join(dossier, nom_fichier)

            with open(chemin_fichier, 'r', encoding='utf-8') as f:
                lignes = f.readlines()

            lignes_nettoyees = []
            for ligne in lignes:
                ligne_strip = ligne.strip()

                if not ligne_strip or ligne_strip == ".":
                    continue

                if ligne_strip.lower().startswith("titre"):
                    continue  

                ligne_minuscule = ligne_strip.lower()
                if any(terme in ligne_minuscule for terme in termes_a_exclure):
                    continue  

                ligne_sans_references = re.sub(r'\[\d+\]', '', ligne)
                ligne_sans_references = re.sub(r'\[réf\.[^\]]*\]', '', ligne)

                ligne_finale = re.sub(r'\s{2,}', ' ', ligne_sans_references).strip()

                ligne_finale = re.sub(non_latin_pattern, '', ligne_finale)

                if ligne_finale:  
                    lignes_nettoyees.append(ligne_finale + '\n')

            with open(chemin_fichier, 'w', encoding='utf-8') as f:
                f.writelines(lignes_nettoyees)


dossier = './corpus/wikipedia/'
nettoyer_fichiers_txt(dossier)
supprimer_dates_parentheses(dossier)
