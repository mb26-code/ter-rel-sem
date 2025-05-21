import os
import re

# Dossier contenant les fichiers .txt
folder_path = 'wikipedia'

# Expressions à retirer
phrases_to_remove = [
    "Cet article est une ébauche concernant la cuisine.",
    "Sur les autres projets Wikimedia :",
    "Cet article ne cite pas suffisamment ses sources",
    "modifier Wikidata",
    "modifier le code",
    "Cet article ne s'appuie pas, ou pas assez, sur des sources secondaires ou tertiaires",
]

# Expression régulière pour les [{nombre}]
pattern_bracket_number = re.compile(r'\[\{\d+\}\]')

# Phrase signalant un fichier à supprimer
phrase_to_delete_file = "Aucun contenu trouvé sur cette page."

# Traitement de chaque fichier .txt
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Suppression du fichier si la phrase est trouvée
        if phrase_to_delete_file in content:
            os.remove(file_path)
            print(f"Supprimé : {filename}")
            continue

        # Nettoyage des phrases
        for phrase in phrases_to_remove:
            content = content.replace(phrase, '')

        # Suppression des [{nombre}]
        content = pattern_bracket_number.sub('', content)

        # Réécriture du fichier nettoyé
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

print("Traitement terminé.")
