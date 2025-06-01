#!/usr/bin/env python3
"""
Script pour enrichir semantic_relations_import.csv avec les données de recipe_relations_refined.csv
Ajoute deux nouvelles colonnes : new_relation et new_relation_w
"""

import csv
import os
from typing import Dict, Tuple, Optional

def load_recipe_relations(file_path: str) -> Dict[Tuple[str, str], Tuple[str, str]]:
    """
    Charge les relations de recettes dans un dictionnaire pour une recherche rapide.
    
    Args:
        file_path: Chemin vers recipe_relations_refined.csv
        
    Returns:
        Dictionnaire mappant (node1, node2) vers (relation, w)
    """
    recipe_relations = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Ignore les lignes de commentaires
            if row['node1'].startswith('//'):
                continue
                
            key = (row['node1'].strip(), row['node2'].strip())
            value = (row['relation'].strip(), row['w'].strip())
            recipe_relations[key] = value
    
    print(f"Chargé {len(recipe_relations)} relations de recettes")
    return recipe_relations

def process_semantic_relations(input_file: str, output_file: str, recipe_relations: Dict[Tuple[str, str], Tuple[str, str]]):
    """
    Traite les relations sémantiques et ajoute de nouvelles colonnes, plus ajoute les relations de recettes manquantes.
    
    Args:
        input_file: Chemin vers semantic_relations_import.csv
        output_file: Chemin vers le fichier de sortie
        recipe_relations: Dictionnaire des relations de recettes
    """
    found_count = 0
    not_found_count = 0
    processed_pairs = set()
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        
        reader = csv.DictReader(infile)
        
        # Ajoute de nouvelles colonnes aux noms de champs
        fieldnames = list(reader.fieldnames) + ['new_relation', 'new_relation_w']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        
        # Écrit l'en-tête
        writer.writeheader()
        
        # Traite les relations sémantiques existantes
        for row in reader:
            # Ignore les lignes de commentaires
            if row['node1'].startswith('//'):
                continue
            
            node1 = row['node1'].strip()
            node2 = row['node2'].strip()
            key = (node1, node2)
            processed_pairs.add(key)
            
            # Recherche dans les relations de recettes
            if key in recipe_relations:
                relation, w = recipe_relations[key]
                row['new_relation'] = relation
                row['new_relation_w'] = w
                found_count += 1
            else:
                # Non trouvé, laisse vide
                row['new_relation'] = ''
                row['new_relation_w'] = ''
                not_found_count += 1
            
            writer.writerow(row)
        
        # Ajoute les relations de recettes manquantes qui n'étaient pas dans les relations sémantiques
        added_count = 0
        for (node1, node2), (relation, w) in recipe_relations.items():
            key = (node1, node2)
            if key not in processed_pairs:
                # Crée une nouvelle ligne avec des colonnes vides sauf pour node1, node2, new_relation, new_relation_w
                new_row = {field: '' for field in fieldnames}
                new_row['node1'] = node1
                new_row['node2'] = node2
                new_row['new_relation'] = relation
                new_row['new_relation_w'] = w
                writer.writerow(new_row)
                added_count += 1
    
    print(f"Traitement terminé :")
    print(f"  - Correspondances trouvées : {found_count}")
    print(f"  - Non trouvées : {not_found_count}")
    print(f"  - Nouvelles entrées ajoutées depuis les relations de recettes : {added_count}")
    print(f"  - Total traité : {found_count + not_found_count + added_count}")

def main():
    """Fonction principale pour exécuter le processus d'enrichissement."""
    # Définit les chemins des fichiers
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Fichiers d'entrée
    semantic_relations_file = os.path.join(base_dir, '..', 'semantic_relations_import.csv')
    recipe_relations_file = os.path.join(base_dir, 'recipe_relations_refined.csv')
    
    # Fichier de sortie
    output_file = os.path.join(base_dir, 'semantic_relations_enriched.csv')
    
    # Vérifie si les fichiers d'entrée existent
    if not os.path.exists(semantic_relations_file):
        print(f"Erreur : {semantic_relations_file} introuvable")
        return
    
    if not os.path.exists(recipe_relations_file):
        print(f"Erreur : {recipe_relations_file} introuvable")
        return
    
    print("Démarrage du processus d'enrichissement...")
    print(f"Relations sémantiques d'entrée : {semantic_relations_file}")
    print(f"Relations de recettes d'entrée : {recipe_relations_file}")
    print(f"Fichier de sortie : {output_file}")
    
    # Charge les relations de recettes
    recipe_relations = load_recipe_relations(recipe_relations_file)
    
    # Traite les relations sémantiques
    process_semantic_relations(semantic_relations_file, output_file, recipe_relations)
    
    print(f"Données enrichies écrites dans : {output_file}")

if __name__ == "__main__":
    main()