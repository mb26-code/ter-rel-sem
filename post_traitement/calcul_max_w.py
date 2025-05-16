#!/usr/bin/env python3
import json
import csv
import ast
import os
import sys

# Ce fichier est pour rajouter la colonne `relation` et `w`
# en calculant leurs `w`

SOURCE_DIR = "marmiton"  # Changez ceci pour cibler différents ensembles de données

def select_best_relation(relations, relation_name_map):
    """
    Sélectionne la meilleure relation à partir d'une liste de relations en utilisant une approche sophistiquée.
    
    Cette fonction implémente plusieurs stratégies pour la sélection des relations:
    1. S'il n'y a pas de relations, retourne des valeurs vides
    2. S'il n'y a qu'une seule relation, utilise celle-ci
    3. S'il y a plusieurs relations, utilise une approche pondérée qui considère:
       - Le poids de la relation
       - La priorité de certains types de relations (carte de priorité prédéfinie)
    
    Args:
        relations (list): Liste des dictionnaires de relations contenant les clés 'type' et 'w'
        relation_name_map (dict): Correspondance entre l'ID du type de relation et le nom de la relation
    
    Returns:
        tuple: (relation_name, weight, change_info) où change_info est un dict avec des données sur les changements de relation
    """
    if not relations or len(relations) == 0:
        return '', '', None
    
    # S'il n'y a qu'une seule relation, on l'utilise
    if len(relations) == 1:
        rel = relations[0]
        rel_type = rel.get('type')
        weight = float(rel.get('w', 0))
        relation_name = relation_name_map.get(rel_type, f"unknown_{rel_type}")
        return relation_name, str(weight), None
    
    # Définit un multiplicateur de priorité pour différents types de relations
    # Des valeurs plus élevées signifient une priorité plus haute
    # Vu que r_associated est souvent donné le poids le plus haut, on la garde 1.0 et augmente les autres
    relation_priority = {
        # 0: 1.05,      # r_associated - Les concepts associés ont une priorité plus élevée
        1: 1.05,      # r_raff_sem - Raffinement sémantique vers un usage particulier du terme source
        6: 1.05,      # r_isa - Les relations taxonomiques sont importantes
        8: 1.05,      # r_hypo - Il est demandé d'énumérer des SPECIFIQUES/hyponymes du terme. Par exemple, 'mouche', 'abeille', 'guêpe' pour 'insecte'
        9: 1.05,      # r_has_part - Les relations partie-tout sont importantes
        10: 1.05,     # r_holo - Il est démandé d'énumérer des 'TOUT' (a pour holonymes)  de l'objet en question. Pour 'main', on aura 'bras', 'corps', 'personne', etc... Le tout est aussi l'ensemble comme 'classe' pour 'élève'.
        16: 1.05,     # r_instr - L'instrument est l'objet avec lequel on fait l'action. Dans - Il mange sa salade avec une fourchette -, fourchette est l'instrument. Des instruments typiques de 'tuer' peuvent être 'arme', 'pistolet', 'poison', ... (couper r_instr couteau)
        17: 1.05,     # r_carac - Les relations caractéristiques sont importantes
        75: 1.05,     # r_accomp - Est souvent accompagné de, se trouve avec... Par exemple : Astérix et Obelix, le pain et le fromage, les fraises et la chantilly.
        
        # Relations de priorité moyenne
        15: 1.05,     # r_lieu - Les relations de lieu ont une priorité moyenne
        53: 1.05,     # r_make - Que peut PRODUIRE le terme ? (par exemple abeille -> miel, usine -> voiture, agriculteur -> blé,  moteur -> gaz carbonique ...)
        67: 1.05      # r_similar - Similaire/ressemble à ; par exemple le congre est similaire à une anguille, ...
    }
    
    # Priorité par défaut pour les types de relation non explicitement listés
    default_priority = 1.0
    
    # Calcule un score pour chaque relation qui prend en compte à la fois le poids et la priorité du type
    best_score = float('-inf')
    best_type = None
    best_weight = 0
    
    # Suit également le poids brut le plus élevé pour comparaison
    max_raw_weight = float('-inf')
    max_raw_type = None
    
    for rel in relations:
        rel_type = rel.get('type')
        weight = float(rel.get('w', 0))
        
        # Suit la relation avec le poids brut le plus élevé
        if weight > max_raw_weight:
            max_raw_weight = weight
            max_raw_type = rel_type
        
        # Obtient le multiplicateur de priorité pour ce type de relation
        priority = relation_priority.get(rel_type, default_priority)
        
        # Calcule le score basé sur le poids et la priorité du type de relation
        score = weight * priority
        
        if score > best_score:
            best_score = score
            best_type = rel_type
            best_weight = weight
    
    # Obtient les noms des relations
    relation_name = relation_name_map.get(best_type, f"unknown_{best_type}")
    max_raw_name = relation_name_map.get(max_raw_type, f"unknown_{max_raw_type}")
    
    # Vérifie si la priorité a changé la sélection
    change_info = None
    if best_type != max_raw_type:
        print(f"La priorité a changé la sélection: {max_raw_name} (w={max_raw_weight}) -> {relation_name} (w={best_weight}, score={best_score})")
        change_info = {
            'raw_relation': max_raw_name,
            'raw_score': max_raw_weight,
            'new_relation': relation_name,
            'new_score': best_weight,
            'difference': best_weight - max_raw_weight
        }
    
    return relation_name, str(best_weight), change_info

def process_csv(input_csv_path, output_csv_path, relation_name_map):
    """Traite un seul fichier CSV et ajoute des colonnes de relation
    
    Returns:
        tuple: (rows_processed, changes_list) où changes_list contient des informations sur les changements de relation
    """
    print(f"Processing: {input_csv_path}")
    print(f"Output to: {output_csv_path}")
    
    # Pour suivre les changements pour le journal
    changes_list = []
    
    # Traite le fichier CSV
    with open(input_csv_path, 'r', encoding='utf-8') as infile, \
         open(output_csv_path, 'w', encoding='utf-8', newline='') as outfile:
        
        # Saute la ligne de commentaire si elle existe
        first_line = infile.readline().strip()
        if first_line.startswith('//'):
            # Lit ensuite l'en-tête réel
            csv_reader = csv.reader(infile)
            header = next(csv_reader)
        else:
            # La première ligne était l'en-tête
            csv_reader = csv.reader([first_line])
            header = next(csv_reader)
            # Continue la lecture depuis le début
            infile.seek(0)
            next(infile)  # Saute l'en-tête
            csv_reader = csv.reader(infile)
        
        # Crée un writer CSV avec les nouvelles colonnes
        csv_writer = csv.writer(outfile)
        new_header = header + ['relation_name', 'w']
        csv_writer.writerow(new_header)
        
        # Traite chaque ligne
        rows_processed = 0
        for row in csv_reader:
            rows_processed += 1
            if len(row) >= 6:  # S'assure qu'il y a suffisamment de colonnes
                relation_str = row[5]
                
                try:
                    # Analyse la chaîne JSON de relation
                    relations = ast.literal_eval(relation_str)
                    
                    if not relations or len(relations) == 0:
                        # Pas de relations, laisse les champs vides
                        new_row = row + ['', '']
                        csv_writer.writerow(new_row)
                    else:
                        # Utilise la fonction select_best_relation
                        relation_name, max_weight, change_info = select_best_relation(relations, relation_name_map)
                        
                        # Ajoute les nouvelles colonnes
                        new_row = row + [relation_name, max_weight]
                        csv_writer.writerow(new_row)
                        
                        # S'il y a eu un changement dans la sélection de relation, l'enregistre pour le journal
                        if change_info:
                            # Récupère le head et le lemma de la ligne (deux premières colonnes)
                            head = row[0] if len(row) > 0 else "unknown"
                            lemma = row[1] if len(row) > 1 else "unknown"
                            
                            # Ajoute le nom de fichier et les informations de paire au changement
                            filename = os.path.basename(input_csv_path)
                            change_info['filename'] = filename
                            change_info['head'] = head
                            change_info['lemma'] = lemma
                            
                            # Ajoute ce changement à notre liste
                            changes_list.append(change_info)
                    
                except (ValueError, SyntaxError) as e:
                    print(f"Erreur lors du traitement de la ligne {rows_processed} dans {os.path.basename(input_csv_path)}: {e}")
                    # Écrit la ligne sans les colonnes supplémentaires
                    csv_writer.writerow(row + ['error', 'error'])
            else:
                print(f"La ligne {rows_processed} dans {os.path.basename(input_csv_path)} n'a pas assez de colonnes: {row}")
                csv_writer.writerow(row + ['', ''])
        
        return rows_processed, changes_list

def main():
    # Définit les chemins de fichiers
    script_dir = os.path.dirname(os.path.abspath(__file__))
    relations_types_path = os.path.join(script_dir, 'relations_types.json')
    
    # Définit les répertoires de ressources et de sortie
    resources_dir = os.path.join(script_dir, f'ressources/{SOURCE_DIR}')
    output_dir = os.path.join(script_dir, f'ressources_completes/{SOURCE_DIR}')
    
    # Définit le chemin du fichier journal
    log_file_path = os.path.join(script_dir, f'log_calcul_w_{SOURCE_DIR}.csv')
    
    print(f"Répertoire du script: {script_dir}")
    print(f"Chargement des types de relations depuis: {relations_types_path}")
    print(f"Répertoire des ressources: {resources_dir}")
    print(f"Répertoire de sortie: {output_dir}")
    print(f"Le fichier journal sera enregistré à: {log_file_path}")
    
    # Vérifie si le répertoire de ressources existe
    if not os.path.exists(resources_dir):
        print(f"ERREUR: Répertoire de ressources non trouvé: {resources_dir}")
        return
        
    # Vérifie si le fichier relations_types existe
    if not os.path.exists(relations_types_path):
        print(f"ERREUR: Fichier de types de relations non trouvé: {relations_types_path}")
        return
    
    # Crée le répertoire de sortie s'il n'existe pas
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Répertoire de sortie créé: {output_dir}")
    
    # Charge la correspondance des types de relations
    with open(relations_types_path, 'r', encoding='utf-8') as f:
        # Saute la première ligne si elle commence par //
        content = f.read()
        if content.startswith('//'):
            content = content[content.find('\n') + 1:]
        relation_types = json.loads(content)
    
    # Crée une correspondance de l'ID de type au nom
    relation_name_map = {}
    for relation in relation_types:
        if isinstance(relation, dict) and 'id' in relation and 'name' in relation:
            relation_name_map[relation['id']] = relation['name']
    
    print(f"Chargé {len(relation_name_map)} types de relations")
    
    # Récupère tous les fichiers CSV du répertoire de ressources
    csv_files = [f for f in os.listdir(resources_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"Aucun fichier CSV trouvé dans {resources_dir}")
        return
    
    print(f"Trouvé {len(csv_files)} fichiers CSV à traiter")
    
    # Traite chaque fichier CSV
    total_files_processed = 0
    total_rows_processed = 0
    all_changes = []  # Pour collecter tous les changements pour le journal
    
    for csv_file in csv_files:
        input_path = os.path.join(resources_dir, csv_file)
        output_path = os.path.join(output_dir, f"{csv_file}")
        
        try:
            rows_processed, changes = process_csv(input_path, output_path, relation_name_map)
            total_rows_processed += rows_processed
            total_files_processed += 1
            all_changes.extend(changes)
            print(f"Traité {rows_processed} lignes dans {csv_file}, trouvé {len(changes)} changements de relation")
        except Exception as e:
            print(f"Erreur lors du traitement du fichier {csv_file}: {e}")
    
    # Écrit le fichier journal avec tous les changements
    if all_changes:
        try:
            with open(log_file_path, 'w', encoding='utf-8', newline='') as log_file:
                log_writer = csv.writer(log_file)
                # Écrit l'en-tête
                log_writer.writerow(['filename', 'head', 'lemma', 'raw_relation', 'raw_score', 'new_relation', 'new_score', 'difference'])
                
                # Écrit chaque changement
                for change in all_changes:
                    log_writer.writerow([
                        change['filename'],
                        change.get('head', ''),
                        change.get('lemma', ''),
                        change['raw_relation'],
                        change['raw_score'],
                        change['new_relation'],
                        change['new_score'],
                        change['difference']
                    ])
            print(f"Fichier journal créé avec {len(all_changes)} changements de relation à: {log_file_path}")
        except Exception as e:
            print(f"Erreur lors de l'écriture du fichier journal: {e}")
    else:
        print("Aucun changement de relation détecté, aucun fichier journal créé")
    
    print(f"Traitement terminé!")
    print(f"Total des fichiers traités: {total_files_processed}")
    print(f"Total des lignes traitées: {total_rows_processed}")
    print(f"Total des changements de relation: {len(all_changes)}")
    print(f"Fichiers de sortie enregistrés dans: {output_dir}")
    if all_changes:
        print(f"Fichier journal enregistré à: {log_file_path}")

if __name__ == "__main__":
    main()
