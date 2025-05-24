#!/usr/bin/env python3
"""
Analyse des relations sémantiques - Script d'analyse des données
Génère un rapport texte et des visualisations sur les relations extraites.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from collections import Counter
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class RelationsAnalyzer:
    def __init__(self, csv_file, relations_types_file, output_dir="analysis_output"):
        """
        Initialise l'analyseur avec les fichiers de données
        
        Args:
            csv_file: Chemin vers le fichier CSV des relations
            relations_types_file: Chemin vers le fichier JSON des types de relations
            output_dir: Répertoire de sortie pour les résultats
        """
        self.csv_file = csv_file
        self.relations_types_file = relations_types_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Charger les données
        self.df = pd.read_csv(csv_file)
        with open(relations_types_file, 'r', encoding='utf-8') as f:
            self.relation_types = json.load(f)
        
        # Créer un mapping des types de relations
        self.relation_map = {r['id']: r for r in self.relation_types}
        
        print(f"Données chargées: {len(self.df)} paires")
        print(f"Types de relations: {len(self.relation_types)}")
        
    def preprocess_data(self):
        """Préprocesse les données pour l'analyse"""
        # Filtrer les lignes avec des relations
        self.df_with_relations = self.df[self.df['relations'].notna() & (self.df['relations'] != '[]')].copy()
        
        # Parser les relations JSON
        self.df_with_relations['relations_parsed'] = self.df_with_relations['relations'].apply(
            lambda x: json.loads(x) if isinstance(x, str) and x.strip() != '[]' else []
        )
        
        # Exploder les relations pour avoir une ligne par relation
        self.relations_expanded = []
        for _, row in self.df_with_relations.iterrows():
            for rel in row['relations_parsed']:
                if isinstance(rel, dict):
                    rel_data = {
                        'node1': row['node1'],
                        'node2': row['node2'],
                        'dep': row['dep'],
                        'pos': row['pos'],
                        'sim': row['sim'],
                        'rel_id': rel.get('id'),
                        'rel_type': rel.get('type'),
                        'weight': rel.get('w', 0),
                        'best_relation': row.get('best_relation', ''),
                        'best_relation_w': row.get('best_relation_w', 0),
                        'new_relation': row.get('new_relation', ''),
                        'new_relation_w': row.get('new_relation_w', 0)
                    }
                    self.relations_expanded.append(rel_data)
        
        self.rel_df = pd.DataFrame(self.relations_expanded)
        print(f"Relations étendues: {len(self.rel_df)} relations individuelles")
        
    def analyze_top_relations(self, top_n=20):
        """Analyse les relations les plus fréquentes"""
        print("\n" + "="*60)
        print("ANALYSE DES RELATIONS LES PLUS FRÉQUENTES")
        print("="*60)
        
        # Compter les relations par type
        relation_counts = self.rel_df['rel_type'].value_counts().head(top_n)
        
        # Ajouter les noms des relations
        relation_stats = []
        for rel_type, count in relation_counts.items():
            rel_info = self.relation_map.get(rel_type, {})
            name = rel_info.get('name', f'Type_{rel_type}')
            gpname = rel_info.get('gpname', 'Inconnu')
            
            # Calculer les statistiques de poids
            weights = self.rel_df[self.rel_df['rel_type'] == rel_type]['weight']
            avg_weight = weights.mean() if len(weights) > 0 else 0
            
            relation_stats.append({
                'rel_type': rel_type,
                'name': name,
                'gpname': gpname,
                'count': count,
                'percentage': (count / len(self.rel_df)) * 100,
                'avg_weight': avg_weight
            })
        
        # Créer le rapport texte
        report = []
        report.append(f"TOP {top_n} RELATIONS SÉMANTIQUES")
        report.append("-" * 60)
        report.append(f"{'Type':<8} {'Nom':<20} {'Label':<25} {'Count':<8} {'%':<8} {'Poids moy.':<10}")
        report.append("-" * 90)
        
        for stat in relation_stats:
            report.append(f"{stat['rel_type']:<8} {stat['name'][:19]:<20} {stat['gpname'][:24]:<25} "
                         f"{stat['count']:<8} {stat['percentage']:.2f}%{'':<3} {stat['avg_weight']:.1f}")
        
        # Sauvegarder le rapport (sera inclus dans le rapport complet)
        self.top_relations_report = report
        
        print("\n".join(report))
        
        # Créer la visualisation
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Graphique 1: Distribution des relations
        top_10_stats = relation_stats[:10]
        names = [f"{s['name'][:15]}..." if len(s['name']) > 15 else s['name'] for s in top_10_stats]
        counts = [s['count'] for s in top_10_stats]
        
        bars = ax1.bar(range(len(names)), counts, color=plt.cm.viridis(np.linspace(0, 1, len(names))))
        ax1.set_xlabel('Type de relation')
        ax1.set_ylabel('Nombre d\'occurrences')
        ax1.set_title('Top 10 Relations Sémantiques')
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=45, ha='right')
        
        # Ajouter les valeurs sur les barres
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(counts),
                    f'{int(height)}', ha='center', va='bottom')
        
        # Graphique 2: Poids moyen par relation
        avg_weights = [s['avg_weight'] for s in top_10_stats]
        bars2 = ax2.bar(range(len(names)), avg_weights, color=plt.cm.plasma(np.linspace(0, 1, len(names))))
        ax2.set_xlabel('Type de relation')
        ax2.set_ylabel('Poids moyen')
        ax2.set_title('Poids Moyen par Type de Relation')
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, rotation=45, ha='right')
        
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(avg_weights),
                    f'{height:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "top_relations_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        return relation_stats
    
    def analyze_frequent_pairs(self, top_n=50):
        """Analyse les paires les plus fréquentes"""
        print("\n" + "="*60)
        print("ANALYSE DES PAIRES LES PLUS FRÉQUENTES")
        print("="*60)
        
        # Compter les paires
        pair_counts = Counter()
        pair_relations = {}
        
        for _, row in self.df_with_relations.iterrows():
            pair = (row['node1'], row['node2'])
            pair_counts[pair] += 1
            
            # Stocker les relations pour cette paire
            if pair not in pair_relations:
                pair_relations[pair] = {
                    'relations': [],
                    'best_relation': row.get('best_relation', ''),
                    'best_weight': row.get('best_relation_w', 0),
                    'dep': row['dep'],
                    'sim': row['sim']
                }
            
            if row['relations_parsed']:
                pair_relations[pair]['relations'].extend(row['relations_parsed'])
        
        # Prendre les top paires
        top_pairs = pair_counts.most_common(top_n)
        
        # Créer le rapport
        report = []
        report.append(f"TOP {top_n} PAIRES LES PLUS FRÉQUENTES")
        report.append("-" * 80)
        report.append(f"{'Paire':<40} {'Freq':<6} {'Relation':<20} {'Poids':<6} {'Sim':<6}")
        report.append("-" * 80)
        
        for (node1, node2), freq in top_pairs:
            pair_info = pair_relations[(node1, node2)]
            relation = pair_info['best_relation'] if pair_info['best_relation'] else 'Aucune'
            weight = pair_info['best_weight'] if pair_info['best_weight'] else 0
            sim = pair_info['sim']
            
            pair_str = f"{node1} → {node2}"
            if len(pair_str) > 39:
                pair_str = pair_str[:36] + "..."
                
            report.append(f"{pair_str:<40} {freq:<6} {relation[:19]:<20} {weight:<6.1f} {sim:<6.3f}")
        
        # Sauvegarder le rapport (sera inclus dans le rapport complet)
        self.frequent_pairs_report = report
        
        print("\n".join(report))
        
        # Visualisation de la distribution des fréquences
        plt.figure(figsize=(12, 6))
        frequencies = [freq for _, freq in top_pairs[:20]]
        labels = [f"{node1[:8]}→{node2[:8]}" for (node1, node2), _ in top_pairs[:20]]
        
        bars = plt.bar(range(len(frequencies)), frequencies, color=plt.cm.coolwarm(np.linspace(0, 1, len(frequencies))))
        plt.xlabel('Paires de mots')
        plt.ylabel('Fréquence')
        plt.title('Top 20 Paires les Plus Fréquentes')
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
        
        # Ajouter les valeurs sur les barres
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(frequencies),
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "frequent_pairs_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        return top_pairs
    
    def analyze_distribution(self):
        """Analyse la distribution générale des données"""
        print("\n" + "="*60)
        print("ANALYSE DE LA DISTRIBUTION DES DONNÉES")
        print("="*60)
        
        total_pairs = len(self.df)
        pairs_with_relations = len(self.df_with_relations)
        pairs_without_relations = total_pairs - pairs_with_relations
        
        # Analyse des relations nouvelles vs existantes
        new_relations = len(self.df[self.df['new_relation'].notna() & (self.df['new_relation'] != '')])
        existing_relations = len(self.df[self.df['best_relation'].notna() & (self.df['best_relation'] != '')])
        
        # Statistiques de similarité (pour la visualisation seulement)
        sim_stats = self.df['sim'].describe()
        
        # Créer le rapport
        report = []
        report.append("DISTRIBUTION GÉNÉRALE DES DONNÉES")
        report.append("-" * 50)
        report.append(f"Total des paires analysées: {total_pairs:,}")
        report.append(f"Paires avec relations: {pairs_with_relations:,} ({pairs_with_relations/total_pairs*100:.2f}%)")
        report.append(f"Paires sans relations: {pairs_without_relations:,} ({pairs_without_relations/total_pairs*100:.2f}%)")
        report.append("")
        report.append("TYPES DE RELATIONS:")
        report.append(f"Relations existantes (JdM): {existing_relations:,}")
        report.append(f"Nouvelles relations: {new_relations:,}")
        
        # Sauvegarder le rapport (sera inclus dans le rapport complet)
        self.distribution_report = report
        
        print("\n".join(report))
        
        # Créer les visualisations
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. Distribution des relations vs sans relations
        labels = ['Avec relations', 'Sans relations']
        sizes = [pairs_with_relations, pairs_without_relations]
        colors = ['#FF6B6B', '#4ECDC4']
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Distribution: Paires avec/sans Relations')
        
        # 2. Distribution de la similarité
        ax2.hist(self.df['sim'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(sim_stats['mean'], color='red', linestyle='--', label=f'Moyenne: {sim_stats["mean"]:.3f}')
        ax2.axvline(sim_stats['50%'], color='orange', linestyle='--', label=f'Médiane: {sim_stats["50%"]:.3f}')
        ax2.set_xlabel('Similarité Word2Vec')
        ax2.set_ylabel('Fréquence')
        ax2.set_title('Distribution de la Similarité')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "data_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Graphique de distribution des poids des relations
        if len(self.rel_df) > 0:
            plt.figure(figsize=(10, 6))
            weights = self.rel_df['weight']
            plt.hist(weights, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
            plt.axvline(weights.mean(), color='red', linestyle='--', label=f'Moyenne: {weights.mean():.1f}')
            plt.axvline(weights.median(), color='orange', linestyle='--', label=f'Médiane: {weights.median():.1f}')
            plt.xlabel('Poids des relations')
            plt.ylabel('Fréquence')
            plt.title('Distribution des Poids des Relations')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(self.output_dir / "relations_weights_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def analyze_new_relations(self, top_n=20):
        """Analyse des nouvelles relations et leurs poids"""
        print("\n" + "="*60)
        print("ANALYSE DES NOUVELLES RELATIONS")
        print("="*60)
        
        # Filtrer les données avec des nouvelles relations
        new_rel_data = self.df[
            (self.df['new_relation'].notna()) & 
            (self.df['new_relation'] != '') &
            (self.df['new_relation_w'].notna()) &
            (self.df['new_relation_w'] > 0)
        ].copy()
        
        total_new_relations = len(new_rel_data)
        
        if total_new_relations == 0:
            report = ["Aucune nouvelle relation trouvée dans le dataset."]
            self.new_relations_report = report
            print("\n".join(report))
            return None
        
        # Statistiques générales des nouvelles relations
        weight_stats = new_rel_data['new_relation_w'].describe()
        
        # Compter les nouvelles relations par type
        new_relation_counts = new_rel_data['new_relation'].value_counts().head(top_n)
        
        # Analyser les poids par type de nouvelle relation
        new_relation_stats = []
        for rel_name, count in new_relation_counts.items():
            rel_weights = new_rel_data[new_rel_data['new_relation'] == rel_name]['new_relation_w']
            avg_weight = rel_weights.mean()
            max_weight = rel_weights.max()
            min_weight = rel_weights.min()
            
            new_relation_stats.append({
                'relation': rel_name,
                'count': count,
                'percentage': (count / total_new_relations) * 100,
                'avg_weight': avg_weight,
                'max_weight': max_weight,
                'min_weight': min_weight
            })
        
        # Créer le rapport texte
        report = []
        report.append(f"ANALYSE DES NOUVELLES RELATIONS (TOP {top_n})")
        report.append("-" * 70)
        report.append(f"Total des nouvelles relations: {total_new_relations:,}")
        report.append(f"Pourcentage du corpus: {(total_new_relations/len(self.df)*100):.2f}%")
        report.append("")
        report.append("STATISTIQUES DES POIDS:")
        report.append(f"  Poids moyen: {weight_stats['mean']:.3f}")
        report.append(f"  Poids médian: {weight_stats['50%']:.3f}")
        report.append(f"  Écart-type: {weight_stats['std']:.3f}")
        report.append(f"  Poids minimum: {weight_stats['min']:.3f}")
        report.append(f"  Poids maximum: {weight_stats['max']:.3f}")
        report.append("")
        report.append(f"{'Nouvelle Relation':<25} {'Count':<8} {'%':<8} {'Poids Moy':<12} {'Poids Max':<12}")
        report.append("-" * 75)
        
        for stat in new_relation_stats:
            report.append(f"{stat['relation'][:24]:<25} {stat['count']:<8} "
                         f"{stat['percentage']:.2f}%{'':<3} {stat['avg_weight']:<12.3f} {stat['max_weight']:<12.3f}")
        
        # Sauvegarder le rapport
        self.new_relations_report = report
        
        print("\n".join(report))
        
        # Créer les visualisations
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Graphique 1: Distribution des nouvelles relations (top 10)
        top_10_stats = new_relation_stats[:10]
        names = [stat['relation'][:15] + "..." if len(stat['relation']) > 15 else stat['relation'] 
                for stat in top_10_stats]
        counts = [stat['count'] for stat in top_10_stats]
        
        bars1 = ax1.bar(range(len(names)), counts, color=plt.cm.viridis(np.linspace(0, 1, len(names))))
        ax1.set_xlabel('Type de nouvelle relation')
        ax1.set_ylabel('Nombre d\'occurrences')
        ax1.set_title('Top 10 Nouvelles Relations')
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=45, ha='right')
        
        # Ajouter les valeurs sur les barres
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(counts),
                    f'{int(height)}', ha='center', va='bottom')
        
        # Graphique 2: Poids moyen par nouvelle relation
        avg_weights = [stat['avg_weight'] for stat in top_10_stats]
        bars2 = ax2.bar(range(len(names)), avg_weights, color=plt.cm.plasma(np.linspace(0, 1, len(names))))
        ax2.set_xlabel('Type de nouvelle relation')
        ax2.set_ylabel('Poids moyen')
        ax2.set_title('Poids Moyen par Nouvelle Relation')
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, rotation=45, ha='right')
        
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(avg_weights),
                    f'{height:.3f}', ha='center', va='bottom')
        
        # Graphique 3: Distribution des poids des nouvelles relations
        ax3.hist(new_rel_data['new_relation_w'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        ax3.axvline(weight_stats['mean'], color='red', linestyle='--', 
                   label=f'Moyenne: {weight_stats["mean"]:.3f}')
        ax3.axvline(weight_stats['50%'], color='orange', linestyle='--', 
                   label=f'Médiane: {weight_stats["50%"]:.3f}')
        ax3.set_xlabel('Poids des nouvelles relations')
        ax3.set_ylabel('Fréquence')
        ax3.set_title('Distribution des Poids - Nouvelles Relations')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Graphique 4: Comparaison avec les relations existantes
        if 'best_relation_w' in self.df.columns:
            existing_weights = self.df[
                (self.df['best_relation_w'].notna()) & 
                (self.df['best_relation_w'] > 0)
            ]['best_relation_w']
            
            # Créer des données pour la comparaison
            new_weights = new_rel_data['new_relation_w']
            
            ax4.hist(existing_weights, bins=30, alpha=0.5, label='Relations existantes', 
                    color='skyblue', density=True)
            ax4.hist(new_weights, bins=30, alpha=0.5, label='Nouvelles relations', 
                    color='lightcoral', density=True)
            
            ax4.axvline(existing_weights.mean(), color='blue', linestyle='--', alpha=0.7,
                       label=f'Moy. existantes: {existing_weights.mean():.3f}')
            ax4.axvline(new_weights.mean(), color='red', linestyle='--', alpha=0.7,
                       label=f'Moy. nouvelles: {new_weights.mean():.3f}')
            
            ax4.set_xlabel('Poids des relations')
            ax4.set_ylabel('Densité')
            ax4.set_title('Comparaison: Nouvelles vs Existantes')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Données de comparaison\nnon disponibles', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Comparaison: Nouvelles vs Existantes')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "new_relations_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        return new_relation_stats
    
    def generate_complete_report(self):
        """Génère un rapport complet unique avec toutes les analyses"""
        total_pairs = len(self.df)
        pairs_with_relations = len(self.df_with_relations)
        total_relations = len(self.rel_df)
        unique_relation_types = len(self.rel_df['rel_type'].unique()) if len(self.rel_df) > 0 else 0
        
        complete_report = []
        
        # En-tête principal
        complete_report.append("="*80)
        complete_report.append("RAPPORT COMPLET - ANALYSE DES RELATIONS SÉMANTIQUES")
        complete_report.append("="*80)
        complete_report.append(f"Date d'analyse: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        complete_report.append("")
        
        # Résumé exécutif
        complete_report.append("RÉSUMÉ EXÉCUTIF")
        complete_report.append("-" * 40)
        complete_report.append(f"• Total des paires analysées: {total_pairs:,}")
        complete_report.append(f"• Paires avec relations: {pairs_with_relations:,} ({pairs_with_relations/total_pairs*100:.1f}%)")
        complete_report.append(f"• Total des relations individuelles: {total_relations:,}")
        complete_report.append(f"• Types de relations uniques utilisés: {unique_relation_types}")
        
        # Ajouter les statistiques des nouvelles relations
        new_relations_count = len(self.df[
            (self.df['new_relation'].notna()) & 
            (self.df['new_relation'] != '') &
            (self.df['new_relation_w'].notna()) &
            (self.df['new_relation_w'] > 0)
        ])
        if new_relations_count > 0:
            new_relations_pct = (new_relations_count / total_pairs) * 100
            complete_report.append(f"• Nouvelles relations identifiées: {new_relations_count:,} ({new_relations_pct:.1f}%)")
        
        complete_report.append("")
        complete_report.append("")
        
        # Section 1: Top Relations
        complete_report.append("="*80)
        complete_report.append("SECTION 1: RELATIONS SÉMANTIQUES LES PLUS FRÉQUENTES")
        complete_report.append("="*80)
        complete_report.extend(self.top_relations_report[1:])  # Skip the title since we have our own
        complete_report.append("")
        complete_report.append("")
        
        # Section 2: Paires fréquentes
        complete_report.append("="*80)
        complete_report.append("SECTION 2: PAIRES DE MOTS LES PLUS FRÉQUENTES")
        complete_report.append("="*80)
        complete_report.extend(self.frequent_pairs_report[1:])  # Skip the title
        complete_report.append("")
        complete_report.append("")
        
        # Section 3: Distribution
        complete_report.append("="*80)
        complete_report.append("SECTION 3: DISTRIBUTION ET STATISTIQUES GÉNÉRALES")
        complete_report.append("="*80)
        complete_report.extend(self.distribution_report[1:])  # Skip the title
        complete_report.append("")
        complete_report.append("")
        
        # Section 4: Nouvelles Relations
        complete_report.append("="*80)
        complete_report.append("SECTION 4: ANALYSE DES NOUVELLES RELATIONS")
        complete_report.append("="*80)
        if hasattr(self, 'new_relations_report'):
            complete_report.extend(self.new_relations_report[1:])  # Skip the title
        else:
            complete_report.append("Aucune nouvelle relation analysée.")
        complete_report.append("")
        complete_report.append("")
        
        # Section 5: Analyses supplémentaires
        complete_report.append("="*80)
        complete_report.append("SECTION 5: ANALYSES DÉTAILLÉES")
        complete_report.append("="*80)
        
        # Analyse des relations par poids
        if len(self.rel_df) > 0:
            weight_stats = self.rel_df['weight'].describe()
            complete_report.append("STATISTIQUES DES POIDS DES RELATIONS:")
            complete_report.append(f"  Poids moyen: {weight_stats['mean']:.2f}")
            complete_report.append(f"  Poids médian: {weight_stats['50%']:.2f}")
            complete_report.append(f"  Écart-type: {weight_stats['std']:.2f}")
            complete_report.append(f"  Poids minimum: {weight_stats['min']:.2f}")
            complete_report.append(f"  Poids maximum: {weight_stats['max']:.2f}")
            complete_report.append("")
            
            # Top relations par poids moyen
            top_by_weight = self.rel_df.groupby('rel_type')['weight'].agg(['mean', 'count']).sort_values('mean', ascending=False).head(10)
            complete_report.append("TOP 10 RELATIONS PAR POIDS MOYEN:")
            complete_report.append(f"{'Type':<8} {'Nom':<25} {'Poids moy.':<12} {'Occurrences':<12}")
            complete_report.append("-" * 60)
            
            for rel_type, (avg_weight, count) in top_by_weight.iterrows():
                rel_info = self.relation_map.get(rel_type, {})
                name = rel_info.get('name', f'Type_{rel_type}')
                complete_report.append(f"{rel_type:<8} {name[:24]:<25} {avg_weight:<12.1f} {count:<12}")
            complete_report.append("")
        
        complete_report.append("")
        complete_report.append("")
        
        # Comparaison nouvelles vs existantes relations
        complete_report.append("COMPARAISON: NOUVELLES VS EXISTANTES RELATIONS:")
        new_rel_data = self.df[
            (self.df['new_relation'].notna()) & 
            (self.df['new_relation'] != '') &
            (self.df['new_relation_w'].notna()) &
            (self.df['new_relation_w'] > 0)
        ]
        existing_rel_data = self.df[
            (self.df['best_relation_w'].notna()) & 
            (self.df['best_relation_w'] > 0)
        ]
        
        if len(new_rel_data) > 0 and len(existing_rel_data) > 0:
            new_avg_weight = new_rel_data['new_relation_w'].mean()
            existing_avg_weight = existing_rel_data['best_relation_w'].mean()
            
            complete_report.append(f"  Relations existantes (JeuxDeMots):")
            complete_report.append(f"    - Nombre: {len(existing_rel_data):,}")
            complete_report.append(f"    - Poids moyen: {existing_avg_weight:.2f}")
            complete_report.append(f"  Nouvelles relations:")
            complete_report.append(f"    - Nombre: {len(new_rel_data):,}")
            complete_report.append(f"    - Poids moyen: {new_avg_weight:.2f}")
            
            ratio = len(new_rel_data) / len(existing_rel_data) if len(existing_rel_data) > 0 else 0
            complete_report.append(f"    - Ratio nouvelles/existantes: {ratio:.1f}:1")
        
        complete_report.append("")
        complete_report.append("")
        
        # Conclusion
        complete_report.append("="*80)
        complete_report.append("CONCLUSION")
        complete_report.append("="*80)
        
        coverage = (pairs_with_relations / total_pairs) * 100
        if coverage > 10:
            quality = "Excellente"
        elif coverage > 5:
            quality = "Bonne"
        elif coverage > 2:
            quality = "Moyenne"
        else:
            quality = "Faible"
            
        complete_report.append(f"Couverture des relations: {coverage:.1f}% - Qualité: {quality}")
        complete_report.append(f"Le corpus contient {unique_relation_types} types de relations différents.")
        
        if len(self.rel_df) > 0:
            dominant_relation = self.rel_df['rel_type'].value_counts().index[0]
            dominant_rel_info = self.relation_map.get(dominant_relation, {})
            dominant_name = dominant_rel_info.get('name', f'Type_{dominant_relation}')
            dominant_pct = (self.rel_df['rel_type'].value_counts().iloc[0] / len(self.rel_df)) * 100
            complete_report.append(f"Relation dominante: {dominant_name} ({dominant_pct:.1f}% des relations)")
        
        complete_report.append("")
        complete_report.append(f"Analyse terminée le {pd.Timestamp.now().strftime('%Y-%m-%d à %H:%M:%S')}")
        complete_report.append("="*80)
        
        # Sauvegarder le rapport complet
        with open(self.output_dir / "complete_analysis_report.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(complete_report))
        
        print("\n" + "="*60)
        print("RAPPORT COMPLET GÉNÉRÉ")
        print("="*60)
        print(f"Fichier sauvegardé: {self.output_dir / 'complete_analysis_report.txt'}")
        print(f"Taille du rapport: {len(complete_report)} lignes")
        print("="*60)

    
    def run_complete_analysis(self):
        """Lance l'analyse complète"""
        print("Début de l'analyse des relations sémantiques...")
        print(f"Répertoire de sortie: {self.output_dir.absolute()}")
        
        # Préprocessing
        self.preprocess_data()
        
        # Analyses
        self.analyze_top_relations()
        self.analyze_frequent_pairs()
        self.analyze_distribution()
        self.analyze_new_relations()
        
        # Rapport final complet
        self.generate_complete_report()
    
    def analyze_new_relations(self, top_n=20):
        """Analyse des nouvelles relations et leurs poids"""
        print("\n" + "="*60)
        print("ANALYSE DES NOUVELLES RELATIONS")
        print("="*60)
        
        # Filtrer les données avec des nouvelles relations
        new_rel_data = self.df[
            (self.df['new_relation'].notna()) & 
            (self.df['new_relation'] != '') &
            (self.df['new_relation_w'].notna()) &
            (self.df['new_relation_w'] > 0)
        ].copy()
        
        total_new_relations = len(new_rel_data)
        
        if total_new_relations == 0:
            report = ["Aucune nouvelle relation trouvée dans le dataset."]
            self.new_relations_report = report
            print("\n".join(report))
            return None
        
        # Statistiques générales des nouvelles relations
        weight_stats = new_rel_data['new_relation_w'].describe()
        
        # Compter les nouvelles relations par type
        new_relation_counts = new_rel_data['new_relation'].value_counts().head(top_n)
        
        # Analyser les poids par type de nouvelle relation
        new_relation_stats = []
        for rel_name, count in new_relation_counts.items():
            rel_weights = new_rel_data[new_rel_data['new_relation'] == rel_name]['new_relation_w']
            avg_weight = rel_weights.mean()
            max_weight = rel_weights.max()
            min_weight = rel_weights.min()
            
            new_relation_stats.append({
                'relation': rel_name,
                'count': count,
                'percentage': (count / total_new_relations) * 100,
                'avg_weight': avg_weight,
                'max_weight': max_weight,
                'min_weight': min_weight
            })
        
        # Créer le rapport texte
        report = []
        report.append(f"ANALYSE DES NOUVELLES RELATIONS (TOP {top_n})")
        report.append("-" * 70)
        report.append(f"Total des nouvelles relations: {total_new_relations:,}")
        report.append(f"Pourcentage du corpus: {(total_new_relations/len(self.df)*100):.2f}%")
        report.append("")
        report.append("STATISTIQUES DES POIDS:")
        report.append(f"  Poids moyen: {weight_stats['mean']:.3f}")
        report.append(f"  Poids médian: {weight_stats['50%']:.3f}")
        report.append(f"  Écart-type: {weight_stats['std']:.3f}")
        report.append(f"  Poids minimum: {weight_stats['min']:.3f}")
        report.append(f"  Poids maximum: {weight_stats['max']:.3f}")
        report.append("")
        report.append(f"{'Nouvelle Relation':<25} {'Count':<8} {'%':<8} {'Poids Moy':<12} {'Poids Max':<12}")
        report.append("-" * 75)
        
        for stat in new_relation_stats:
            report.append(f"{stat['relation'][:24]:<25} {stat['count']:<8} "
                         f"{stat['percentage']:.2f}%{'':<3} {stat['avg_weight']:<12.3f} {stat['max_weight']:<12.3f}")
        
        # Sauvegarder le rapport
        self.new_relations_report = report
        
        print("\n".join(report))
        
        # Créer les visualisations
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Graphique 1: Distribution des nouvelles relations (top 10)
        top_10_stats = new_relation_stats[:10]
        names = [stat['relation'][:15] + "..." if len(stat['relation']) > 15 else stat['relation'] 
                for stat in top_10_stats]
        counts = [stat['count'] for stat in top_10_stats]
        
        bars1 = ax1.bar(range(len(names)), counts, color=plt.cm.viridis(np.linspace(0, 1, len(names))))
        ax1.set_xlabel('Type de nouvelle relation')
        ax1.set_ylabel('Nombre d\'occurrences')
        ax1.set_title('Top 10 Nouvelles Relations')
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=45, ha='right')
        
        # Ajouter les valeurs sur les barres
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(counts),
                    f'{int(height)}', ha='center', va='bottom')
        
        # Graphique 2: Poids moyen par nouvelle relation
        avg_weights = [stat['avg_weight'] for stat in top_10_stats]
        bars2 = ax2.bar(range(len(names)), avg_weights, color=plt.cm.plasma(np.linspace(0, 1, len(names))))
        ax2.set_xlabel('Type de nouvelle relation')
        ax2.set_ylabel('Poids moyen')
        ax2.set_title('Poids Moyen par Nouvelle Relation')
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, rotation=45, ha='right')
        
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(avg_weights),
                    f'{height:.3f}', ha='center', va='bottom')
        
        # Graphique 3: Distribution des poids des nouvelles relations
        ax3.hist(new_rel_data['new_relation_w'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        ax3.axvline(weight_stats['mean'], color='red', linestyle='--', 
                   label=f'Moyenne: {weight_stats["mean"]:.3f}')
        ax3.axvline(weight_stats['50%'], color='orange', linestyle='--', 
                   label=f'Médiane: {weight_stats["50%"]:.3f}')
        ax3.set_xlabel('Poids des nouvelles relations')
        ax3.set_ylabel('Fréquence')
        ax3.set_title('Distribution des Poids - Nouvelles Relations')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Graphique 4: Comparaison avec les relations existantes
        if 'best_relation_w' in self.df.columns:
            existing_weights = self.df[
                (self.df['best_relation_w'].notna()) & 
                (self.df['best_relation_w'] > 0)
            ]['best_relation_w']
            
            # Créer des données pour la comparaison
            new_weights = new_rel_data['new_relation_w']
            
            ax4.hist(existing_weights, bins=30, alpha=0.5, label='Relations existantes', 
                    color='skyblue', density=True)
            ax4.hist(new_weights, bins=30, alpha=0.5, label='Nouvelles relations', 
                    color='lightcoral', density=True)
            
            ax4.axvline(existing_weights.mean(), color='blue', linestyle='--', alpha=0.7,
                       label=f'Moy. existantes: {existing_weights.mean():.3f}')
            ax4.axvline(new_weights.mean(), color='red', linestyle='--', alpha=0.7,
                       label=f'Moy. nouvelles: {new_weights.mean():.3f}')
            
            ax4.set_xlabel('Poids des relations')
            ax4.set_ylabel('Densité')
            ax4.set_title('Comparaison: Nouvelles vs Existantes')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Données de comparaison\nnon disponibles', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Comparaison: Nouvelles vs Existantes')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "new_relations_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        return new_relation_stats

def main():
    """Fonction principale"""
    # Chemins des fichiers
    base_dir = Path(__file__).parent.parent
    csv_file = base_dir / "post_traitement" / "FINAL_semantic_relations_enriched.csv"
    relations_types_file = base_dir / "post_traitement" / "relations_types.json"
    output_dir = base_dir / "analyse" / "analysis_output"
    
    # Vérifier l'existence des fichiers
    if not csv_file.exists():
        print(f"Erreur: Fichier CSV non trouvé: {csv_file}")
        return
    
    if not relations_types_file.exists():
        print(f"Erreur: Fichier types de relations non trouvé: {relations_types_file}")
        return
    
    # Lancer l'analyse
    analyzer = RelationsAnalyzer(csv_file, relations_types_file, output_dir)
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()