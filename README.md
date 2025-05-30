# Extraction de relations sémantiques dans des textes culinaires

Ce projet a été réalisé dans le cadre du T.E.R. de Master 1 Informatique (parcours ICo) à l’Université de Montpellier. Il porte sur l’extraction de relations sémantiques à partir de textes en français dans le domaine de la gastronomie.

## Objectifs

- Développer un algorithme capable d’extraire des relations sémantiques dans des textes culinaires non-structurés (Wikipedia, Marmiton).
- Identifier automatiquement des relations typées (ex. : r_isa, r_carac, r_ingredient_bundle, etc.).
- Créer une base de données structurée de relations sémantiques.
- Proposer une interface simple de démonstration à travers un bot Discord.

## Membres du projet

- Ngo Hoai Nguyen 
- Marc Mathieu 
- Mehdi Bakhtar 
- Florian Lachièze 

## Fonctionnalités principales

- Extraction de textes culinaires depuis Wikipedia et Marmiton.
- Traitement linguistique via spaCy (analyse morpho-syntaxique).
- Filtrage par similarité sémantique (modèle FastText pré-entraîné).
- Interrogation de l’API JeuxDeMots pour obtenir ou valider les relations.
- Génération automatique de fichiers CSV contenant les relations extraites.
- Stockage des relations dans une base PostgreSQL via Supabase.
- Interface Discord interactive pour l’analyse en ligne.


### Prérequis

- Python 3.9 ou supérieur
- Modèle spaCy français : `fr_core_news_lg`
- Modèle FastText français : `cc.fr.300.vec.gz`
