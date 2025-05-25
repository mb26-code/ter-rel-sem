import spacy

# Charger le modèle SpaCy français (assure-toi d’avoir exécuté : python -m spacy download fr_core_news_md)
nlp = spacy.load("fr_core_news_md")

# Dictionnaire de conversion des dépendances grammaticales SpaCy → relations sémantiques
SPACY_TO_SEMANTIC = {
    "obj": "r_object",
    "iobj": "r_object",
    "obl": "r_instr",         # souvent instrument ou complément de moyen
    "nmod": "r_location",     # modifieur nominal (lieu, contenant)
    "nsubj": "r_agent",       # sujet → celui qui agit
    "nsubj:pass": "r_patient" # sujet passif → celui qui subit
}

def extract_spacy_relations(text, mot):
    """
    Analyse un texte avec SpaCy pour détecter des relations syntaxiques entre `mot` et d'autres termes.
    Renvoie une liste de relations au format compatible avec le bot Discord.
    """
    doc = nlp(text)
    relations = []
    mot_lower = mot.lower()

    for token in doc:
        # Cas où le mot est le gouverneur
        if token.text.lower() == mot_lower or token.lemma_.lower() == mot_lower:
            for child in token.children:
                rel_type = SPACY_TO_SEMANTIC.get(child.dep_)
                if rel_type:
                    relations.append({
                        "type": rel_type,
                        "related": child.text,
                        "weight": 22,
                        "context": f"{token.text} → {child.text}",
                        "source": "spacy"
                    })

        # Cas où le mot est dépendant d'un verbe
        elif token.dep_ in SPACY_TO_SEMANTIC and token.head.text.lower() == mot_lower:
            rel_type = SPACY_TO_SEMANTIC[token.dep_]
            relations.append({
                "type": rel_type,
                "related": token.text,
                "weight": 22,
                "context": f"{token.text} ← {mot}",
                "source": "spacy"
            })

    return relations
