import re

# List of strings that will cause a line to be skipped when found
SKIP_STRINGS = [
    "Sauce tomate cuisinée Créations \"Oignons caramélisés et échalotes\"",
    "&",
    "sot - le y - laisse",
    "haricot mange-tout",
    "®",

]

# List of French adjectives that naturally end with 's' in singular form
SINGULAR_ADJ_WITH_S = {
    "frais", "gris", "gros", "épais", "bas", "gras", "anglais", "danois", "chinois", 
    "japonais", "finlandais", "irlandais", "écossais", "polonais", "portugais", "suédois", 
    "français", "mauvais", "précis", "concis", "exquis", "soumis", "permis", "acquis", 
    "assis", "compris", "pris", "surpris", "repris"
}

# Map of strings that should be replaced during text processing
# Key: original string, Value: replacement string
STRING_REPLACEMENTS = {
    "’": "'",
    "un passoire": "passoire",
    "douille canneler ou uni": "douille cannelée",
    "agar - agar": "agar-agar",
    "céleri - rave": "céleri-rave",
    "cinq - épices": "cinq-épices",
    "emporter - pièce": "emporte-pièce",
    "pain à hot - dog": "pain à hot-dog",
}

def remove_parentheses(text):
    """
    Removes content within parentheses, including the parentheses themselves.
    
    Args:
        text (str): Text to process
        
    Returns:
        str: Text with parentheses and their content removed
    """
    # Remove content within parentheses and the parentheses themselves
    cleaned = re.sub(r'\s*\([^)]*\)\s*', ' ', text)
    # Clean up extra spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def extract_utensils_terms(input_text, nlp=None):
    """
    Traite un texte contenant des listes d'ustensiles ou d'équipements et retourne 
    une liste de termes nettoyés et lemmatisés.
    
    Args:
        input_text (str): Texte avec une liste d'items (un par ligne)
        
    Returns:
        list: Liste des termes nettoyés et lemmatisés
    """
    
    should_remove_terms = [
        r'top des meilleur(e)?s',
        r'top\s+\d*',
        r'meilleur(e)?s',
        r'le(s)? meilleur(e)?s',
        r'lot',
        r'set',
    ]
    
    # Créer une regex pour capturer toutes ces expressions
    promotional_pattern = re.compile(r'(' + '|'.join(should_remove_terms) + r')\s*', re.IGNORECASE)
    
    cleaned_terms = []
    
    for line in input_text.strip().split("\n"):
        # Skip lines containing any string from SKIP_STRINGS
        should_skip = False
        for skip_str in SKIP_STRINGS:
            if skip_str.lower() in line.lower():
                should_skip = True
                break
        if should_skip:
            continue
        
        # 2. Supprimer la quantité (ex. "1 ")
        clean = re.sub(r'^\d+\s*', '', line).strip()
        
        # 2.1 Supprimer les mots promotionnels (ex. "Top des meilleures")
        clean = promotional_pattern.sub('', clean).strip()
        
        doc = nlp(clean)
        
        # 3. Récupérer le premier noun chunk
        #    (on suppose qu'il recouvre tout l'item utile)
        try:
            chunk = next(doc.noun_chunks).text
        except StopIteration:
            # Si pas de noun chunk, utiliser le texte complet
            chunk = clean
        
        # 4. Retirer toute marque finale (suite de Proper Nouns)
        #    et re-passer dans spaCy pour lemmatisation
        tokens = list(nlp(chunk))
        filtered = []
        for tok in tokens:
            # on coupe dès qu'on trouve un PROPN *après* le premier token
            if tok.pos_ == "PROPN" and filtered:
                break
            filtered.append(tok)
        
        # 5. Lemmatiser et reconstituer la chaîne
        lemmas = [tok.lemma_.lower() for tok in filtered 
                if tok.pos_ not in ("NUM",)]  # on retire aussi d'éventuels restes de nombres
        item = " ".join(lemmas)
        
        # Apply STRING_REPLACEMENTS and remove parentheses just before adding to the final array
        if item.strip():
            for original, replacement in STRING_REPLACEMENTS.items():
                item = item.replace(original, replacement)
            # Remove parentheses and their contents
            item = remove_parentheses(item)
            # Only add if we still have content after cleaning
            if item.strip():
                cleaned_terms.append(item)
    
    return cleaned_terms

def extract_ingredients_terms(input_text, nlp=None):
    """
    Traite un texte contenant des listes d'ingrédients déjà propres et retourne 
    une liste de termes d'ingrédients avec les pluriels convertis au singulier.
    
    Args:
        input_text (str): Texte avec une liste d'ingrédients (un par ligne)
        
    Returns:
        list: Liste des ingrédients nettoyés (singularisés)
    """
    
    # Dictionnaire de corrections pour les erreurs connues de lemmatisation
    lemma_corrections = {
        "poivr": "poivre",
        "cèper": "cèpe",
        # Ajouter d'autres corrections si nécessaire
    }
    
    # Liste d'expressions figées à ne pas lemmatiser
    fixed_expressions = [
        "à la", "à l'", "aux", "d'", "de la", "de l'", "des",
    ]
    
    cleaned_ingredients = []
    
    for line in input_text.strip().split("\n"):
        # Ignorer les lignes vides
        if not line.strip():
            continue
        
        # Skip lines containing any string from SKIP_STRINGS
        should_skip = False
        for skip_str in SKIP_STRINGS:
            if skip_str.lower() in line.lower():
                should_skip = True
                break
        if should_skip:
            continue
        
        # Traiter avec spaCy pour la lemmatisation (singularisation)
        doc = nlp(line.strip())
        
        # Vérifier si la ligne contient des expressions figées à préserver
        text = line.strip()
        text_lower = text.lower()
        has_fixed_expr = False
        
        # Ajouter des espaces au début et à la fin pour assurer la détection des mots entiers
        text_for_search = f" {text_lower} "
        for expr in fixed_expressions:
            # Construire un pattern qui vérifie les limites de mots
            # Special handling for expressions starting with apostrophes like "d'"
            if expr.endswith("'"):
                pattern = fr'(\s|^)({expr})'
            else:
                pattern = fr'(\s|^)({expr})(\s|$|[,.;:!?])'
                
            if re.search(pattern, text_for_search):
                # Si elle contient une expression figée, on traite les parties avant et après séparément
                parts = re.split(fr'({expr})', text_lower, flags=re.IGNORECASE)
                processed_parts = []
                
                for part in parts:
                    # Si c'est l'expression figée, on la garde telle quelle
                    if part.lower() in fixed_expressions:
                        processed_parts.append(part)
                    # Sinon on la traite avec spaCy
                    elif part.strip():
                        doc_part = nlp(part)
                        lemmas_part = []
                        for token in doc_part:
                            if token.pos_ == "ADJ":
                                # Traiter les adjectifs comme avant
                                token_text = token.text.lower()
                                if token.morph.get("Number") == ["Plur"]:
                                    if token_text not in SINGULAR_ADJ_WITH_S:
                                        lemma = token_text[:-1] if token_text.endswith(('es', 's')) else token_text
                                    else:
                                        lemma = token_text
                                else:
                                    lemma = token_text
                            else:
                                lemma = token.lemma_.lower()
                            lemma = lemma_corrections.get(lemma, lemma)
                            lemmas_part.append(lemma)
                        processed_parts.append(" ".join(lemmas_part))
                
                # Assembler les parties traitées en une chaîne unique
                # Utiliser une logique spéciale pour éviter les espaces après des expressions qui se terminent par une apostrophe
                result = ""
                filtered_parts = [p for p in processed_parts if p.strip()]
                for i, part in enumerate(filtered_parts):
                    # Ajouter un espace sauf si c'est le premier élément ou si l'élément précédent finit par une apostrophe
                    if i > 0 and not filtered_parts[i-1].endswith("'"):
                        result += " "
                    result += part
                cleaned_text = re.sub(r'\s+', ' ', result).strip()
                lemmas = [cleaned_text]
                
                has_fixed_expr = True
                break
        
        if not has_fixed_expr:
            # Si pas d'expression figée, on utilise le traitement standard
            # Récupérer les lemmes (forme de base) pour chaque token
            # et appliquer les corrections si nécessaire
            lemmas = []
            for token in doc:
                if token.pos_ == "ADJ":
                    # Pour les adjectifs: conserver le genre mais assurer le singulier
                    text_lower = token.text.lower()
                    
                    # Pour tous les adjectifs (féminins et masculins)
                    if token.morph.get("Number") == ["Plur"]:
                        # Enlever le 's' final pour mettre au singulier, sauf si l'adjectif est dans la liste
                        if text_lower not in SINGULAR_ADJ_WITH_S:
                            lemma = text_lower[:-1] if text_lower.endswith(('es', 's')) else text_lower
                        else:
                            lemma = text_lower
                    else:
                        # Déjà au singulier
                        lemma = text_lower
                else:
                    # Pour les autres types de tokens (noms, verbes, etc.)
                    lemma = token.lemma_.lower()
                
                # Appliquer la correction si elle existe
                lemma = lemma_corrections.get(lemma, lemma)
                lemmas.append(lemma)
        
        # Reconstituer l'ingrédient
        ingredient = " ".join(lemmas)
        
        # Nettoyer les espaces multiples
        ingredient = re.sub(r'\s+', ' ', ingredient).strip()
        
        # Apply STRING_REPLACEMENTS and remove parentheses just before adding to the final array
        if ingredient:
            for original, replacement in STRING_REPLACEMENTS.items():
                ingredient = ingredient.replace(original, replacement)
            # Remove parentheses and their contents
            ingredient = remove_parentheses(ingredient)
            # Only add if we still have content after cleaning
            if ingredient.strip():
                cleaned_ingredients.append(ingredient)
    
    return cleaned_ingredients
