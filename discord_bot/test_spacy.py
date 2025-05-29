import spacy

def test_spacy_on_example():
    try:
        # Charger le modèle français
        nlp = spacy.load("fr_core_news_md")
        print("✅ Modèle SpaCy chargé avec succès.")

        # Exemple de phrase culinaire
        texte = "Le chef prépare la soupe avec une cuillère dans une casserole."
        doc = nlp(texte)

        print("\n🔍 Analyse des dépendances :")
        for token in doc:
            print(f"{token.text:<12} — dep: {token.dep_:<10} — head: {token.head.text:<12} — pos: {token.pos_}")

    except Exception as e:
        print(f"❌ Erreur lors de l'utilisation de SpaCy : {str(e)}")

if __name__ == "__main__":
    test_spacy_on_example()
