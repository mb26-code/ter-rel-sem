
from spacy_utils import extract_spacy_relations

def test_extract_spacy_relations():
    texte = "Le chef prépare la soupe avec une cuillère dans une casserole."
    mot = "soupe"

    relations = extract_spacy_relations(texte, mot)

    print("✅ Relations SpaCy extraites :")
    for rel in relations:
        print(f"- {mot} {rel['type']} {rel['related']} (source: {rel['source']})")
        print(f"  ↳ contexte : {rel['context']}")

if __name__ == "__main__":
    test_extract_spacy_relations()
