from gensim.models import KeyedVectors
import os

# Chemin vers ton modèle
model_path = "models/cc.fr.300.vec.gz"

# Vérifie si le fichier existe
if not os.path.exists(model_path):
    print(f"❌ Fichier introuvable : {model_path}")
else:
    print("✅ Fichier trouvé. Chargement en cours...")

    try:
        model = KeyedVectors.load_word2vec_format(model_path, binary=False, limit=50000)  # on charge 50k mots pour tester

        print(f"✅ Modèle chargé avec succès : {len(model.key_to_index)} mots")
        print()

        # Test avec quelques mots connus
        for word in ["chocolat", "fromage", "cuisine"]:
            if word in model:
                similar = model.most_similar(word, topn=5)
                print(f"🔍 Mots similaires à « {word} » :")
                for sim_word, score in similar:
                    print(f"   {sim_word} ({score:.4f})")
                print()
            else:
                print(f"❌ Le mot « {word} » n'est pas dans le vocabulaire du modèle.")
    except Exception as e:
        print(f"❌ Erreur lors du chargement ou de la lecture du modèle : {e}")
