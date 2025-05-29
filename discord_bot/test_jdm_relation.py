import requests
import json
from configs import BASE_JDM_URL, PATH_RELATIONS_TYPES
from ngo import select_best_jdm_relation

def query_jdm_api(head, lemma):
    try:
        url = f"{BASE_JDM_URL}/from/{head}/to/{lemma}"
        print(f"Requête vers : {url}")
        resp = requests.get(url)
        resp.raise_for_status()
        return resp.json().get("relations", [])
    except Exception as e:
        print(f"Erreur : {e}")
        return []

def load_relation_types():
    with open(PATH_RELATIONS_TYPES, "r", encoding="utf-8") as f:
        rels = json.load(f)
        return {r["id"]: r["name"] for r in rels}

def main():
    head = input("Mot 1 (ex: citron) : ").strip()
    lemma = input("Mot 2 (ex: acide) : ").strip()

    relation_types = load_relation_types()
    relations = query_jdm_api(head, lemma)

    if not relations:
        print("❌ Aucune relation trouvée.")
        return

    print(f"\nNombre de relations brutes : {len(relations)}")

    best, weight = select_best_jdm_relation(relations, relation_types)
    if best:
        print(f"\n✅ Meilleure relation : {head} → {best} → {lemma} (poids : {weight})")
    else:
        print("⚠️ Aucune relation exploitable trouvée.")

if __name__ == "__main__":
    main()
