import requests

mot = "citron"

# Étape 1 : Récupérer l'ID du mot grâce à l'autocomplétion
autocomplete_url = "https://jdm-api.demo.lirmm.fr/v1/nodes/public/autocomplete"
params = {"term": mot}
response = requests.get(autocomplete_url, params=params)

if response.status_code != 200:
    print("Erreur lors de l'autocomplétion :", response.status_code)
    exit()

data = response.json()
if not data:
    print(f"Aucun résultat trouvé pour le mot : {mot}")
    exit()

mot_id = data[0]["id"]
mot_nom = data[0]["name"]
print(f"Mot trouvé : {mot_nom} (id={mot_id})")

# Étape 2 : Récupérer les relations pour cet ID
relations_url = f"https://jdm-api.demo.lirmm.fr/v1/graphs/jdm-relations/public/nodes/{mot_id}/relations"
response = requests.get(relations_url)

if response.status_code != 200:
    print("Erreur lors de la récupération des relations :", response.status_code)
    exit()

relations = response.json()

# Étape 3 : Affichage des relations
print(f"\nRelations pour le mot '{mot_nom}' :\n")
for relation in relations:
    type_rel = relation["type"]["name"]
    cible = relation["node"]["name"]
    poids = relation["weight"]
    print(f"{type_rel} → {cible} (poids={poids})")
