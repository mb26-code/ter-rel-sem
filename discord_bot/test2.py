import wikipediaapi

# Définition de l'API Wikipédia avec User-Agent personnalisé
wiki = wikipediaapi.Wikipedia(user_agent="FlorianBot/1.0 (florianlachieze@example.com)", language="fr")

# Exemple : Récupérer un article
page = wiki.page("Cuisine française")
if page.exists():
    print(page.summary[:500])  # Affiche un extrait
else:
    print(" L'article n'existe pas.")


import requests
from bs4 import BeautifulSoup

# URL d'une recette
url = "https://www.marmiton.org/recettes/recette_pate-a-crepes-des-plus-raffinees_49665.aspx"

# Récupération du contenu HTML
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Extraction du titre de la recette
titre = soup.find("h1").text

# Extraction des ingrédients (supposons qu'ils sont dans des <li>)
ingredients = [item.text for item in soup.find_all("li", class_="ingredient")]

print("Titre :", titre)
print("Ingrédients :", ingredients)

