import requests
from bs4 import BeautifulSoup
import string
import os

def get_page_count(url):
    response = requests.get(url)
    if response.status_code != 200:
        return None
    
    soup = BeautifulSoup(response.text, 'html.parser')
    pagination = soup.find('nav', class_='af-pagination')
    
    if pagination:
        ul = pagination.find('ul')
        if ul:
            return len(ul.find_all('li'))
    return 1

def get_valid_recipe_links(url, max_pages):
    recipe_links = []
    
    for page in range(1, max_pages + 1):
        page_url = f"{url}/{page}"
        response = requests.get(page_url)
        if response.status_code != 200:
            continue
        
        soup = BeautifulSoup(response.text, 'html.parser')
        recipe_container = soup.find('div', class_='recipe-results fix-inline-block')
        if not recipe_container:
            continue
        
        for recipe in recipe_container.find_all('div', class_='index-item-card'):
            a_tag = recipe.find('a', href=True)
            img_tag = recipe.find('img', src=True)
            if a_tag and img_tag:
                img_src = img_tag['src']
                if "ingredient_default" not in img_src:
                    recipe_links.append("https://www.marmiton.org" + a_tag['href'])
    
    return recipe_links


def get_recipe_links(urls):
    all_recipes = []
    last_percentage = -1  # Stocke la dernière valeur affichée

    for i, url in enumerate(urls):
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            
            recipe_div = soup.find("div", class_="recipe-scroll-list")
            if recipe_div:
                recipe_items = recipe_div.find_all("li", class_="recipe-scroll-list__list__item", limit=7)

                for item in recipe_items:
                    link_tag = item.find("a")  # Cherche directement le lien dans <li>
                    if link_tag and link_tag.get("href") and link_tag["href"] not in all_recipes:
                        all_recipes.append(link_tag["href"])
        
        except requests.exceptions.RequestException as e:
            print(f"Erreur lors de la récupération de {url}: {e}")

        percentage = round((i + 1) / len(urls) * 100)  # Calcul du pourcentage actuel
        if percentage != last_percentage:  # Vérifie si le pourcentage a changé
            print(f"{percentage} %")
            last_percentage = percentage  # Met à jour la dernière valeur affichée

    return all_recipes

def scrape_marmiton(urls):
     # Ensure the main 'marmiton' directory exists

    for i, url in enumerate(urls):
        print(f"{i} / {len(urls)}")
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Erreur {response.status_code} pour {url}")
            continue

        soup = BeautifulSoup(response.text, 'html.parser')
        
        h1 = soup.find('h1').text.strip() if soup.find('h1') else "Sans titre"
        ingredients = [span.text.strip() for span in soup.find_all('span', class_='ingredient-name')]
        utensils = [" ".join(div.text.split()) for div in soup.find_all('div', class_='card-utensil-quantity')]
        steps = [" ".join(p.text.split()) for div in soup.find_all('div', class_='recipe-step-list__container') for p in div.find_all('p')]

        
        filename = os.path.join("marmiton", f"{h1.replace('/', '-')}.txt")  # Save the file in the proper directory

        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"Titre: {h1}\n")
                f.write("Ingrédients:\n" + "\n".join(ingredients) + "\n")
                f.write("Ustensiles:\n" + "\n".join(utensils) + "\n")
                f.write("Étapes:\n" + "\n".join(steps) + "\n")

        except Exception as e:
            print(f"Une erreur est survenue lors de l'enregistrement du fichier : {e}")



def extraction():
    base_url = "https://www.marmiton.org/recettes/index/ingredient"
    letters = list(string.ascii_lowercase) 
    recipe_links = []
    
    print("Premiere extraction : ")
    for i, letter in enumerate(letters):
        url = f"{base_url}/{letter}"
        print(f"{round(i/len(letters)*100)} %")
        max_pages = get_page_count(url)
        if max_pages is not None:
            recipe_links.extend(get_valid_recipe_links(url, max_pages))
    
    print("Seconde extraction : ")
    re = get_recipe_links(recipe_links)

    if not os.path.exists("marmiton"):
   
        os.makedirs("marmiton")

    scrape_marmiton(re)

    return recipe_links

if __name__ == "__main__":
    extraction()