import requests
from bs4 import BeautifulSoup
import os
import random

URL = "https://fr.wikipedia.org/wiki/Cat%C3%A9gorie:Cuisine_par_pays"
BASE_URL = "https://fr.wikipedia.org"

# ------------- PREMIER NIVEAU : liens de la liste des plats par pays

def niveau_1(url, base_url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    
    category_div = soup.find("div", class_="mw-category mw-category-columns")
    cuisine_links = []
    
    if category_div:
        for group in category_div.find_all("div", class_="mw-category-group")[2:]:
            for link in group.find_all("a"):
                href = link.get("href")
                if href :
                    cuisine_links.append(base_url + href)
    
    return cuisine_links

# --------------- 2eme NIVEAU : 

# ------- 2.1 : lien des plats par pays

# -------- 2.2 : lien des sous listes de plats par pays (catégories différentes)

def niveau_2(url):

    tous_les_plats_du_pays = []

    response = requests.get(url)
    if response.status_code != 200:
        print(f"Erreur {response.status_code} lors de l'accès à {url}")
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    #---------- TEST DU 2.2 (sous catégories de plats par pays)  SI OUI -- > on prend ces sous plats et on les ajoutera dans les plats par pays du 2.1

    mw_subcategories_div = soup.find(id='mw-subcategories')

    liens_ss_plats = []
    if mw_subcategories_div :  
        links = []

        # Extraire toutes les sous-divisions de classe 'mw-category-group'
        category_groups = mw_subcategories_div.find_all('div', class_='mw-category-group')

        # Parcourir chaque groupe et extraire les liens
        for group in category_groups:
         
            # Trouver tous les liens dans chaque sous-div
            links_in_group = group.find_all('a', href=True)
            
            # Ajouter les liens à la liste
            for link in links_in_group:
                links.append(f"https://fr.wikipedia.org{link['href']}")


            for url in links :
                l = niveau_2_extraction_plat(url)
                if l : liens_ss_plats.extend(l)

                
     
    category_div = soup.find('div', class_='mw-category mw-category-columns')
    
    if not category_div:
        #print(f"Pas de catégorie trouvée sur {url}")
        return []
    
    links1 = []
    for group in category_div.find_all('div', class_='mw-category-group'):
        h3_tag = group.find('h3')
        if h3_tag and not h3_tag.text.strip().isalpha():  # Vérifie si ce n'est pas une lettre
            continue  # Ignore ce groupe

        for li in group.find_all('li'):
            a_tag = li.find('a')
            if 'href' in a_tag.attrs:
                links1.append(f"https://fr.wikipedia.org{a_tag['href']}")


    #---------- ICI ON RASSEMBLE LES PLATS PAR PAYS (2.1) ET LES PLATS CATEGORISES DU PAYS (2.2)
    
    if liens_ss_plats : tous_les_plats_du_pays.extend(liens_ss_plats)
    if links1 : tous_les_plats_du_pays.extend(links1)

    return tous_les_plats_du_pays

    
def niveau_2_extraction_plat(url) :
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Erreur {response.status_code} lors de l'accès à {url}")
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')

    category_div = soup.find('div', class_='mw-category mw-category-columns')
    
    if not category_div:
        category_div = soup.find('div', class_='mw-category')

        if not category_div :
            #print(f"Pas de catégorie trouvée sur {url}")
            return []
    
    links1 = []
    for group in category_div.find_all('div', class_='mw-category-group'):
        h3_tag = group.find('h3')
        if h3_tag and not h3_tag.text.strip().isalpha():  # Vérifie si ce n'est pas une lettre
            continue  # Ignore ce groupe

        for li in group.find_all('li'):
            a_tag = li.find('a')
            if 'href' in a_tag.attrs:
                links1.append(f"https://fr.wikipedia.org{a_tag['href']}")

    return links1

def scrape_wikipedia(urls):
    
    folder_name = "wikipedia"
    os.makedirs(folder_name, exist_ok=True)
    
    for i, url in enumerate(urls):
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            
            title_element = soup.find('span', class_='mw-page-title-main')
            if title_element:
                title = title_element.text.strip()
            else:
                
                h1_element = soup.find('h1', id='firstHeading')
                if h1_element and h1_element.i:
                    title = h1_element.i.text.strip()
                elif h1_element:
                    title = h1_element.text.strip()
                else:
                    title = "SansTitre"
            
            
            title = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)
            
            
            content_div = soup.find('div', class_='mw-parser-output')
            paragraphs = content_div.find_all('p') if content_div else []
            
            content = '\n'.join(p.text.strip() for p in paragraphs if p.text.strip())
            
            
            if not content:
                content = "Aucun contenu trouvé sur cette page."
            
            
            filename = os.path.join(folder_name, f"{title}.txt")
            with open(filename, 'w', encoding='utf-8') as file:
                file.write(f"Titre: {title}\n\n")
                file.write(content)
            
        
        except requests.RequestException as e:
            print(f"Erreur lors de l'exploration de {url}: {e}")


        print(f"{i} / {len(urls)}")


def extraction() :

    liste_niveau_1 = niveau_1(URL, BASE_URL)  # liste par pays

    liste_niveau_2 = []                       # liste de ttes les recettes de ts les pays

    for i, url in enumerate(liste_niveau_1) :

        liste_niveau_2.extend(niveau_2(url))
        print(f"{round((i/len(liste_niveau_1))*100, 2)} %")


    with open("liste_wikipedia_v1.txt", 'w', encoding='utf-8') as file:
        for el in liste_niveau_2 : 
            file.write(el + "\n")

    tirage_hasard = random.sample(liste_niveau_2, 8500)         # ICI ON PREND 5500 LIENS AU HASARS PARMIS LES 22000 

    scrape_wikipedia(tirage_hasard)

if __name__ == "__main__":
    extraction()