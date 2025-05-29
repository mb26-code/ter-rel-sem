import wikipedia
import nltk
import requests
from bs4 import BeautifulSoup


#extraire les articles wikipedia
wikipedia.set_lang("fr")
results = wikipedia.search("Cuisine française")
print(results)



#extraire les articles marmiton
url = "https://www.marmiton.org/recettes/recette_tarte-aux-pommes_12345.aspx"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")
recette = soup.find("div", class_="recipe-step-list").text
print(recette)




from nltk.corpus import stopwords

nltk.download("stopwords")

stop_words = set(stopwords.words("french"))
print(stop_words)
