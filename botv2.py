import discord
import requests
from discord.ext import commands
import re

# Configuration du bot
TOKEN = "MTM3MDM3NTE5MTU0NDQ2NzQ2Ng.GA87T5.kE498SuVMivtGp_mq3CdCOqZ_gKWH_AjkebuV0"  # Remplace par ton token Discord
PREFIX = "!"  # Préfixe des commandes
API_URL = "https://jdm-api.demo.lirmm.fr/term/{}"

# Liste simple d'ingrédients pour identifier les mots clés dans une recette
INGREDIENTS = [
    "tomate", "pomme de terre", "carotte", "oignon", "ail", "poivron", "courgette",
    "aubergine", "poulet", "boeuf", "porc", "poisson", "crevette", "riz", "pâtes",
    "huile", "beurre", "sel", "poivre", "sucre", "farine", "oeuf", "lait", "fromage"
]

# Initialisation du bot avec intents
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix=PREFIX, intents=intents)

@bot.event
async def on_ready():
    """Événement déclenché lorsque le bot se connecte à Discord."""
    print(f"✅ Bot connecté en tant que {bot.user}")

@bot.command()
async def jdm(ctx, *, mot):
    """Commande !jdm <mot> : récupère les relations sémantiques pour un mot via l'API JDM."""
    if not mot:
        await ctx.send("❌ Tu dois écrire un mot après `!jdm`. Exemple : `!jdm tomate`")
        return

    # Appel à l'API JDM
    url = API_URL.format(mot.strip())
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()  # Vérifie si la requête a réussi
        data = response.json()
    except requests.RequestException as e:
        await ctx.send(f"❌ Erreur lors de l'appel à l'API : {str(e)}")
        return

    # Extraction des relations
    relations = data.get("relations", [])
    if not relations:
        await ctx.send(f"ℹ️ Aucune relation trouvée pour « {mot} ».")
        return

    # Formatage des résultats (limité à 5 relations pour éviter un message trop long)
    resultats = [f"**Relations pour « {mot} » :**"]
    for rel in relations[:5]:
        rel_name = rel.get("name", "inconnu")
        node_name = rel.get("node", {}).get("name", "inconnu")
        rel_type = rel.get("type", "type inconnu")
        ligne = f"🔗 {rel_name} → **{node_name}** ({rel_type})"
        resultats.append(ligne)

    await ctx.send("\n".join(resultats))

@bot.command()
async def recipe(ctx, *, recette):
    """Commande !recipe <recette> : analyse une recette et affiche les relations pour un mot clé."""
    if not recette:
        await ctx.send("❌ Tu dois fournir une recette. Exemple : `!recipe Émincer un oignon et cuire avec des tomates.`")
        return

    # Nettoyage et tokenisation simple de la recette
    recette = recette.lower().strip()
    mots = re.findall(r'\b\w+\b', recette)  # Extrait les mots

    # Recherche d'un ingrédient dans la recette
    mot_cle = None
    for mot in mots:
        if mot in INGREDIENTS:
            mot_cle = mot
            break

    if not mot_cle:
        await ctx.send("ℹ️ Aucun ingrédient reconnu dans la recette. Essaye avec des termes comme 'tomate', 'oignon', etc.")
        return

    # Recherche du contexte du mot clé dans la recette
    phrases = re.split(r'[.!?]', recette)
    contexte = None
    for phrase in phrases:
        if mot_cle in phrase:
            contexte = phrase.strip()
            break

    # Appel à l'API JDM pour le mot clé
    url = API_URL.format(mot_cle)
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        await ctx.send(f"❌ Erreur lors de l'appel à l'API pour « {mot_cle} » : {str(e)}")
        return

    # Extraction des relations
    relations = data.get("relations", [])
    if not relations:
        await ctx.send(f"ℹ️ Aucune relation trouvée pour « {mot_cle} ».")
        return

    # Formatage des résultats
    resultats = [f"**Relations pour « {mot_cle} » (trouvé dans la recette) :**"]
    if contexte:
        resultats.append(f"📜 Contexte : « {contexte} »")
    for rel in relations[:5]:
        rel_name = rel.get("name", "inconnu")
        node_name = rel.get("node", {}).get("name", "inconnu")
        rel_type = rel.get("type", "type inconnu")
        ligne = f"🔗 {rel_name} → **{node_name}** ({rel_type})"
        resultats.append(ligne)

    await ctx.send("\n".join(resultats))

# Lancement du bot
bot.run(TOKEN)

