import discord
from discord.ext import commands
import re
import requests

# Configuration du bot
TOKEN = "MTM3MDM3NTE5MTU0NDQ2NzQ2Ng.GA87T5.kE498SuVMivtGp_mq3CdCOqZ_gKWH_AjkebuV0"  # Remplace par ton token Discord
PREFIX = "!"  # Préfixe des commandes
API_URL = "https://jdm-api.demo.lirmm.fr/term/{}"

# Liste d'ingrédients et termes culinaires pour filtrer les mots pertinents
CULINARY_TERMS = [
    "tomate", "pomme de terre", "carotte", "oignon", "ail", "poivron", "courgette",
    "aubergine", "poulet", "boeuf", "porc", "poisson", "crevette", "riz", "pâtes",
    "huile", "beurre", "sel", "poivre", "sucre", "farine", "oeuf", "lait", "fromage",
    "sauce", "soupe", "salade", "pain", "vin", "épice", "herbes"
]

# Liste d'adjectifs pour détecter les caractéristiques
ADJECTIVES = [
    "rouge", "vert", "jaune", "croustillant", "moelleux", "juteux", "sucré", "salé",
    "épicé", "amer", "acide", "doux", "crémeux", "frit", "cuit", "cru"
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
async def help2(ctx):
    """Commande !help : affiche les instructions d'utilisation du bot."""
    help_message = (
        "**Aide pour le Bot de Relations Sémantiques**\n"
        "Ce bot extrait des relations sémantiques à partir d'un texte pour un mot donné.\n\n"
        "**Commandes disponibles :**\n"
        f"- `{PREFIX}relations <mot> <texte>` : Analyse le texte et affiche les relations sémantiques pour le mot.\n"
        f"  Exemple : `{PREFIX}relations tomate Émincer un oignon et cuire avec des tomates rouges.`\n"
        f"- `{PREFIX}help2` : Affiche ce message d'aide.\n\n"
        "**Notes :**\n"
        "- Le mot doit être un terme culinaire (ex. : tomate, oignon, sauce).\n"
        "- Le texte doit contenir le mot pour que des relations soient détectées.\n"
        "Pour plus d'info, contactez votre encadrant."
    )
    await ctx.send(help_message)

@bot.command()
async def relations(ctx, mot, *, texte):
    """Commande !relations <mot> <texte> : extrait les relations sémantiques pour le mot à partir du texte."""
    if not mot or not texte:
        await ctx.send(
            f"❌ Usage : `{PREFIX}relations <mot> <texte>`\n"
            f"Exemple : `{PREFIX}relations tomate Émincer un oignon et cuire avec des tomates rouges.`\n"
            f"Utilisez `{PREFIX}help` pour plus d'informations."
        )
        return

    mot = mot.lower().strip()
    texte = texte.lower().strip()

    # Vérifier si le mot est un terme culinaire
    if mot not in CULINARY_TERMS:
        await ctx.send(
            f"⚠️ Le mot « {mot} » n'est pas reconnu comme un terme culinaire.\n"
            f"Essayez avec un ingrédient ou un terme comme : {', '.join(CULINARY_TERMS[:5])}, etc.\n"
            f"Utilisez `{PREFIX}help` pour plus d'informations."
        )
        return

    # Vérifier si le mot apparaît dans le texte
    if mot not in texte:
        await ctx.send(
            f"❌ Le mot « {mot} » n'apparaît pas dans le texte fourni.\n"
            f"Utilisez `{PREFIX}help` pour plus d'informations."
        )
        return

    # Extraire les phrases du texte
    phrases = re.split(r'[.!?]+', texte)
    phrases = [p.strip() for p in phrases if p.strip() and mot in p]

    # Recherche des relations sémantiques
    relations = []
    for phrase in phrases:
        # Hyperonymie : "<mot> est un <autre_mot>"
        match_hyper = re.search(rf"{mot}\s+est\s+un\s+(\w+)", phrase)
        if match_hyper and match_hyper.group(1) in CULINARY_TERMS:
            relations.append({
                "type": "hyperonymie",
                "related": match_hyper.group(1),
                "context": phrase
            })

        # Caractéristique : "<mot> est <adjectif>"
        match_char = re.search(rf"{mot}\s+est\s+({'|'.join(ADJECTIVES)})", phrase)
        if match_char:
            relations.append({
                "type": "caractéristique",
                "related": match_char.group(1),
                "context": phrase
            })

        # Co-occurrence : autres termes culinaires dans la même phrase
        for term in CULINARY_TERMS:
            if term != mot and term in phrase:
                relations.append({
                    "type": "relation fonctionnelle",
                    "related": term,
                    "context": phrase
                })

    # Si aucune relation n'est trouvée, tenter un appel à l'API JDM (optionnel)
    if not relations:
        try:
            response = requests.get(API_URL.format(mot), timeout=5)
            response.raise_for_status()
            data = response.json()
            api_relations = data.get("relations", [])
            for rel in api_relations[:3]:  # Limiter à 3 relations
                rel_name = rel.get("name", "inconnu")
                node_name = rel.get("node", {}).get("name", "inconnu")
                rel_type = rel.get("type", "type inconnu")
                relations.append({
                    "type": rel_type,
                    "related": node_name,
                    "context": "Extrait de l'API JeuxDeMots (texte non spécifique)"
                })
        except requests.RequestException:
            pass  # Ignorer l'erreur API, on affiche juste les résultats locaux

    # Afficher les résultats
    if not relations:
        await ctx.send(f"ℹ️ Aucune relation sémantique trouvée pour « {mot} » dans le texte ou via l'API.")
        return

    resultats = [f"**Relations sémantiques pour « {mot} » :**"]
    for rel in relations[:5]:  # Limiter à 5 relations
        ligne = f"🔗 {rel['type']} → **{rel['related']}**\n   📜 Contexte : « {rel['context']} »"
        resultats.append(ligne)

    await ctx.send("\n".join(resultats))

@bot.event
async def on_command_error(ctx, error):
    """Gestion des erreurs de commande : affiche un message d'aide."""
    if isinstance(error, commands.CommandNotFound):
        await ctx.send(
            f"❌ Commande inconnue. Utilisez `{PREFIX}help` pour voir les commandes disponibles."
        )
    elif isinstance(error, commands.MissingRequiredArgument):
        await ctx.send(
            f"❌ Arguments manquants. Exemple : `{PREFIX}relations tomate Émincer un oignon et cuire avec des tomates rouges.`\n"
            f"Utilisez `{PREFIX}help` pour plus d'informations."
        )
    else:
        await ctx.send(
            f"❌ Une erreur s'est produite : {str(error)}\n"
            f"Utilisez `{PREFIX}help` pour plus d'informations."
        )

# Lancement du bot
bot.run(TOKEN)