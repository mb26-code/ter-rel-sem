import discord
from discord.ext import commands
import re
import requests
import csv

# Configuration du bot
TOKEN = "MTM3MDM3NTE5MTU0NDQ2NzQ2Ng.GA87T5.kE498SuVMivtGp_mq3CdCOqZ_gKWH_AjkebuV0"
PREFIX = "!"  # Préfixe des commandes
API_URL = "https://jdm-api.demo.lirmm.fr/term/{}"

# Liste de termes culinaires (ingrédients, ustensiles, lieux), incluant des expressions multi-mots
CULINARY_TERMS = [
    "tomate", "pomme de terre", "carotte", "oignon", "ail", "poivron", "courgette",
    "aubergine", "poulet", "boeuf", "porc", "poisson", "crevette", "riz", "pâtes",
    "huile", "beurre", "sel", "poivre", "sucre", "farine", "oeuf", "lait", "fromage",
    "sauce", "soupe", "salade", "pain", "vin", "épice", "herbes", "couteau", "cuillère",
    "four", "poêle", "casserole", "frigo", "cuisine", "table"
]

semantic_db = []

with open("semantic_relations_rows.csv", newline='', encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        semantic_db.append({
            "mot1": row["node"],
            "mot2": row["node2"],
            "relation": row.get("relation_name") or "inconnue"
        })

# Initialisation du bot avec intents
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix=PREFIX, intents=intents)

@bot.event
async def on_ready():
    """Événement déclenché lorsque le bot se connecte à Discord."""
    print(f" Bot connecté en tant que {bot.user}")

@bot.command()
async def help2(ctx):
    """Commande !help2 : affiche les instructions d'utilisation du bot."""
    help_message = (
        "**Aide pour le Bot Gastronomique**\n"
        "Ce bot extrait des relations sémantiques culinaires à partir d'un texte pour un mot donné.\n\n"
        "**Commandes disponibles :**\n"
        f"- `{PREFIX}relations <mot> :: <texte>` : Analyse le texte et affiche les relations sémantiques pour le mot.\n"
        f"  Exemple : `{PREFIX}relations pomme de terre :: éplucher des pommes de terre avec un couteau`\n"
        f"  Relations possibles : r.instr, r.manner, r.agent, r.object, r.location.\n"
        f"- `{PREFIX}help2` : Affiche ce message d'aide.\n\n"
        "**Notes :**\n"
        "- Le mot peut être un terme simple (ex. : tomate) ou composé (ex. : pomme de terre).\n"
        "- Le texte doit contenir le mot et des indices de relations (ex. : 'avec', 'dans').\n"
        "- Les relations sont basées sur des motifs simples (ex. : 'avec un couteau' → r.instr).\n"
        "Pour plus d'info, contactez votre encadrant."
    )
    await ctx.send(help_message)

@bot.command()
async def relations(ctx, *, input_str):
    """Commande !relations <mot> :: <texte> : extrait les relations sémantiques pour le mot à partir du texte."""
    # Séparer le mot et le texte avec le séparateur ::
    if "::" not in input_str:
        await ctx.send(
            f" Usage : `{PREFIX}relations <mot> :: <texte>`\n"
            f"Exemple : `{PREFIX}relations pomme de terre :: éplucher des pommes de terre avec un couteau`\n"
            f"Utilisez `{PREFIX}help2` pour plus d'informations."
        )
        return

    mot, texte = [part.strip() for part in input_str.split("::", 1)]
    if not mot or not texte:
        await ctx.send(
            f" Usage : `{PREFIX}relations <mot> :: <texte>`\n"
            f"Exemple : `{PREFIX}relations pomme de terre :: éplucher des pommes de terre avec un couteau`\n"
            f"Utilisez `{PREFIX}help2` pour plus d'informations."
        )
        return

    mot = mot.lower().strip()
    texte = texte.lower().strip()

    # Vérifier si le mot est un terme culinaire (simple ou composé)
    if mot not in CULINARY_TERMS:
        await ctx.send(
            f"⚠ Le mot « {mot} » n'est pas reconnu comme un terme culinaire.\n"
            f"Essayez avec : {', '.join(CULINARY_TERMS[:5])}, etc.\n"
            f"Utilisez `{PREFIX}help2` pour plus d'informations."
        )
        return

    # Vérifier si le mot apparaît dans le texte (même sous forme composée)
    if mot not in texte:
        await ctx.send(
            f" Le mot « {mot} » n'apparaît pas dans le texte fourni.\n"
            f"Utilisez `{PREFIX}help2` pour plus d'informations."
        )
        return

    # Extraire les phrases du texte
    phrases = re.split(r'[.!?]+', texte)
    phrases = [p.strip() for p in phrases if p.strip() and mot in p]

    # Recherche des relations sémantiques
    relations = []
    for phrase in phrases:
        # r.instr : avec un/une <instrument> ou à l'aide de <instrument>
        match_instr = re.search(rf"avec\s+(un|une|à\s+l'aide\s+de)\s+({'|'.join(CULINARY_TERMS)})", phrase)
        if match_instr and match_instr.group(2) != mot:
            relations.append({
                "type": "r.instr",
                "related": match_instr.group(2),
                "context": phrase
            })

        # r.manner : avec <manière> ou en utilisant <manière>
        match_manner = re.search(rf"(avec|en\s+utilisant)\s+({'|'.join(CULINARY_TERMS)})", phrase)
        if match_manner and match_manner.group(2) != mot:
            relations.append({
                "type": "r.manner",
                "related": match_manner.group(2),
                "context": phrase
            })

        # r.agent : par <agent> ou <agent> <verbe>
        match_agent = re.search(rf"par\s+({'|'.join(CULINARY_TERMS)})|({'|'.join(CULINARY_TERMS)})\s+(épluche|cuit)", phrase)
        if match_agent and match_agent.group(1) and match_agent.group(1) != mot:
            relations.append({
                "type": "r.agent",
                "related": match_agent.group(1),
                "context": phrase
            })

        # r.object : <mot> <verbe> <objet>
        match_object = re.search(rf"{re.escape(mot)}\s+(épluche|cuit|coupe)\s+({'|'.join(CULINARY_TERMS)})", phrase)
        if match_object and match_object.group(2) != mot:
            relations.append({
                "type": "r.object",
                "related": match_object.group(2),
                "context": phrase
            })

        # r.location : dans <lieu> ou sur <lieu>
        match_loc = re.search(rf"(dans|sur)\s+({'|'.join(CULINARY_TERMS)})", phrase)
        if match_loc and match_loc.group(2) != mot:
            relations.append({
                "type": "r.location",
                "related": match_loc.group(2),
                "context": phrase
            })

    # Si aucune relation n'est trouvée, tenter un appel à l'API JDM
    if not relations:
        try:
            response = requests.get(API_URL.format(mot.replace(" ", "_")), timeout=5)
            response.raise_for_status()
            data = response.json()
            api_relations = data.get("relations", [])
            for rel in api_relations[:3]:
                rel_type = rel.get("type", "type inconnu")
                node_name = rel.get("node", {}).get("name", "inconnu").replace("_", " ")
                if node_name != mot and rel_type in ["r.instr", "r.manner", "r.agent", "r.object", "r.location"]:
                    relations.append({
                        "type": rel_type,
                        "related": node_name,
                        "context": "Extrait de l'API JeuxDeMots (texte non spécifique)"
                    })
        except requests.RequestException:
            pass

    # Afficher les résultats
    if not relations:
        await ctx.send(f"ℹ Aucune relation sémantique trouvée pour « {mot} » dans le texte ou via l'API.")
        return

    resultats = [f"**Relations sémantiques pour « {mot} » :**"]
    for rel in relations[:5]:
        ligne = f" {mot} {rel['type']} {rel['related']}\n    Contexte : « {rel['context']} »"
        resultats.append(ligne)

    await ctx.send("\n".join(resultats))

@bot.event
async def on_command_error(ctx, error):
    """Gestion des erreurs de commande : affiche un message d'aide."""
    if isinstance(error, commands.CommandNotFound):
        await ctx.send(
            f" Commande inconnue. Utilisez `{PREFIX}help2` pour voir les commandes disponibles."
        )
    elif isinstance(error, commands.MissingRequiredArgument):
        await ctx.send(
            f" Arguments manquants. Exemple : `{PREFIX}relations pomme de terre :: éplucher des pommes de terre avec un couteau`\n"
            f"Utilisez `{PREFIX}help2` pour plus d'informations."
        )
    else:
        await ctx.send(
            f" Une erreur s'est produite : {str(error)}\n"
            f"Utilisez `{PREFIX}help2` pour plus d'informations."
        )

# Lancement du bot
bot.run(TOKEN)