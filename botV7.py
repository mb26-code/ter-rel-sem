import discord
from discord.ext import commands
import re
import requests
import csv
from io import StringIO




def load_semantic_relations(filepath):
    relations_db = []
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            relations_db.append({
                "node1": row["node1"],
                "node2": row["node2"],
                "relation": row.get("best_relation", "inconnue"),
                "dep": row.get("dep", ""),
                "pos": row.get("pos", ""),
                "sim": float(row.get("sim", 0))
            })
    return relations_db


# Configuration du bot
TOKEN = "MTM3MDM3NTE5MTU0NDQ2NzQ2Ng.GA87T5.kE498SuVMivtGp_mq3CdCOqZ_gKWH_AjkebuV0"
PREFIX = "!"
API_URL = "https://jdm-api.demo.lirmm.fr/term/{}"



# Charger la base de données des relations sémantiques
def load_semantic_relations(csv_data):
    relations_db = {}
    csv_file = StringIO(csv_data)
    reader = csv.DictReader(csv_file, delimiter=',')
    for row in reader:
        node1, node2 = row['node1'], row['node2']
        best_rel = row['best_relation']
        weight = float(row['best_relation_w']) if row['best_relation_w'] else 0.0
        if node1 and node2 and best_rel:
            if node1 not in relations_db:
                relations_db[node1] = {}
            relations_db[node1][node2] = {'relation': best_rel, 'weight': weight}
    return relations_db










# Initialisation du bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix=PREFIX, intents=intents)

@bot.event
async def on_ready():
    print(f" Bot connecté en tant que {bot.user}")

@bot.command()
async def help2(ctx):
    help_message = (
        "**Aide pour le Bot Gastronomique**\n"
        "Ce bot extrait des relations sémantiques culinaires à partir d'un texte pour un mot donné.\n\n"
        "**Commandes disponibles :**\n"
        f"- `{PREFIX}relations <mot> :: <texte>` : Analyse le texte et affiche les relations sémantiques pour le mot.\n"
        f"  Exemple : `{PREFIX}relations pomme de terre :: éplucher des pommes de terre avec un couteau`\n"
        f"  Relations possibles : r.instr, r.manner, r.agent, r.object, r.location, etc.\n"
        f"- `{PREFIX}help2` : Affiche ce message d'aide.\n\n"
        "**Notes :**\n"
        "- Le mot peut être un terme simple (ex. : tomate) ou composé (ex. : pomme de terre).\n"
        "- Le texte doit contenir le mot et des indices de relations (ex. : 'avec', 'dans').\n"
        "- Les relations utilisent une base de données sémantique et des motifs simples.\n"
        "Pour plus d'info, contactez votre encadrant."
    )
    await ctx.send(help_message)

@bot.command()
async def relations(ctx, *, input_str):
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

    if mot not in CULINARY_TERMS:
        await ctx.send(
            f"⚠ Le mot « {mot} » n'est pas reconnu comme un terme culinaire.\n"
            f"Essayez avec : {', '.join(CULINARY_TERMS[:5])}, etc.\n"
            f"Utilisez `{PREFIX}help2` pour plus d'informations."
        )
        return

    if mot not in texte:
        await ctx.send(
            f" Le mot « {mot} » n'apparaît pas dans le texte fourni.\n"
            f"Utilisez `{PREFIX}help2` pour plus d'informations."
        )
        return

    # Extraire les termes du texte
    text_terms = [term for term in CULINARY_TERMS if term in texte and term != mot]
    relations = []

    SEMANTIC_RELATIONS = load_semantic_relations("semantic_relations_rows.csv")



    # Utiliser la base de données pour les relations
    if mot in SEMANTIC_RELATIONS:
        for term in text_terms:
            if term in SEMANTIC_RELATIONS[mot]:
                rel = SEMANTIC_RELATIONS[mot][term]
                relations.append({
                    "type": rel['relation'],
                    "related": term,
                    "weight": rel['weight'],
                    "context": texte
                })

    # Si aucune relation trouvée dans la base, utiliser les motifs regex
    if not relations:
        phrases = re.split(r'[.!?]+', texte)
        phrases = [p.strip() for p in phrases if p.strip() and mot in p]
        for phrase in phrases:
            match_instr = re.search(rf"avec\s+(un|une|à\s+l'aide\s+de)\s+({'|'.join(CULINARY_TERMS)})", phrase)
            if match_instr and match_instr.group(2) != mot and match_instr.group(2) in text_terms:
                relations.append({"type": "r.instr", "related": match_instr.group(2), "context": phrase})

            match_manner = re.search(rf"(avec|en\s+utilisant)\s+({'|'.join(CULINARY_TERMS)})", phrase)
            if match_manner and match_manner.group(2) != mot and match_manner.group(2) in text_terms:
                relations.append({"type": "r.manner", "related": match_manner.group(2), "context": phrase})

            match_agent = re.search(rf"par\s+({'|'.join(CULINARY_TERMS)})|({'|'.join(CULINARY_TERMS)})\s+(épluche|cuit)", phrase)
            if match_agent and match_agent.group(1) and match_agent.group(1) != mot and match_agent.group(1) in text_terms:
                relations.append({"type": "r.agent", "related": match_agent.group(1), "context": phrase})

            match_object = re.search(rf"{re.escape(mot)}\s+(épluche|cuit|coupe)\s+({'|'.join(CULINARY_TERMS)})", phrase)
            if match_object and match_object.group(2) != mot and match_object.group(2) in text_terms:
                relations.append({"type": "r.object", "related": match_object.group(2), "context": phrase})

            match_loc = re.search(rf"(dans|sur)\s+({'|'.join(CULINARY_TERMS)})", phrase)
            if match_loc and match_loc.group(2) != mot and match_loc.group(2) in text_terms:
                relations.append({"type": "r.location", "related": match_loc.group(2), "context": phrase})

    # Si toujours aucune relation, tenter l'API JDM
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
    for rel in sorted(relations, key=lambda x: x.get('weight', 0), reverse=True)[:5]:
        ligne = f" {mot} {rel['type']} {rel['related']}"
        if 'weight' in rel:
            ligne += f" (poids: {rel['weight']:0.1f})"
        ligne += f"\n    Contexte : « {rel['context']} »"
        resultats.append(ligne)

    await ctx.send("\n".join(resultats))

@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound):
        await ctx.send(f" Commande inconnue. Utilisez `{PREFIX}help2` pour voir les commandes disponibles.")
    elif isinstance(error, commands.MissingRequiredArgument):
        await ctx.send(
            f" Arguments manquants. Exemple : `{PREFIX}relations pomme de terre :: éplucher des pommes de terre avec un couteau`\n"
            f"Utilisez `{PREFIX}help2` pour plus d'informations."
        )
    else:
        await ctx.send(f" Une erreur s'est produite : {str(error)}\nUtilisez `{PREFIX}help2` pour plus d'informations.")

# Lancement du bot
bot.run(TOKEN)