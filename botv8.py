import discord
from discord.ext import commands
import re
import requests
import csv
from fuzzywuzzy import fuzz  # For similarity matching

# Configuration du bot
TOKEN = "MTM3MDM3NTE5MTU0NDQ2NzQ2Ng.GA87T5.kE498SuVMivtGp_mq3CdCOqZ_gKWH_AjkebuV0"
PREFIX = "!"
API_URL = "https://jdm-api.demo.lirmm.fr/term/{}"

from generated_culinary_terms import CULINARY_TERMS


# Charger la base de données des relations sémantiques depuis un fichier CSV
def load_semantic_relations(filepath):
    relations_db = {}
    try:
        with open(filepath, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                node1, node2 = row['node1'].lower(), row['node2'].lower()
                best_rel = row.get('best_relation', '')
                weight = float(row.get('best_relation_w', 0)) if row.get('best_relation_w') else 0.0
                sim = float(row.get('sim', 0)) if row.get('sim') else 0.0
                dep = row.get('dep', '')
                pos = row.get('pos', '')
                if node1 and node2 and best_rel:
                    if node1 not in relations_db:
                        relations_db[node1] = {}
                    relations_db[node1][node2] = {
                        'relation': best_rel,
                        'weight': weight,
                        'sim': sim,
                        'dep': dep,
                        'pos': pos
                    }
    except FileNotFoundError:
        print(f"Erreur : Le fichier {filepath} n'a pas été trouvé.")
        return {}
    except Exception as e:
        print(f"Erreur lors du chargement du fichier CSV : {str(e)}")
        return {}
    return relations_db

# Initialisation du bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix=PREFIX, intents=intents)

# Charger la base de données au démarrage
SEMANTIC_RELATIONS = load_semantic_relations("semantic_relations_rows.csv")

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
        "- Les relations sont basées sur une base de données sémantique et des motifs avancés.\n"
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

    # Vérifier si le mot est un terme culinaire
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

    # 1. Recherche directe dans la base de données
    if mot in SEMANTIC_RELATIONS:
        for term in text_terms:
            if term in SEMANTIC_RELATIONS[mot]:
                rel = SEMANTIC_RELATIONS[mot][term]
                if rel['weight'] >= 20:  # Seuil de poids pour filtrer les relations faibles
                    relations.append({
                        "type": rel['relation'],
                        "related": term,
                        "weight": rel['weight'],
                        "context": texte,
                        "source": "database"
                    })

    # 2. Recherche par similarité si aucune relation directe n'est trouvée
    if not relations:
        best_sim = 0
        best_match = None
        for db_node1 in SEMANTIC_RELATIONS:
            sim_score = fuzz.ratio(mot, db_node1)
            if sim_score > best_sim and sim_score >= 80:  # Seuil de similarité
                best_sim = sim_score
                best_match = db_node1

        if best_match:
            for term in text_terms:
                if term in SEMANTIC_RELATIONS[best_match]:
                    rel = SEMANTIC_RELATIONS[best_match][term]
                    if rel['weight'] >= 20:
                        relations.append({
                            "type": rel['relation'],
                            "related": term,
                            "weight": rel['weight'] * (best_sim / 100),  # Ajuster le poids selon la similarité
                            "context": texte,
                            "source": f"database (similar term: {best_match}, sim: {best_sim}%)"
                        })

    # 3. Analyse contextuelle avec motifs regex améliorés
    phrases = re.split(r'[.!?]+', texte)
    phrases = [p.strip() for p in phrases if p.strip() and mot in p]
    for phrase in phrases:
        for term in text_terms:
            # r.instr: "avec un/une", "à l'aide de"
            if re.search(rf"avec\s+(un|une|à\s+l'aide\s+de)\s+{re.escape(term)}", phrase):
                weight = 30 if "à l'aide de" in phrase else 25  # Poids basé sur la force du motif
                relations.append({"type": "r.instr", "related": term, "weight": weight, "context": phrase, "source": "regex"})

            # r.manner: "de manière", "en utilisant"
            if re.search(rf"(de\s+manière|en\s+utilisant)\s+{re.escape(term)}", phrase):
                relations.append({"type": "r.manner", "related": term, "weight": 25, "context": phrase, "source": "regex"})

            # r.agent: "par", ou sujet/verbe
            if re.search(rf"par\s+{re.escape(term)}", phrase) or re.search(rf"{re.escape(term)}\s+(épluche|cuit|prépare)", phrase):
                relations.append({"type": "r.agent", "related": term, "weight": 30, "context": phrase, "source": "regex"})

            # r.object: "<mot> <verbe> <objet>"
            if re.search(rf"{re.escape(mot)}\s+(épluche|cuit|coupe|prépare)\s+{re.escape(term)}", phrase):
                relations.append({"type": "r.object", "related": term, "weight": 30, "context": phrase, "source": "regex"})

            # r.location: "dans", "sur"
            if re.search(rf"(dans|sur)\s+{re.escape(term)}", phrase):
                relations.append({"type": "r.location", "related": term, "weight": 25, "context": phrase, "source": "regex"})

    # 4. Si aucune relation n'est trouvée, tenter l'API JDM
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
                        "weight": 20,  # Poids par défaut pour l'API
                        "context": "Extrait de l'API JeuxDeMots (texte non spécifique)",
                        "source": "api"
                    })
        except requests.RequestException:
            pass

    # 5. Afficher les résultats
    if not relations:
        feedback = (
            f"ℹ Aucune relation sémantique trouvée pour « {mot} ».\n"
            "- Vérifiez si le mot ou les termes associés sont dans la base de données.\n"
            "- Essayez un texte avec des indices clairs (ex. : 'avec', 'dans', 'cuit par').\n"
            f"Utilisez `{PREFIX}help2` pour plus d'informations."
        )
        await ctx.send(feedback)
        return

    resultats = [f"**Relations sémantiques pour « {mot} » :**"]
    for rel in sorted(relations, key=lambda x: x.get('weight', 0), reverse=True)[:5]:
        ligne = f" {mot} {rel['type']} {rel['related']} (poids: {rel.get('weight', 0):0.1f}, source: {rel['source']})"
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