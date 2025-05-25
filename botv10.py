import discord
from discord.ext import commands
import re
import requests
import csv
from fuzzywuzzy import fuzz
from datetime import datetime
import spacy
import random
import os

# Configuration du bot
TOKEN = os.getenv("DISCORD_TOKEN") or "MTM3MDM3NTE5MTU0NDQ2NzQ2Ng.GA87T5.kE498SuVMivtGp_mq3CdCOqZ_gKWH_AjkebuV0"  # Utilise une variable d'environnement
PREFIX = "!"
API_URL = "https://jdm-api.demo.lirmm.fr/term/{}"

# Liste des termes culinaires (remplace generated_culinary_terms)
CULINARY_TERMS = [
    "tomate", "pomme de terre", "carotte", "oignon", "ail", "poivron", "courgette",
    "aubergine", "poulet", "boeuf", "porc", "poisson", "crevette", "riz", "pâtes",
    "huile", "beurre", "sel", "poivre", "sucre", "farine", "oeuf", "lait", "fromage",
    "sauce", "soupe", "salade", "pain", "vin", "épice", "herbes"
]

# Modèles pour textes aléatoires
TEXT_TEMPLATES = [
    "Préparer une {plat} avec {ingr1} et {ingr2} dans un {ustensile}.",
    "Cuire {ingr1} avec {ingr2} et une pincée de {epice}.",
    "{ingr1} est utilisé pour faire une {plat} savoureuse dans un {ustensile}.",
    "Mélanger {ingr1}, {ingr2} et {ingr3} pour obtenir une {plat}.",
    "Dans une {ustensile}, faire revenir {ingr1} avec {epice} et {ingr2}."
]
PLATS = ["soupe", "sauce", "salade", "ragoût", "tarte"]
USTENSILES = ["poêle", "casserole", "four", "saladier", "mixeur"]
EPICES = ["sel", "poivre", "curcuma", "paprika", "cannelle"]

# Charger le modèle SpaCy
try:
    nlp = spacy.load("fr_core_news_sm")
except Exception as e:
    print(f"Erreur lors du chargement de SpaCy : {str(e)}")
    nlp = None

# Charger la base de données des relations sémantiques
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
    except Exception as e:
        print(f"Erreur lors du chargement du fichier CSV : {str(e)}")
    return relations_db

# Enregistrer les relations apprises
def save_learned_relation(mot, texte, relation):
    try:
        with open("learned_relations.csv", "a", newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([timestamp, mot, relation["related"], relation["type"], texte])
    except Exception as e:
        print(f"Erreur lors de l'enregistrement de la relation : {str(e)}")

# Remplacement de find_fuzzy_match
def find_fuzzy_match(mot, texte):
    best_score, best_term = 0, None
    for term in CULINARY_TERMS:
        score = fuzz.ratio(mot, term)
        if score > best_score and score >= 80:
            best_score, best_term = score, term
    return best_term if best_term in texte else mot

# Remplacement de find_all_fuzzy_terms
def find_all_fuzzy_terms(terms, texte, exclude=None):
    matched_terms = []
    for term in terms:
        if term != exclude and (term in texte or any(fuzz.ratio(term, word) >= 80 for word in texte.split())):
            matched_terms.append(term)
    return matched_terms

# Extraction des relations
def extract_relations(mot, texte):
    relations = []
    text_terms = find_all_fuzzy_terms(CULINARY_TERMS, texte, exclude=mot)

    # 1. Analyse SpaCy
    if nlp:
        doc = nlp(texte)
        for token in doc:
            if token.text.lower() == mot:
                for child in token.children:
                    related = child.text.lower()
                    if related in text_terms:
                        dep = child.dep_
                        rel_type = {
                            'dobj': 'r.object',
                            'pobj': 'r.location',
                            'nmod': 'r.associated',
                            'nsubj': 'r.agent'
                        }.get(dep, 'r.associated')
                        relations.append({
                            'type': rel_type,
                            'related': related,
                            'weight': 40 if dep in ['dobj', 'nsubj'] else 35,
                            'context': texte,
                            'source': 'spacy'
                        })
                for token_in_phrase in doc:
                    if token_in_phrase.dep_ == 'prep' and token_in_phrase.head.text.lower() == mot:
                        for grandchild in token_in_phrase.children:
                            if grandchild.text.lower() in text_terms:
                                rel_type = 'r.instr' if token_in_phrase.text == 'avec' else 'r.location'
                                relations.append({
                                    'type': rel_type,
                                    'related': grandchild.text.lower(),
                                    'weight': 40,
                                    'context': texte,
                                    'source': 'spacy'
                                })

    # 2. Base de données
    if mot in SEMANTIC_RELATIONS:
        for term in text_terms:
            if term in SEMANTIC_RELATIONS[mot]:
                rel = SEMANTIC_RELATIONS[mot][term]
                weight = rel['weight'] * 0.6 + rel['sim'] * 0.4
                if weight >= 20:
                    relations.append({
                        'type': rel['relation'],
                        'related': term,
                        'weight': weight,
                        'context': texte,
                        'source': 'database'
                    })

    # 3. Similarité avec fuzzywuzzy
    best_sim, best_match = 0, None
    for db_node1 in SEMANTIC_RELATIONS:
        sim_score = fuzz.ratio(mot, db_node1)
        if sim_score > best_sim and sim_score >= 80:
            best_sim, best_match = sim_score, db_node1
    if best_match:
        for term in text_terms:
            if term in SEMANTIC_RELATIONS[best_match]:
                rel = SEMANTIC_RELATIONS[best_match][term]
                weight = rel['weight'] * 0.6 + rel['sim'] * 0.4 + (best_sim / 100) * 10
                if weight >= 20:
                    relations.append({
                        'type': rel['relation'],
                        'related': term,
                        'weight': weight,
                        'context': texte,
                        'source': f'database (similar term: {best_match}, sim: {best_sim}%)'
                    })

    # 4. Motifs regex
    phrases = re.split(r'[.!?]+', texte)
    phrases = [p.strip() for p in phrases if p.strip() and mot in p]
    for phrase in phrases:
        for term in text_terms:
            if re.search(rf"avec\s+(un|une|à\s+l'aide\s+de)\s+{re.escape(term)}", phrase):
                weight = 30 if "à l'aide de" in phrase else 25
                boost = 10 if term in SEMANTIC_RELATIONS.get(mot, {}) and SEMANTIC_RELATIONS[mot][term]['dep'] in ['nmod', 'prep'] else 0
                relations.append({'type': 'r.instr', 'related': term, 'weight': weight + boost, 'context': phrase, 'source': 'regex'})
            if re.search(rf"(dans|sur)\s+{re.escape(term)}", phrase):
                relations.append({'type': 'r.location', 'related': term, 'weight': 25, 'context': phrase, 'source': 'regex'})
            if re.search(rf"{re.escape(mot)}\s+(épluche|cuit|coupe|prépare)\s+{re.escape(term)}", phrase):
                relations.append({'type': 'r.object', 'related': term, 'weight': 30, 'context': phrase, 'source': 'regex'})
            if re.search(rf"(de\s+manière|en\s+utilisant)\s+{re.escape(term)}", phrase):
                relations.append({'type': 'r.manner', 'related': term, 'weight': 25, 'context': phrase, 'source': 'regex'})

    # 5. API JDM (si aucune relation)
    if not relations:
        try:
            response = requests.get(API_URL.format(mot.replace(" ", "_")), timeout=5)
            response.raise_for_status()
            data = response.json()
            for rel in data.get("relations", [])[:3]:
                rel_type = rel.get("type", "type inconnu")
                node_name = rel.get("node", {}).get("name", "inconnu").replace("_", " ")
                if node_name != mot and rel_type in ["r.instr", "r.manner", "r.agent", "r.object", "r.location"]:
                    relations.append({
                        'type': rel_type,
                        'related': node_name,
                        'weight': 20,
                        'context': "Extrait de l'API JeuxDeMots",
                        'source': 'api'
                    })
        except requests.RequestException:
            pass

    # Filtrer les relations (éliminer les doublons)
    unique_relations = {}
    for rel in relations:
        key = (rel['type'], rel['related'])
        if key not in unique_relations or rel['weight'] > unique_relations[key]['weight']:
            unique_relations[key] = rel
    return list(unique_relations.values())

# Initialisation du bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix=PREFIX, intents=intents)

# Charger la base de données
SEMANTIC_RELATIONS = load_semantic_relations("semantic_relations_rows.csv")
pending_add_term = {}  # clé : nom d’utilisateur, valeur : (mot, texte)

@bot.event
async def on_ready():
    print(f"✅ Bot connecté en tant que {bot.user}")

@bot.command()
async def help2(ctx):
    """Commande !help2 : affiche les instructions d'utilisation."""
    embed = discord.Embed(
        title="Aide pour le Bot Gastronomique",
        description="Ce bot extrait des relations sémantiques culinaires à partir de textes.",
        color=discord.Color.green()
    )
    embed.add_field(
        name="Commandes disponibles",
        value=(
            f"- `{PREFIX}relations <mot> :: <texte>` : Analyse le texte pour les relations du mot.\n"
            f"  Ex. : `{PREFIX}relations tomate :: Cuire des tomates avec de l'huile.`\n"
            f"- `{PREFIX}random` : Analyse un texte aléatoire avec un mot culinaire aléatoire.\n"
            f"- `{PREFIX}oui` : Confirme l'ajout d'un nouveau terme culinaire.\n"
            f"- `{PREFIX}non` : Annule l'ajout d'un terme.\n"
            f"- `{PREFIX}help2` : Affiche cette aide."
        ),
        inline=False
    )
    embed.add_field(
        name="Notes",
        value=(
            "- Les mots doivent être des termes culinaires (ex. : tomate, sauce).\n"
            "- Les relations incluent r.instr, r.manner, r.agent, r.object, r.location.\n"
            "- Contactez votre encadrant pour plus d'info."
        ),
        inline=False
    )
    await ctx.send(embed=embed)

@bot.command()
async def oui(ctx):
    """Confirme l'ajout d'un terme culinaire."""
    user_id = ctx.author.id
    if user_id not in pending_add_term:
        await ctx.send(embed=discord.Embed(
            title="Erreur",
            description="Aucune demande d'ajout en attente.",
            color=discord.Color.red()
        ))
        return

    mot, _ = pending_add_term.pop(user_id)
    if mot not in CULINARY_TERMS:
        CULINARY_TERMS.append(mot)
        await ctx.send(embed=discord.Embed(
            title="Succès",
            description=f"Le mot « {mot} » a été ajouté à la liste des termes culinaires.",
            color=discord.Color.green()
        ))
    else:
        await ctx.send(embed=discord.Embed(
            title="Erreur",
            description=f"Le mot « {mot} » est déjà dans la liste des termes culinaires.",
            color=discord.Color.red()
        ))

@bot.command()
async def non(ctx):
    """Annule l'ajout d'un terme culinaire."""
    if ctx.author.id in pending_add_term:
        pending_add_term.pop(ctx.author.id)
        await ctx.send(embed=discord.Embed(
            title="Annulation",
            description="Demande d'ajout annulée.",
            color=discord.Color.orange()
        ))
    else:
        await ctx.send(embed=discord.Embed(
            title="Erreur",
            description="Aucune demande d'ajout en attente.",
            color=discord.Color.red()
        ))

@bot.command()
async def relations(ctx, *, input_str):
    """Commande !relations <mot> :: <texte> : extrait les relations sémantiques."""
    if "::" not in input_str:
        await ctx.send(embed=discord.Embed(
            title="Erreur",
            description=(
                f"Usage : `{PREFIX}relations <mot> :: <texte>`\n"
                f"Exemple : `{PREFIX}relations tomate :: Cuire des tomates avec de l'huile.`\n"
                f"Utilisez `{PREFIX}help2` pour plus d'informations."
            ),
            color=discord.Color.red()
        ))
        return

    mot, texte = [part.strip().lower() for part in input_str.split("::", 1)]
    if not mot or not texte:
        await ctx.send(embed=discord.Embed(
            title="Erreur",
            description=(
                f"Usage : `{PREFIX}relations <mot> :: <texte>`\n"
                f"Exemple : `{PREFIX}relations tomate :: Cuire des tomates avec de l'huile.`\n"
                f"Utilisez `{PREFIX}help2` pour plus d'informations."
            ),
            color=discord.Color.red()
        ))
        return

    # Vérifier si le mot est culinaire
    matched_term = find_fuzzy_match(mot, texte)
    if matched_term != mot:
        mot = matched_term
    if mot not in CULINARY_TERMS:
        pending_add_term[ctx.author.id] = (mot, texte)
        await ctx.send(embed=discord.Embed(
            title="Terme non reconnu",
            description=(
                f"« {mot} » n'est pas un terme culinaire.\n"
                f"Souhaitez-vous l’ajouter ? Répondez avec `{PREFIX}oui` ou `{PREFIX}non`."
            ),
            color=discord.Color.orange()
        ))
        return

    # Vérifier si le mot est dans le texte
    if mot not in texte:
        await ctx.send(embed=discord.Embed(
            title="Erreur",
            description=(
                f"« {mot} » n'apparaît pas dans le texte.\n"
                f"Utilisez `{PREFIX}help2` pour plus d'informations."
            ),
            color=discord.Color.red()
        ))
        return

    # Extraire les relations
    relations = extract_relations(mot, texte)

    # Afficher les résultats
    if not relations:
        await ctx.send(embed=discord.Embed(
            title=f"Aucune relation pour « {mot} »",
            description=(
                "Vérifiez si le texte contient des indices clairs (ex. : 'avec', 'dans').\n"
                f"Utilisez `{PREFIX}help2` pour plus d'informations."
            ),
            color=discord.Color.orange()
        ))
        return

    embed = discord.Embed(
        title=f"Relations sémantiques pour « {mot} »",
        color=discord.Color.blue()
    )
    for rel in sorted(relations, key=lambda x: x.get('weight', 0), reverse=True)[:5]:
        embed.add_field(
            name=f"{rel['type']} → {rel['related']}",
            value=f"Poids : {rel['weight']:.1f} | Source : {rel['source']}\nContexte : « {rel['context']} »",
            inline=False
        )
    await ctx.send(embed=embed)

    # Enregistrer la relation principale
    if relations:
        save_learned_relation(mot, texte, relations[0])

@bot.command()
async def random(ctx):
    """Commande !random : analyse un texte aléatoire avec un mot culinaire aléatoire."""
    # Générer un texte aléatoire
    template = random.choice(TEXT_TEMPLATES)
    ingr1 = random.choice(CULINARY_TERMS)
    ingr2 = random.choice([t for t in CULINARY_TERMS if t != ingr1])
    ingr3 = random.choice([t for t in CULINARY_TERMS if t not in [ingr1, ingr2]])
    plat = random.choice(PLATS)
    ustensile = random.choice(USTENSILES)
    epice = random.choice(EPICES)
    texte = template.format(
        ingr1=ingr1, ingr2=ingr2, ingr3=ingr3, plat=plat, ustensile=ustensile, epice=epice
    ).lower()

    # Choisir un mot aléatoire
    text_terms = find_all_fuzzy_terms(CULINARY_TERMS, texte)
    if not text_terms:
        await ctx.send(embed=discord.Embed(
            title="Erreur",
            description="Aucun terme culinaire trouvé dans le texte généré.",
            color=discord.Color.red()
        ))
        return
    mot = random.choice(text_terms)

    # Extraire les relations
    relations = extract_relations(mot, texte)

    # Afficher les résultats
    embed = discord.Embed(
        title=f"Analyse aléatoire pour « {mot} »",
        description=f"**Texte généré :** {texte}",
        color=discord.Color.green()
    )
    if not relations:
        embed.add_field(
            name="Aucune relation trouvée",
            value="Le texte ne contient pas d'indices clairs pour des relations sémantiques.",
            inline=False
        )
    else:
        for rel in sorted(relations, key=lambda x: x.get('weight', 0), reverse=True)[:5]:
            embed.add_field(
                name=f"{rel['type']} → {rel['related']}",
                value=f"Poids : {rel['weight']:.1f} | Source : {rel['source']}\nContexte : « {rel['context']} »",
                inline=False
            )
    await ctx.send(embed=embed)

    # Enregistrer la relation principale
    if relations:
        save_learned_relation(mot, texte, relations[0])

@bot.event
async def on_command_error(ctx, error):
    """Gestion des erreurs de commande."""
    embed = discord.Embed(title="Erreur", color=discord.Color.red())
    if isinstance(error, commands.CommandNotFound):
        embed.description = f"Commande inconnue. Utilisez `{PREFIX}help2` pour voir les commandes disponibles."
    elif isinstance(error, commands.MissingRequiredArgument):
        embed.description = (
            f"Arguments manquants. Exemple : `{PREFIX}relations tomate :: Cuire des tomates avec de l'huile.`\n"
            f"Utilisez `{PREFIX}help2` pour plus d'informations."
        )
    else:
        embed.description = f"Une erreur s'est produite : {str(error)}\nUtilisez `{PREFIX}help2` pour plus d'informations."
    await ctx.send(embed=embed)

# Lancement du bot
bot.run(TOKEN)