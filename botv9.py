import discord
from discord.ext import commands
import re
import requests
import csv
import spacy

from datetime import datetime
nlp = spacy.load("fr_core_news_md")
from spacy_utils import extract_spacy_relations
from fuzzy_utils import find_fuzzy_match, find_all_fuzzy_terms

import importlib
import generated_culinary_terms
importlib.reload(generated_culinary_terms)
CULINARY_TERMS = generated_culinary_terms.CULINARY_TERMS

# Configuration du bot
TOKEN = "MTM3MDM3NTE5MTU0NDQ2NzQ2Ng.GA87T5.kE498SuVMivtGp_mq3CdCOqZ_gKWH_AjkebuV0"
PREFIX = "!"
API_URL = "https://jdm-api.demo.lirmm.fr/term/{}"


last_relations_by_user = {}
blacklisted_relations = {}


# Charger le modèle SpaCy (modèle français)


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

# Enregistrer les relations apprises
def save_learned_relation(mot, texte, relation):
    with open("learned_relations.csv", "a", newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([timestamp, mot, relation["related"], relation["type"], texte])


# Initialisation du bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix=PREFIX, intents=intents)

# Charger la base de données au démarrage
SEMANTIC_RELATIONS = load_semantic_relations("semantic_relations_rows.csv")
pending_add_term = {}  # clé : nom d’utilisateur, valeur : (mot, texte)
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
        "- Les relations sont basées sur une base de données sémantique, SpaCy, et des motifs avancés.\n"
        "Pour plus d'info, contactez votre encadrant."
    )
    await ctx.send(help_message)
@bot.command()
async def oui(ctx):
    user_id = ctx.author.id
    if user_id not in pending_add_term:
        await ctx.send("Aucune demande d'ajout en attente.")
        return

    mot, _ = pending_add_term.pop(user_id)

    try:
        # Ajouter le mot dans generated_culinary_terms.py
        with open("generated_culinary_terms.py", "r+", encoding='utf-8') as f:
            content = f.read()
            if mot not in content:
                new_line = f'    "{mot}",\n'
                content = content.replace("CULINARY_TERMS = [", "CULINARY_TERMS = [\n" + new_line)
                f.seek(0)
                f.write(content)
                f.truncate()
        # Recharger dynamiquement
        import importlib
        import generated_culinary_terms
        importlib.reload(generated_culinary_terms)

        global CULINARY_TERMS
        CULINARY_TERMS = generated_culinary_terms.CULINARY_TERMS

        await ctx.send(f" Le mot « {mot} » a été ajouté à la liste des termes culinaires.")
    except Exception as e:
        await ctx.send(f" Erreur lors de l'ajout du mot : {str(e)}")

@bot.command()
async def non(ctx):
    if ctx.author.id in pending_add_term:
        pending_add_term.pop(ctx.author.id)
        await ctx.send(" Demande d'ajout annulée.")
    else:
        await ctx.send("Aucune demande d'ajout en attente.")

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
    relations2 = []

    # Vérifier si le mot est un terme culinaire
    if mot not in CULINARY_TERMS:
        pending_add_term[ctx.author.id] = (mot, texte)
        await ctx.send(
            f"⚠ Le mot « {mot} » n'est pas reconnu comme un terme culinaire.\n"
            f"Souhaitez-vous l’ajouter à la liste ? Répondez avec `!oui` ou `!non`."
        )
        return

    from fuzzywuzzy import fuzz
    matched_term = find_fuzzy_match(mot, texte)
    if not matched_term:
        await ctx.send(
            f"⚠ Le mot « {mot} » n'apparaît pas clairement dans le texte fourni.\n"
            f"Essayez de reformuler ou d’utiliser des formes proches (ex. : pluriel).\n"
            f"Utilisez `{PREFIX}help2` pour plus d'informations."
        )
        return
    else:
        mot = matched_term  # On continue avec le mot détecté

    text_terms = find_all_fuzzy_terms(CULINARY_TERMS, texte, exclude=mot)



    # 1. Analyse avec SpaCy pour des relations grammaticales
    doc = nlp(texte)
    for token in doc:
        if token.text.lower() == mot:
            for child in token.children:
                related_term = child.text.lower()
                if related_term in text_terms:
                    dep = child.dep_
                    if dep in ['dobj', 'pobj', 'nmod']:  # Objet direct, prépositionnel, ou modificateur nominal
                        relations2.append({
                            "type": "r.object" if dep == 'dobj' else "r.location" if dep == 'pobj' else "r.associated",
                            "related": related_term,
                            "weight": 40 if dep == 'dobj' else 35,
                            "context": texte,
                            "source": "spacy"
                        })
                    elif dep == 'nsubj' and child.text in CULINARY_TERMS:  # Sujet
                        relations2.append({
                            "type": "r.agent",
                            "related": related_term,
                            "weight": 40,
                            "context": texte,
                            "source": "spacy"
                        })
            for token_in_phrase in doc:
                if token_in_phrase.dep_ == 'prep' and token_in_phrase.head == token:
                    for grandchild in token_in_phrase.children:
                        if grandchild.text.lower() in text_terms:
                            relations2.append({
                                "type": "r.instr" if "avec" in token_in_phrase.text else "r.location",
                                "related": grandchild.text.lower(),
                                "weight": 40,
                                "context": texte,
                                "source": "spacy"
                            })

    # 2. Recherche directe dans la base de données
    if mot in SEMANTIC_RELATIONS:
        for term in text_terms:
            if term in SEMANTIC_RELATIONS[mot]:
                rel = SEMANTIC_RELATIONS[mot][term]
                combined_weight = rel['weight'] * 0.6 + rel['sim'] * 0.4  # Combinaison de poids et similarité
                if combined_weight >= 20:
                    relations2.append({
                        "type": rel['relation'],
                        "related": term,
                        "weight": combined_weight,
                        "context": texte,
                        "source": "database"
                    })

    # 3. Recherche par similarité
    if not relations2:
        best_sim = 0
        best_match = None
        for db_node1 in SEMANTIC_RELATIONS:
            sim_score = fuzz.ratio(mot, db_node1)
            if sim_score > best_sim and sim_score >= 80:
                best_sim = sim_score
                best_match = db_node1
        if best_match:
            for term in text_terms:
                if term in SEMANTIC_RELATIONS[best_match]:
                    rel = SEMANTIC_RELATIONS[best_match][term]
                    combined_weight = rel['weight'] * 0.6 + rel['sim'] * 0.4 + (best_sim / 100) * 10
                    if combined_weight >= 20:
                        relations2.append({
                            "type": rel['relation'],
                            "related": term,
                            "weight": combined_weight,
                            "context": texte,
                            "source": f"database (similar term: {best_match}, sim: {best_sim}%)"
                        })

                        # 3.2 Analyse grammaticale avec SpaCy
                        spacy_results = extract_spacy_relations(texte, mot)
                        relations2.extend(spacy_results)

                        doc = nlp(texte)

                        for token in doc:
                            if token.head.lemma_ == mot and token.dep_ in ["obj", "iobj", "obl", "nmod"]:
                                relations2.append({
                                    "type": f"dep_{token.dep_}",
                                    "related": token.text,
                                    "weight": 22,
                                    "context": f"{token.head.text} → {token.text}",
                                    "source": "spacy"
                                })
                            elif token.text == mot and token.dep_ in ["obj", "iobj", "obl", "nmod"]:
                                relations2.append({
                                    "type": f"dep_{token.dep_}",
                                    "related": token.head.text,
                                    "weight": 22,
                                    "context": f"{token.text} ← {token.head.text}",
                                    "source": "spacy"
                                })

    # 4. Analyse contextuelle avec motifs regex améliorés
    phrases = re.split(r'[.!?]+', texte)
    phrases = [p.strip() for p in phrases if p.strip() and mot in p]
    for phrase in phrases:
        for term in text_terms:
            # r.instr: "avec un/une", "à l'aide de"
            if re.search(rf"avec\s+(un|une|à\s+l'aide\s+de)\s+{re.escape(term)}", phrase):
                base_weight = 30 if "à l'aide de" in phrase else 25
                # Boost si la relation est dans la base
                boost = 10 if term in SEMANTIC_RELATIONS.get(mot, {}) and SEMANTIC_RELATIONS[mot][term]['dep'] in ['nmod', 'prep'] else 0
                relations2.append({"type": "r.instr", "related": term, "weight": base_weight + boost, "context": phrase, "source": "regex"})

            # r.manner: "de manière", "en utilisant"
            if re.search(rf"(de\s+manière|en\s+utilisant)\s+{re.escape(term)}", phrase):
                base_weight = 25
                boost = 10 if term in SEMANTIC_RELATIONS.get(mot, {}) and SEMANTIC_RELATIONS[mot][term]['dep'] == 'advmod' else 0
                relations2.append({"type": "r.manner", "related": term, "weight": base_weight + boost, "context": phrase, "source": "regex"})

            # r.agent: "par", ou sujet/verbe
            if re.search(rf"par\s+{re.escape(term)}", phrase) or re.search(rf"{re.escape(term)}\s+(épluche|cuit|prépare)", phrase):
                relations2.append({"type": "r.agent", "related": term, "weight": 30, "context": phrase, "source": "regex"})

            # r.object: "<mot> <verbe> <objet>"
            if re.search(rf"{re.escape(mot)}\s+(épluche|cuit|coupe|prépare)\s+{re.escape(term)}", phrase):
                relations2.append({"type": "r.object", "related": term, "weight": 30, "context": phrase, "source": "regex"})

            # r.location: "dans", "sur"
            if re.search(rf"(dans|sur)\s+{re.escape(term)}", phrase):
                relations2.append({"type": "r.location", "related": term, "weight": 25, "context": phrase, "source": "regex"})

    # 5. Si aucune relation n'est trouvée, tenter l'API JDM
    if not relations2:
        try:
            response = requests.get(API_URL.format(mot.replace(" ", "_")), timeout=5)
            response.raise_for_status()
            data = response.json()
            api_relations = data.get("relations", [])
            for rel in api_relations[:3]:
                rel_type = rel.get("type", "type inconnu")
                node_name = rel.get("node", {}).get("name", "inconnu").replace("_", " ")
                if node_name != mot and rel_type in ["r.instr", "r.manner", "r.agent", "r.object", "r.location"]:
                    relations2.append({
                        "type": rel_type,
                        "related": node_name,
                        "weight": 20,
                        "context": "Extrait de l'API JeuxDeMots (texte non spécifique)",
                        "source": "api"
                    })
        except requests.RequestException:
            pass

    # 6. Enregistrer les relations apprises
    if relations2:
        for rel in relations2[:1]:  # Enregistrer la relation principale
            save_learned_relation(mot, texte, rel)

    # 7. Afficher les résultats
    if not relations2:
        feedback = (
            f"ℹ Aucune relation sémantique trouvée pour « {mot} ».\n"
            "- Vérifiez si le mot ou les termes associés sont dans la base de données.\n"
            "- Essayez un texte avec des indices clairs (ex. : 'avec', 'dans', 'cuit par').\n"
            f"Utilisez `{PREFIX}help2` pour plus d'informations."
        )
        await ctx.send(feedback)
        return

    # Filtrer les relations blacklistées par l'utilisateur
    filtered_relations = []
    for rel in relations2:
        rel_id = (mot, rel["type"], rel["related"])
        if rel_id not in blacklisted_relations.get(ctx.author.id, []):
            filtered_relations.append(rel)

    if not filtered_relations:
        await ctx.send("⚠️ Toutes les relations détectées sont actuellement ignorées (blacklistées).")
        return

    resultats = [f"**Relations sémantiques pour « {mot} » :**"]
    for rel in sorted(filtered_relations, key=lambda x: x.get('weight', 0), reverse=True)[:5]:
        ligne = f" {mot} {rel['type']} {rel['related']} (poids: {rel.get('weight', 0):0.1f}, source: {rel['source']})"
        ligne += f"\n    Contexte : « {rel['context']} »"
        resultats.append(ligne)

    await ctx.send("\n".join(resultats))

    # Stocker les relations pour permettre correction
    last_relations_by_user[ctx.author.id] = {
        "mot": mot,
        "texte": texte,
        "relations": relations2
    }

    await ctx.send(
        " Si certains termes ne sont pas corrects, vous pouvez répondre avec `!corriger` pour les supprimer."
    )

@bot.command()
async def corriger(ctx):
    user_id = ctx.author.id
    if user_id not in last_relations_by_user:
        await ctx.send(" Aucune relation à corriger pour vous.")
        return

    data = last_relations_by_user[user_id]
    relations = data["relations"]

    if not relations:
        await ctx.send("Aucune relation enregistrée.")
        return

    liste = "\n".join(
        [f"{idx+1}. {data['mot']} {rel['type']} {rel['related']} (source: {rel['source']})"
         for idx, rel in enumerate(relations)]
    )

    await ctx.send(
        f" Voici les relations extraites :\n{liste}\n\n"
        "✏ Pour supprimer une ou plusieurs relations, tapez par exemple : `!supprimer 2 4`"
    )

@bot.command()
async def supprimer(ctx, *indices):
    user_id = ctx.author.id
    if user_id not in last_relations_by_user:
        await ctx.send(" Aucune relation enregistrée pour vous.")
        return

    try:
        indices = [int(i) - 1 for i in indices]  # Conversion vers index Python
    except ValueError:
        await ctx.send(" Les indices doivent être des nombres (ex. : `!supprimer 1 3`).")
        return

    relations = last_relations_by_user[user_id]["relations"]

    filtered = [rel for idx, rel in enumerate(relations) if idx not in indices]
    last_relations_by_user[user_id]["relations"] = filtered

    if not filtered:
        await ctx.send("🗑️ Toutes les relations ont été supprimées.")
    else:
        updated = "\n".join(
            [f"{idx+1}. {last_relations_by_user[user_id]['mot']} {rel['type']} {rel['related']} (source: {rel['source']})"
             for idx, rel in enumerate(filtered)]
        )
        await ctx.send(
            f" Relations mises à jour :\n{updated}"
        )


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