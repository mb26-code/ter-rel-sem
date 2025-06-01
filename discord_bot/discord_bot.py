

import discord
from discord.ext import commands
import requests
import io
import contextlib
from dotenv import load_dotenv
from discord_bot.algo_bot import process_text, load_word2vec_model, load_relation_types
import os

load_dotenv()
intents = discord.Intents.default()
bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    print(f"Connecté en tant que {bot.user}")

@bot.command(name='analyse_web')
async def analyse_mot_depuis_url(ctx, *, message: str):
    if "::" not in message:
        await ctx.send("Utilisation : `!analyse_web <mot> :: <url>` (utiliser `::` comme séparateur)")
        return

    mot, url = map(str.strip, message.split("::", 1))
    await ctx.send(f"Analyse du mot **{mot}** à partir de l'URL : {url}")

    try:
        response = requests.get(url)
        response.raise_for_status()
        texte = response.text

        if mot not in texte:
            await ctx.send("Mot non trouvé dans le texte.")
            return

        model = load_word2vec_model()
        relation_types = load_relation_types()
        results, all_info = process_text(texte, model, relation_types)

        filtered = [r for r in all_info if r['head'] == mot or r['lemma'] == mot]

        if not filtered:
            await ctx.send("Aucune relation trouvée contenant ce mot.")
        else:
            msg = ""
            for i, rel in enumerate(filtered[:10]):
                msg += f"{i+1}. {rel['head']} → {rel['relation_name']} → {rel['lemma']} (poids: {rel['w']})\n"
            await ctx.send(msg)

    except Exception as e:
        await ctx.send(f"Erreur lors de l’analyse : {e}")


@bot.command(name='analyse_texte')
async def analyse_mot_dans_texte(ctx, *, message: str):
    if "::" not in message:
        await ctx.send("Utilisation : `!analyse_texte <mot> :: <texte>` (utiliser `::` comme séparateur)")
        return

    mot, texte = map(str.strip, message.split("::", 1))
    await ctx.send(f"Analyse du mot **{mot}** dans le texte fourni...")

    try:
        model = load_word2vec_model()
        relation_types = load_relation_types()
        results, all_info = process_text(texte, model, relation_types)

        filtered = [r for r in all_info if r['head'] == mot or r['lemma'] == mot]

        if not filtered:
            await ctx.send("Aucune relation trouvée contenant ce mot.")
        else:
            msg = ""
            for i, rel in enumerate(filtered[:10]):
                msg += f"{i+1}. {rel['head']} → {rel['relation_name']} → {rel['lemma']} (poids: {rel['w']})\n"
            await ctx.send(msg)

    except Exception as e:
        await ctx.send(f"Erreur lors de l’analyse : {e}")


@bot.command(name='analyse_texte_detail')
async def analyse_texte_avec_debug(ctx, *, message: str):
    if "::" not in message:
        await ctx.send("Utilisation : `!analyse_texte_detail <mot> :: <texte>`")
        return

    mot, texte = map(str.strip, message.split("::", 1))
    await ctx.send(f"Analyse du mot **{mot}** avec affichage des détails...")

    f = io.StringIO()
    try:
        with contextlib.redirect_stdout(f):
            model = load_word2vec_model()
            relation_types = load_relation_types()
            results, all_info = process_text(texte, model, relation_types)

        debug_output = f.getvalue()

        if len(debug_output) > 1900:
            debug_output = debug_output[:1900] + "\n...[troncature]..."

        await ctx.send(f"```{debug_output}```")

    except Exception as e:
        await ctx.send(f"Erreur lors de l’analyse : {e}")


@bot.command(name='aide')
async def afficher_aide_embed(ctx):
    embed = discord.Embed(
        title="Aide du Bot de Relations Sémantiques",
        description="Commandes disponibles pour analyser un mot ou un texte dans le domaine de la gastronomie.",
        color=discord.Color.blue()
    )

    embed.add_field(
        name="1. !analyse_web <mot> :: <url>",
        value="Analyse un mot à partir du contenu HTML d’une page web.\n"
              "**Exemple :** `!analyse_web chocolat :: https://fr.wikipedia.org/wiki/Chocolat`",
        inline=False
    )

    embed.add_field(
        name="2. !analyse_texte <mot> :: <texte>",
        value="Analyse uniquement les relations contenant le mot donné dans un court texte.\n"
              "**Exemple :** `!analyse_texte sel :: Ajoutez du sel au plat.`",
        inline=False
    )

    embed.add_field(
        name="3. !analyse_texte_detail <mot> :: <texte>",
        value="Comme `!analyse_texte`, mais avec les détails du traitement (similarité, source, poids...).\n"
              "**Exemple :** `!analyse_texte_detail sel :: Ajoutez du sel au plat.`",
        inline=False
    )

    embed.add_field(
        name="4. !executer <mot> :: <texte>",
        value="Affiche toutes les relations détectées autour du mot, avec sourcing et poids.\n"
              "**Exemple :** `!executer sucre :: Ajoutez le sucre au lait chaud, puis mélangez avec la vanille.`",
        inline=False
    )

    embed.add_field(
        name="5. !generer_csv <nom_fichier> :: <texte>",
        value="Génère un fichier CSV avec toutes les relations extraites depuis un texte donné.\n"
              "**Exemple :** `!generer_csv recette_sucre :: Ajoutez le sucre au lait chaud.`",
        inline=False
    )

    embed.set_footer(text="Projet académique – Extraction de relations sémantiques à partir de textes culinaires.")
    await ctx.send(embed=embed)




@bot.command(name='executer')
async def executer_analyse_complete(ctx, *, message: str):
    if "::" not in message:
        await ctx.send("Utilisation : `!executer <mot> :: <texte>`")
        return

    mot, texte = map(str.strip, message.split("::", 1))
    await ctx.send(f"Exécution complète de l’analyse pour le mot **{mot}**...")

    f = io.StringIO()
    try:
        with contextlib.redirect_stdout(f):
            model = load_word2vec_model()
            relation_types = load_relation_types()
            process_text(texte, model, relation_types)

        output = f.getvalue()

        # Filtrage pour ne garder que les lignes contenant le mot (optionnel)
        filtered_output = "\n".join(
            line for line in output.splitlines()
            if mot.lower() in line.lower()
            or "STATISTIQUES DE SOURCING" in line
            or "RELATIONS SÉMANTIQUES" in line
            or line.strip().startswith("=")  # Pour le cadre
        )

        to_send = filtered_output if filtered_output.strip() else output

        if len(to_send) <= 1900:
            await ctx.send(f"```{to_send}```")
        else:
            chunks = [to_send[i:i+1900] for i in range(0, len(to_send), 1900)]
            for chunk in chunks:
                await ctx.send(f"```{chunk}```")

    except Exception as e:
        await ctx.send(f"Erreur pendant l'exécution : {e}")

import subprocess
import csv

@bot.command(name='generer_csv')
async def executer_script_et_envoyer_csv(ctx, *, message: str):
    if "::" not in message:
        await ctx.send("Utilisation : `!generer_csv <nom_de_fichier> :: <texte>`")
        return

    nom_fichier, texte = map(str.strip, message.split("::", 1))
    output_path = f"output/{nom_fichier}.csv"

    # Appel du script avec subprocess
    try:
        await ctx.send("Lancement de l'analyse avec `algo_bot.py`...")

        command = [
            "python3",
            "algo_bot.py",
            "--text", texte,
            "--output", output_path
        ]

        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Affiche le log du script dans Discord
        log = result.stdout + "\n" + result.stderr
        if len(log) > 1900:
            log = log[:1900] + "\n...[troncature]"
        await ctx.send(f"```{log}```")

        # Vérifie et envoie le fichier CSV généré
        if os.path.exists(output_path):
            await ctx.send("Voici le fichier CSV généré :", file=discord.File(output_path))
        else:
            await ctx.send("Erreur : aucun fichier CSV n’a été généré.")

    except Exception as e:
        await ctx.send(f"Erreur lors de l'exécution : {e}")

load_dotenv()
bot.run(os.getenv("DISCORD_TOKEN"))
