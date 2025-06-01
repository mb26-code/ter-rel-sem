import discord
from discord.ext import commands
import requests
from bs4 import BeautifulSoup
import io
import contextlib
from dotenv import load_dotenv
from algo_bot import process_text, load_word2vec_model, load_relation_types
import os
import subprocess
import csv

load_dotenv()
intents = discord.Intents.default()
bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    print(f"Connecté en tant que {bot.user}")

@bot.command(name='analyse_web')
async def analyse_url(ctx, *, url: str):
    
    
    output_path = f"relations.csv"

    await ctx.send("Recherche du lien et traitement du texte ... \n") 

    texte = get_clean_text_from_url(url)

    await ctx.send("Texte traité ... \n") 

    try:

        await ctx.send("Recherche de relations ... \n") 

        command = [
            "python3",
            "discord_bot/algo_bot.py",
            "--text", texte,
            "--output", output_path
        ]

        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


        log = result.stdout + "\n" + result.stderr
        max_length = 1900

        filtered_lines = [line for line in log.splitlines() if not line.startswith("Interrogation")]
        filtered_log = "\n".join(filtered_lines)

        for i in range(0, len(filtered_log), max_length):
            chunk = filtered_log[i:i+max_length]
            await ctx.send(f"```{chunk}```")


        if os.path.exists(output_path):
            await ctx.send("Voici le fichier CSV généré :", file=discord.File(output_path))
        else:
            await ctx.send("Erreur : aucun fichier CSV n’a été généré.")


        await ctx.send("FIN") 


    except Exception as e:
        await ctx.send(f"Erreur lors de l'exécution : {e}")


@bot.command(name='analyse_texte')
async def analyse_url(ctx, *, texte: str):

    output_path = f"relations.csv"
    try:

        await ctx.send("Recherche de relations ... \n") 

        command = [
            "python3",
            "discord_bot/algo_bot.py",
            "--text", texte,
            "--output", output_path
        ]

        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


        log = result.stdout + "\n" + result.stderr
        max_length = 1900

        filtered_lines = [line for line in log.splitlines() if not line.startswith("Interrogation")]
        filtered_log = "\n".join(filtered_lines)

        for i in range(0, len(filtered_log), max_length):
            chunk = filtered_log[i:i+max_length]
            await ctx.send(f"```{chunk}```")


        if os.path.exists(output_path):
            await ctx.send("Voici le fichier CSV généré :", file=discord.File(output_path))
        else:
            await ctx.send("Erreur : aucun fichier CSV n’a été généré.")


        await ctx.send("FIN") 


    except Exception as e:
        await ctx.send(f"Erreur lors de l'exécution : {e}")



print("--------------- aide -------------------")



@bot.command(name='aide')
async def afficher_aide_embed(ctx):
    embed = discord.Embed(
        title="Aide du Bot de Relations Sémantiques",
        description="Commandes disponibles pour analyser un mot ou un texte dans le domaine de la gastronomie.",
        color=discord.Color.blue()
    )

    embed.add_field(
        name="1. !analyse_web <url>",
        value="Analyse un mot à partir du contenu HTML d’une page web.\n"
              "**Exemple :** `!analyse_web https://fr.wikipedia.org/wiki/Chocolat`",
        inline=False
    )

    embed.add_field(
        name="2. !analyse_texte <texte>",
        value="Analyse uniquement les relations contenant le mot donné dans un court texte.\n"
              "**Exemple :** `!analyse_texte \"Ajoutez du sel au plat.\"`",
        inline=False
    )

    embed.set_footer(text="Projet académique – Extraction de relations sémantiques à partir de textes culinaires.")
    await ctx.send(embed=embed)




def get_clean_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  
        soup = BeautifulSoup(response.text, 'html.parser')

        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()
        text = soup.get_text(separator='\n')

        lines = [line.strip() for line in text.splitlines()]
        clean_text = '\n'.join(line for line in lines if line)

        return clean_text
    except Exception as e:
        return f"Erreur : {e}"


load_dotenv()
bot.run(os.getenv("DISCORD_TOKEN"))
