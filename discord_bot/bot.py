import discord
import requests

TOKEN = "MTM3MDM3NTE5MTU0NDQ2NzQ2Ng.GA87T5.kE498SuVMivtGp_mq3CdCOqZ_gKWH_AjkebuV0"  # <-- Remplace ceci par ton vrai token

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f"✅ Bot connecté en tant que {client.user}")

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith("!jdm "):
        mot = message.content[len("!jdm "):].strip()
        if not mot:
            await message.channel.send("Tu dois écrire un mot après `!jdm`.")
            return

        url = f"https://jdm-api.demo.lirmm.fr/term/{mot}"
        try:
            r = requests.get(url)
            data = r.json()
        except:
            await message.channel.send("Erreur lors de l'appel à l'API.")
            return

        relations = data.get("relations", [])
        if not relations:
            await message.channel.send(f"Aucune relation trouvée pour « {mot} ».")
            return

        resultats = []
        for rel in relations[:5]:
            ligne = f"🔗 {rel['name']} → **{rel['node']['name']}** ({rel['type']})"
            resultats.append(ligne)

        await message.channel.send("\n".join(resultats))

client.run(TOKEN)
