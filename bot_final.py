import discord
from discord.ext import commands
from dotenv import load_dotenv
import os
import asyncio

# Importiere die beiden Gehirne
from inference import load_model as load_ctm, predict as ctm_predict
from inference_transformer import load_kitten_lora, generate_kitten_response

# --- CONFIGURATION ---
load_dotenv() 
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

DECISION_THRESHOLD = 0.0

# Discord Setup
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# Memory
context_history = []

@bot.event
async def on_ready():
    print(f'✅ Bot online als {bot.user}')
    
    # 1. Lade CTM (Der Wächter)
    print("🧠 Lade CTM Modell...")
    global ctm_model, ctm_tokenizer
    ctm_model, ctm_tokenizer = load_ctm("checkpoints/best_model.pt", device="cpu")
    
    # 2. Lade Transformer (Der Sprecher)
    print("🗣️ Lade Transformer Modell...")
    global transformer_model, transformer_tokenizer
    # Pfad zum Ordner anpassen (wenn er anders heißt)
    # on_ready Funktion:
    lora_path = "C:/Users/Domi/Desktop/kitten-ctm-main/Kitten-LoRA/models/kitten_simple/best/hope_lora.pt" # <--- Dateiname, nicht Ordner!
    transformer_model, transformer_tokenizer = load_kitten_lora(lora_model_path=lora_path, device="cpu")
    # Optional: model_name="meta-llama/Llama-3-8b" etc.

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    # Kontext updaten
    context_history.append({"username": message.author.name, "content": message.content})
    if len(context_history) > 50:
        context_history.pop(0)

    # --- TEIL 1: CTM CHECK ---
    try:
        prob, ticks = ctm_predict(context_history, ctm_model, ctm_tokenizer, device="cpu")
        print(f"CTM Score: {prob:.4f} | Ticks: {ticks}")
    except Exception as e:
        print(f"CTM Error: {e}")
        return

    # WICHTIG: Entscheiden wir?
    if prob < DECISION_THRESHOLD:
        print(">> NEIN. Keine Antwort.")
        return

    print(">> JA. Rufe Transformer an...")
    
    # --- TEIL 2: TRANSFORMER AUFRUF ---
    try:
        # Kontext formatieren für den Transformer
        # Hier nutzen wir den gleichen Format-String wie für den CTM
        # ACHTUNG: Transformers verstehen oft kein "User: Msg" Format ohne Training!
        # Für echte Projekte musst du den String ggf. anpassen.
        history_string = "\n".join([f"{m['username']}: {m['content']}" for m in context_history])
        
        # Generiere Antwort (kann dauern)
        response_text = generate_kitten_response(history_string, transformer_model, transformer_tokenizer)
        
        await message.channel.send(response_text)
        print(f">> Antwort gesendet: {response_text[:50]}...")
        
    except Exception as e:
        print(f"Transformer Error: {e}")
        # Fallback falls Transformer crasht
        await message.channel.send(f"Error: {e}")

# --- START ---
bot.run(DISCORD_TOKEN)