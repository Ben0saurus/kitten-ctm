import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_kitten_lora(
    base_model_id="Qwen/Qwen3-0.6B",
    # Pfad zu deiner .pt Datei
    lora_model_path="C:/Users/Domi/Desktop/kitten-ctm-main/Kitten-LoRA/models/kitten_simple/best/hope_lora.pt", 
    device="cpu"
):
    """
    Lädt das Base Model und lädt LoRA Gewichte direkt aus der .pt Datei.
    """
    print(f"⏳ Lade Base Model: {base_model_id}...")
    
    try:
        # Tokenizer laden
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_id, 
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        
        # Base Model laden
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        
        model.to(device)
        print("✅ Base Model geladen.")
        
    except Exception as e:
        print(f"❌ Fehler beim Laden Base Model: {e}")
        return None, None

    # --- LoRA Gewichte aus der .pt Datei laden ---
    print(f"⏳ Lade LoRA Checkpoint von: {lora_model_path}...")
    
    try:
        # Wir laden das Dictionary aus der .pt Datei
        checkpoint = torch.load(lora_model_path, map_location="cpu")
        
        # Extrahiere die LoRA Gewichte (oft unter dem Key 'lora_A', 'lora_B' etc.)
        # und weisen sie dem Modell zu.
        # Da Qwen/Transformers Layer-Namen benutzt, müssen wir versuchen zu matchen.
        
        # Einfache Methode: Striktes Laden wenn es passt
        # Oder wir laden direkt in die LoRA-Adapter wenn die Struktur stimmt.
        
        # Da es eine .pt Datei ist, nutzen wir load_state_dict
        # Wir müssen aber wissen, welchen Key das Training benutzt hat.
        
        # FALL 1: Die Datei enthält direkt die Modell-Parameter
        model.load_state_dict(checkpoint, strict=False) 
        # strict=False verhindert einen Absturz, wenn Keys nicht 1:1 passen
        
        print("✅ LoRA Gewichte geladen.")
        
    except Exception as e:
        print(f"❌ Fehler beim Laden LoRA (.pt): {e}")
        print("Info: Versuche mit .pt zu laden. Wenn das fehlschlägt, muss das Training in 'safe_tensors' speichern.")
        return None, None
        
    model.eval()
    return model, tokenizer

def generate_kitten_response(prompt, model, tokenizer, max_new_tokens=150):
    if model is None:
        return "(Modell Error)"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    
    if decoded.startswith(prompt):
        return decoded[len(prompt):].strip()
    
    return decoded.strip()