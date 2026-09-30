#i = 1003 is sadness confirmed

from pathlib import Path
from models import FusionMLP
import soundfile as sf
from transformers import AutoFeatureExtractor, WavLMModel, AutoTokenizer, AutoModel
import torch, numpy as np, pandas as pd, torch.nn as nn
import mlx_whisper


ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
DATA = ROOT / "data"
EMOTIONS = ["neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"]
manifest = pd.read_csv(DATA / "manifest.csv")
i = 1003
idx = pd.read_csv(CACHE / "index.csv")


X = np.load(CACHE / "text.npy").astype(np.float32)
X_audio = np.load(CACHE / "audio_wavlm_13L.npy").astype(np.float32).reshape(-1, 13, 1536)



# Text Pipeline

MODEL = 'roberta-base'
BATCH = 64
dev = "mps" if torch.backends.mps.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(MODEL)
enc = AutoModel.from_pretrained(MODEL).to(dev)
enc.eval()

def encode_text(text, tok, enc, dev):
    b = tok(text, truncation=True, max_length=128, return_tensors="pt").to(dev)
    with torch.no_grad():
        h = enc(**b).last_hidden_state                      # [1, seq, 768]
    m = b["attention_mask"].unsqueeze(-1).float()
    return ((h * m).sum(1) / m.sum(1)).cpu()                # [1, 768]

gold_text = manifest.Utterance[i]
xt = encode_text(gold_text, tok, enc, dev)

#Audio Pipeline


wav, sr = sf.read(ROOT / manifest.audio_path[i], dtype="float32")   # row i's clip
dev = "mps" if torch.backends.mps.is_available() else "cpu" 
fe  = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
wlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").eval().to(dev)

inp = fe(wav, sampling_rate=16000, return_tensors="pt").to(dev)
with torch.no_grad():
    out = wlm(**inp, output_hidden_states=True)
feats = [f for h in out.hidden_states for f in (h.mean(1), h.std(1))]
xa = torch.cat(feats, 1).reshape(1, 13, 1536).cpu()

def encode_audio(wav, fe, wlm, dev):
    inp = fe(wav, sampling_rate=16000, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = wlm(**inp, output_hidden_states=True)
    feats = [f for h in out.hidden_states for f in (h.mean(1), h.std(1))]
    return torch.cat(feats, 1).reshape(1, 13, 1536).cpu()


#Whisper text transcription
WHISPER = "mlx-community/whisper-base.en-mlx"
def transcribe(wav):
    # wav: float32 numpy array, 16kHz, mono — same format everything else uses
    result = mlx_whisper.transcribe(wav, path_or_hf_repo=WHISPER)
    return result["text"].strip()

w_text = transcribe(wav)
xt_w = encode_text(w_text, tok, enc, dev)




m = FusionMLP(768, 1536, 13, 7)
m.load_state_dict(torch.load(CACHE / "fusion_model.pt"))
m.eval()




# for i in range(1000, 1010):
#     wav, _ = sf.read(manifest.audio_path[i], dtype="float32")
#     xt = encode_text(manifest.Utterance[i], tok, enc, dev)
#     xt_w  = encode_text(transcribe(wav), tok, enc, dev)
#     xa      = encode_audio(wav, fe, wlm, dev)
#     with torch.no_grad():
#         pg = m(xt, xa).argmax(1).item()
#         pw = m(xt_w, xa).argmax(1).item()
#     print(f"{i}: gold {EMOTIONS[pg]} | whisper {EMOTIONS[pw]} | true {idx.iloc[i].Emotion}")

# from sklearn.metrics import f1_score
# dev_indices = np.where(manifest["split"] == "dev")[0]
# EMO2ID = {e: i for i, e in enumerate(EMOTIONS)}
# manifest["emo_id"] = manifest["Emotion"].map(EMO2ID)


# gold_pred, asr_pred, y = [], [], []
# for i in dev_indices:
#     wav, _ = sf.read(manifest.audio_path[i], dtype="float32")
#     xa = encode_audio(wav, fe, wlm, dev)
#     xt_g = encode_text(manifest.Utterance[i], tok, enc, dev)
#     xt_a = encode_text(transcribe(wav) or "[unclear]", tok, enc, dev)
#     with torch.no_grad():
#         gold_pred.append(m(xt_g, xa).argmax(1).item())
#         asr_pred.append(m(xt_a, xa).argmax(1).item())
#     y.append(manifest.iloc[i].emo_id)

# print("gold  wF1", f1_score(y, gold_pred, average="weighted"))
# print("asr   wF1", f1_score(y, asr_pred, average="weighted"))


#llm response

from mlx_lm import load, stream_generate

llm, llm_tok = load("mlx-community/Qwen2.5-3B-Instruct-4bit")   # ~3.1B; project total ~3.4B (cap 6B)

def build_prompt(state, llm_tok):
    # tone goes in the system prompt; the user turn is only what they said,
    # so the model replies to the person instead of describing them
    system = (
        f"You are a warm friend in a spoken conversation. The person you're talking to "
        f"sounds {state['emotion']} ({describe_prosody(state['prosody'])}). "
        "Reply to them directly in one short sentence, speaking as 'you'. "
        "Do not describe them, do not name their emotion."
    )
    user = state["transcript"]
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    return llm_tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

def compute_prosody(wav, sr):
    return {"rms": float(np.sqrt(np.mean(wav ** 2))), "duration_s": len(wav) / sr}

def describe_prosody(p):
    loud = "raised volume" if p["rms"] > 0.05 else "low volume"
    return f"{loud}, {p['duration_s']:.1f}s"

def generate(prompt, llm, llm_tok, max_tokens=40):
    response = ""
    for chunk in stream_generate(llm, llm_tok, prompt, max_tokens=max_tokens):
        print(chunk.text, end="", flush=True)   # live streaming to terminal
        response += chunk.text
    print()
    return response


with torch.no_grad():
    probs = torch.softmax(m(xt_w, xa), 1)[0]      # Whisper text + audio, no gold text
pred = probs.argmax().item()


state = {
    "transcript": w_text,                     # from Whisper
    "emotion": EMOTIONS[pred],                # from your model, not hardcoded
    "confidence": probs[pred].item(),
    "prosody": compute_prosody(wav, sr),      # raw numbers; build_prompt describes them
}

print(f"utterance {i} | true {manifest.Emotion[i]}")
print(f"whisper: {w_text}")
print(f"pred: {state['emotion']} ({state['confidence']:.2f}) | {describe_prosody(state['prosody'])}")
print("llm: ", end="")
response = generate(build_prompt(state, llm_tok), llm, llm_tok)

