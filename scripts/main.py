"""End-to-end: audio -> Whisper + WavLM -> fusion emotion -> LLM reply.

  python3 scripts/main.py --replay recording.wav
  python3 scripts/main.py --record [out.wav]
"""
import argparse
from math import gcd
from pathlib import Path

import mlx_whisper
import numpy as np
import soundfile as sf
import torch
from mlx_lm import load, stream_generate
from scipy.signal import resample_poly
from transformers import AutoFeatureExtractor, AutoModel, AutoTokenizer, WavLMModel

from models import FusionMLP
from transcribe import record_until_enter

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
EMOTIONS = ["neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"]
SR = 16000

TEXT_ENC = "roberta-base"
AUDIO_ENC = "microsoft/wavlm-base-plus"
WHISPER = "mlx-community/whisper-base.en-mlx"
LLM = "mlx-community/Qwen2.5-3B-Instruct-4bit"   # ~3.1B; project total ~3.4B (cap 6B)

DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def load_models():
    M = argparse.Namespace()
    M.tok = AutoTokenizer.from_pretrained(TEXT_ENC)
    M.enc = AutoModel.from_pretrained(TEXT_ENC).eval().to(DEV)
    M.fe = AutoFeatureExtractor.from_pretrained(AUDIO_ENC)
    M.wlm = WavLMModel.from_pretrained(AUDIO_ENC).eval().to(DEV)
    M.fusion = FusionMLP(768, 1536, 13, len(EMOTIONS))
    M.fusion.load_state_dict(torch.load(CACHE / "fusion_model.pt"))
    M.fusion.eval()
    M.llm, M.llm_tok = load(LLM)
    return M


def load_wav(path):
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:                          # stereo -> mono
        wav = wav.mean(axis=1)
    if sr != SR:                              # e.g. 44.1/48 kHz -> 16 kHz
        g = gcd(SR, sr)
        wav = resample_poly(wav, SR // g, sr // g).astype(np.float32)
    return wav


def transcribe(wav):
    return mlx_whisper.transcribe(wav, path_or_hf_repo=WHISPER)["text"].strip()


def encode_text(text, M):
    b = M.tok(text, truncation=True, max_length=128, return_tensors="pt").to(DEV)
    with torch.no_grad():
        h = M.enc(**b).last_hidden_state                      # [1, seq, 768]
    m = b["attention_mask"].unsqueeze(-1).float()
    return ((h * m).sum(1) / m.sum(1)).cpu()                  # [1, 768]


def encode_audio(wav, M):
    inp = M.fe(wav, sampling_rate=SR, return_tensors="pt").to(DEV)
    with torch.no_grad():
        out = M.wlm(**inp, output_hidden_states=True)
    feats = [f for h in out.hidden_states for f in (h.mean(1), h.std(1))]
    return torch.cat(feats, 1).reshape(1, 13, 1536).cpu()     # [1, 13, 1536]


def classify(xt, xa, M):
    with torch.no_grad():
        probs = torch.softmax(M.fusion(xt, xa), 1)[0]
    pred = probs.argmax().item()
    return EMOTIONS[pred], probs[pred].item()


def compute_prosody(wav):
    return {"rms": float(np.sqrt(np.mean(wav ** 2))), "duration_s": len(wav) / SR}


def describe_prosody(p):
    loud = "raised volume" if p["rms"] > 0.05 else "low volume"
    return f"{loud}, {p['duration_s']:.1f}s"


def build_prompt(state, M):
    # tone goes in the system prompt; the user turn is only what they said,
    # so the model replies to the person instead of describing them
    system = (
        f"You are a warm friend in a spoken conversation. The person you're talking to "
        f"sounds {state['emotion']} ({describe_prosody(state['prosody'])}). "
        "Reply to them directly in one short sentence, speaking as 'you'. "
        "Do not describe them, do not name their emotion."
    )
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": state["transcript"]}]
    return M.llm_tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def generate(prompt, M, max_tokens=40):
    response = ""
    for chunk in stream_generate(M.llm, M.llm_tok, prompt, max_tokens=max_tokens):
        print(chunk.text, end="", flush=True)   # live streaming to terminal
        response += chunk.text
    print()
    return response


def run(wav, M):
    text = transcribe(wav) or "[unclear]"       # silence -> Whisper returns ""
    emotion, conf = classify(encode_text(text, M), encode_audio(wav, M), M)
    state = {"transcript": text, "emotion": emotion, "confidence": conf,
             "prosody": compute_prosody(wav)}

    print(f"whisper: {text}")
    print(f"pred: {emotion} ({conf:.2f}) | {describe_prosody(state['prosody'])}")
    print("llm: ", end="")
    return generate(build_prompt(state, M), M)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--replay", metavar="WAV", help="run on an existing audio file")
    g.add_argument("--record", nargs="?", const="recording.wav", metavar="OUT",
                   help="record from the mic (Enter to stop), save to OUT, then run")
    args = ap.parse_args()

    M = load_models()                           # before recording, so the reply isn't delayed
    if args.record:
        record_until_enter(args.record, sample_rate=SR)
    run(load_wav(args.record or args.replay), M)


if __name__ == "__main__":
    main()
