#!/usr/bin/env python3
"""Cache WavLM features (13 layers, mean+std pooled) for every MELD clip."""

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from transformers import AutoFeatureExtractor, WavLMModel

dev = "mps" if torch.backends.mps.is_available() else "cpu"
fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
wlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").eval().to(dev)

manifest = pd.read_csv("data/manifest.csv")
X = np.zeros((len(manifest), 19968), dtype=np.float16)

for i, path in enumerate(manifest.audio_path):
    wav, _ = sf.read(path, dtype="float32")
    wav = wav[:16000 * 15]                                   # cap at 15s
    inp = fe(wav, sampling_rate=16000, return_tensors="pt").to(dev)
    with torch.no_grad():
        hs = wlm(**inp, output_hidden_states=True).hidden_states
    feats = [f for h in hs for f in (h.mean(1), h.std(1))]
    X[i] = torch.cat(feats, 1).squeeze(0).cpu().numpy()
    if i % 500 == 0:
        print(i)
    if i % 1000 == 0 and i:
        np.save("cache/audio_partial.npy", X)               # crash recovery

np.save("cache/audio_wavlm_13L.npy", X)
print(X.shape)
