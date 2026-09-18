#!/usr/bin/env python3

import pandas as pd, numpy as np, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

ROOT = Path(__file__).parent.parent
manifest = pd.read_csv(ROOT / 'data' / 'manifest.csv')

MODEL = 'roberta-base'
BATCH = 64
dev = "mps" if torch.backends.mps.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(MODEL)
enc = AutoModel.from_pretrained(MODEL).to(dev)
enc.eval()

K = 3

tagged = "[" + manifest["Speaker"] + "] " + manifest["Utterance"]

Contexts = tagged.copy()
for i in range(1, K + 1):
    prev = tagged.groupby([manifest["split"], manifest["Dialogue_ID"]]).shift(i).fillna("")
    Contexts = prev + " " + Contexts

Contexts = Contexts.str.strip().tolist()



output = []


batch_size = 64
with torch.no_grad():
    for i in range(0, len(Contexts), batch_size):
        b = tok(Contexts[i:i+batch_size], padding=True,truncation=True, max_length=128,return_tensors="pt")
        b = b.to(dev)
        h = enc(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        output.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())   # pool all vectors into 1 (one sentance could have 6 vectors another could have 20)
        print(f"{min(i+BATCH, len(Contexts))}/{len(Contexts)}", end="\r")

X = np.concatenate(output)
print(X.shape)   # (13706, 768)
        #outputs = enc(**batch)   # or model(**batch).last_hidden_state


CACHE = ROOT / 'cache'
CACHE.mkdir(exist_ok=True)
np.save(CACHE / 'text_concat.npy', X)
manifest[["stem", "split", "Emotion"]].to_csv(CACHE / 'index.csv', index=False)
print(f"\n{X.shape} -> cache/text_concat.npy")

