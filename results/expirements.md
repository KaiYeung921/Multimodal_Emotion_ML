## text-only, bare utterance, no context
### *Claude generated reports*
roberta-base frozen, masked-mean pooling, 2-layer MLP (768→256→128→7)
10k full-batch epochs, no dropout, no early stopping — heavily overfit
train wF1 0.976 / dev wF1 0.505

              precision  recall  f1  support
neutral          0.67     0.75  0.71    469
joy              0.44     0.47  0.45    163
...
macro avg        0.36     0.35  0.35   1108
weighted avg     0.50     0.52  0.50   1108

## Run 3 — context k=1, regularised

Input: previous 1 turn with speaker tag, mean over all tokens.

Head: 768 → 256 → 128 → 7, BatchNorm + ReLU + Dropout(0.5) after each hidden layer.
AdamW lr 1e-4, weight decay 1e-2. Mini-batches of 256.
Class weights: 1/√count, normalised to sum to 7.
Early stopping on dev loss, patience 10, best weights restored (~epoch 65).

| | wF1 | mF1 |
|---|---|---|
| train | 0.664 | — |
| dev | **0.510** | **0.39** |


          precision    recall  f1-score   support
 neutral       0.67      0.65      0.66       469
     joy       0.51      0.48      0.50       163
surprise       0.43      0.48      0.45       150
   anger       0.37      0.35      0.36       153
 sadness       0.39      0.37      0.38       111
 disgust       0.19      0.36      0.25        22
    fear       0.12      0.12      0.12        40


## Run 4 — bare vs context, regularised (final text comparison)

Same head and training as Run 3. Only the input string changes.

| context | dev wF1 | mF1 | anger F1 | anger recall |
|---|---|---|---|---|
| bare | **0.574** | **0.44** | 0.39 | 0.38 |
| k=1  | 0.510 | 0.39 | 0.36 | 0.35 |
| k=3  | 0.461 | 0.35 | 0.39 | 0.42 |

Context hurts monotonically under mean-over-all-tokens pooling. Anger is the
exception — flat F1, rising recall — consistent with context carrying real
signal that the pooling then dilutes. Target-span pooling is the obvious fix;
not attempted within the timebox.

**Text baseline going forward: bare utterance, 0.574 / 0.44.**
Anger (0.39) remains the weakest non-rare class — the pre-registered target
for the audio branch.
