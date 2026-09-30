# Emotion-aware responses from voice: a Text + Audio prototype on MELD

A small, local, end-to-end prototype for a character robot: you speak one turn, and it returns **structured emotional state** (a MELD emotion category plus confidence) and a **short streamed reply grounded in what you said and how you said it. The demo below uses 2 wav files I recorded myself**.

```
$ python3 scripts/main.py --replay samples/Happy.wav
whisper: Yeah, that's just great.
emotion: joy (0.68)
response: You know it's meant to be.
```

![demo](demo.gif)

**Track:** Text + Audio. Everything runs locally on an 8 GB M1 MacBook Air, with **3.38B total parameters** (cap: 6B).

---

## Contents
- [How it works](#how-it-works)
- [Quickstart](#quickstart)
- [What "real-time" means here](#what-real-time-means-here)
- [Results](#results)
- [Parameter budget](#parameter-budget)
- [Hardware and resource use](#hardware-and-resource-use)
- [Design decisions and trade-offs](#design-decisions-and-trade-offs)
- [Limitations](#limitations)
- [Completed vs. intentionally left out](#completed-vs-intentionally-left-out)
- [Reproducing training](#reproducing-training)
- [External components](#external-components)
- [Repository layout](#repository-layout)

---

## How it works

One spoken utterance travels through the system to both outputs:

```
                  ┌─> Whisper-base.en ──> transcript ──> RoBERTa-base ──> text vec [768] ─┐
 mic / wav ──> 16 kHz mono                                                               ├─> Fusion MLP ──> emotion + confidence ─┐
                  ├─> WavLM-base-plus ──> 13 layers × (mean,std) [13×1536] ──────────────┘   (learned layer weights)            │
                  │                                                                                                              ├─> Qwen2.5-3B ──> streamed reply
                  └─> prosody (RMS loudness, duration) ───────────────────────────────────────────────────────────────────────────┘
                  transcript ─────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

1. **Input:** a wav file (`--replay`) or a push-to-talk mic recording (`--record`). Any sample rate or channel count is converted to 16 kHz mono.
2. **Speech → text:** Whisper-base.en (MLX) transcribes the utterance.
3. **Text features:** frozen RoBERTa-base, with attention-masked mean pooling, produces a 768-d vector.
4. **Audio features:** frozen WavLM-base-plus. Mean and standard deviation are pooled over time for all 13 hidden layers, giving 13 × 1536.
5. **Fusion classifier (trained here):** a learned softmax weighting collapses WavLM's 13 layers into one 1536-d vector. It is concatenated with the text vector and passed through a 2-layer MLP (2304 → 256 → 128 → 7). The output is one of the 7 MELD emotions plus its softmax probability.
6. **Structured state** is passed to the generator:
   ```python
   {"transcript": "Yeah, that's just great.",
    "emotion": "joy", "confidence": 0.68,
    "prosody": {"rms": 0.031, "duration_s": 2.5}}      # samples/Happy.wav
   ```
7. **Response:** Qwen2.5-3B-Instruct (4-bit, MLX) gets the emotion and prosody in its system prompt and the transcript as the user turn. It streams one short reply token by token and is told not to name the emotion.

---

## Quickstart

**Requirements:** an Apple Silicon Mac (MLX), Python 3.13, and a microphone for `--record`. The first run downloads about 2.5 GB of pretrained weights from Hugging Face. After that everything runs offline.

```bash
pip install -r requirements.txt

# run on an existing audio file (wav/flac/ogg, any sample rate, mono or stereo)
python3 scripts/main.py --replay samples/Happy.wav

# record from the mic, press Enter to stop (saves recording.wav, or the path you give)
python3 scripts/main.py --record
python3 scripts/main.py --record recordings/take1.wav

# parameter / latency / memory report
python3 scripts/benchmark.py --replay samples/Happy.wav > results/benchmark.md
```

The trained fusion head is loaded from `cache/fusion_model.pt`. The other models come from Hugging Face.

Paths are relative to the folder you run from. `.m4a` voice memos need converting first: `ffmpeg -i in.m4a out.wav`.

---

## What "real-time" means here

**The target interaction is turn-based conversation with a character robot.** The robot waits for the speaker to finish a turn, then responds. The goal is for the reply to start within about 1 s of the end of the turn, which is roughly the pause people leave between turns in conversation. The reply then streams in, so the robot can start speaking before the sentence is complete.

**Assumptions:**
- One speaker, English, one utterance per turn, usually 1–15 s. MELD utterances average about 3 s.
- The end of the turn is marked explicitly (push-to-talk: Enter). There is no voice-activity detection.
- Emotion is labeled once per utterance, as in MELD, not frame by frame.

**Measured on the M1 (warm, median of 5, 2.5 s clip):**

| Stage | ms |
|---|---:|
| Load + resample audio | 1 |
| Whisper transcribe | 122 |
| RoBERTa text encode | 23 |
| WavLM audio encode | 94 |
| Fusion MLP classify | 1 |
| Prosody + prompt build | 0 |
| LLM first token | 496 |
| LLM full reply (~9 tokens) | 797 |
| **End of turn → first reply token** | **~740** |
| **End of turn → full reply** | **~1,035** |

The LLM decodes at about 33 tokens/s. **Caveat:** each `main.py` run is a new process. It pays about 6 s to load the models and a slower first pass (about 9–17 s, mostly one-time GPU kernel compilation and lazy weight loading) before the warm numbers apply. With `--record`, loading happens while you speak. A persistent multi-turn session would only pay this once (see [left out](#completed-vs-intentionally-left-out)).

---

## Results

All models use the same classifier head and training recipe: AdamW (lr 1e-4, wd 1e-2), BatchNorm + Dropout 0.5, class weights of 1/√count, and early stopping. The text and audio encoders are frozen, and only the head is trained. The splits are MELD's official train (9,988), dev (1,108) and test (2,610).

| Model | dev wF1 | dev mF1 | **test wF1** | **test mF1** |
|---|---:|---:|---:|---:|
| Text only (RoBERTa, bare utterance) | 0.574 | 0.44 | — | — |
| Audio only (WavLM, 13 layers) | 0.439 | 0.30 | — | — |
| **Fusion (text + audio)** | **0.609** | **0.47** | **0.618** | **0.45** |

wF1 is weighted F1 and mF1 is macro F1. Dev was used for early stopping, so **test is the unbiased number**. The single-modality baselines were only evaluated on dev.

**Per-class dev F1: each modality contributes something different**

| Class | Text | Audio | Fusion |
|---|---:|---:|---:|
| neutral | 0.72 | 0.60 | 0.74 |
| joy | 0.54 | 0.21 | 0.58 |
| surprise | 0.62 | 0.38 | 0.63 |
| anger | 0.39 | **0.51** | **0.52** |
| sadness | 0.39 | 0.31 | 0.43 |
| disgust | 0.16 | 0.00 | 0.12 |
| fear | 0.22 | 0.08 | 0.24 |

Audio is much weaker overall, but it was the only modality that beat text on **anger** (recall 0.38 → 0.67). That's the case this project was built around: words that sound harmless but are said angrily. Fusion keeps text's strengths and picks up audio's anger signal, improving dev wF1 by 3.5 points over text alone. Disgust (22 dev examples) and fear (40) remain weak for every model.

Full experiment log: [`results/expirements.md`](results/expirements.md).

**Things I tried that didn't help:** adding previous dialogue turns as context. With mean pooling over all tokens, performance *dropped* as context grew: bare 0.574, k=1 0.510, k=3 0.461 dev wF1. The context dilutes the target utterance, so the final system uses the bare utterance.

---

## Parameter budget

Every learned parameter on the local inference path counts toward the total, not just active or trainable ones. Counted by `scripts/benchmark.py`:

| Component | Role | Parameters |
|---|---|---:|
| `mlx-community/whisper-base.en-mlx` | Speech recognition | 71,825,418 |
| `roberta-base` | Text encoder | 124,645,632 |
| `microsoft/wavlm-base-plus` | Audio encoder | 94,381,936 |
| Fusion MLP (this repo) | Layer weights + classifier head | 624,660 |
| `mlx-community/Qwen2.5-3B-Instruct-4bit` | Response generator | 3,085,938,688 |
| **Total** | | **3,377,416,334 (3.38B), 56% of the 6B cap** |

PyTorch models are counted with `numel()`. The MLX models are counted from their parameter trees. For the 4-bit LLM, each `uint32` stores 8 packed weights, so a naive element count reports only 0.48B. The script unpacks them and excludes the quantization scales, which matches Qwen's published 3.09B.

---

## Hardware and resource use

| | |
|---|---|
| Machine | MacBook Air, **Apple M1, 8 GB** unified memory, macOS 26.2 |
| Frameworks | PyTorch 2.10 on MPS (RoBERTa, WavLM, fusion), MLX 0.32.3 (Whisper, Qwen) |
| **Peak memory** | **4.17 GB physical footprint** |
| Disk | About 2.5 GB of model downloads, plus about 1.1 GB for the cached WavLM training features |
| Model load | About 6 s, from local cache |

Memory is reported as macOS *physical footprint*, which is what Activity Monitor shows, and it was verified against `/usr/bin/time -l`. RSS isn't used because it misses GPU buffers in unified memory (it reports only 0.44 GB for the same run).

---

## Design decisions and trade-offs

| Decision | Why | Trade-off |
|---|---|---|
| **Text + Audio track** | A robot hears speech. Tone of voice is exactly the signal that changes how words should be read, e.g. sarcasm and anger. | No facial-expression signal. |
| **Frozen encoders + small trained head** | About 10k training utterances is too few to fine-tune 100M+ parameter encoders without overfitting. Caching features made each training run take seconds on an M1. | Leaves some accuracy on the table compared with fine-tuning. |
| **All 13 WavLM layers, mean + std** | Emotion cues live in different layers than words do. Standard deviation over time captures variation in pitch and energy that the mean loses. | 19,968-d features per clip (1.1 GB cache). |
| **Learned layer weights in the fusion model** | Lets the model pick which WavLM layers to use instead of hand-picking. | 13 extra parameters, so negligible. |
| **Bare utterance, no dialogue context** | Measured: context hurt under mean pooling (see Results). | Loses conversational cues. Pooling only over the target utterance's tokens might recover them (not tried). |
| **Whisper-base.en** | 74M parameters, about 120 ms per utterance. Budget-friendly, and fast. | Makes mistakes that can flip the emotion (see Limitations). |
| **Qwen2.5-3B-Instruct, 4-bit** | Tested 0.5B, 1.5B and 3B with the same prompt. 0.5B described the speaker instead of replying, and 1.5B gave generic replies. 3B is the smallest that engaged with the content. 4-bit keeps it at about 2 GB. | About 0.5 s to the first token, and it uses 91% of the parameter budget. |
| **Tone in the system prompt, transcript as the user turn** | When the transcript was quoted inside a report-style prompt, the LLM described the person instead of replying to them. | The reply relies on a classifier label that may be wrong. |
| **MLX for Whisper + LLM, PyTorch/MPS for the encoders** | MLX is the fastest local option on Apple Silicon. The encoders are standard `transformers` models. | Two frameworks, and the project is tied to Apple Silicon. |
| **Recorder in a child process** | Stopping a PortAudio mic stream on macOS sometimes deadlocks inside CoreAudio. The child writes the wav file and exits without stopping the stream. | One extra process start per recording. |

---

## Limitations

- **Train/inference text mismatch.** The fusion head was trained on MELD's gold transcripts but runs on Whisper's output at inference. On a small check of 10 clips, gold text got 7/10 emotions right and Whisper text got 4/10. Whisper errors pushed predictions toward neutral. The full dev-set comparison hasn't been run, so treat this as anecdotal.
- **Low confidence and fragile predictions.** Many predictions have 0.3–0.4 confidence. Small changes in wording, such as "I mean you'd" vs. "But you would", can flip the label. The same sentence said in a happy and a neutral voice gave the same label, so on those clips the text dominated.
- **Domain shift.** MELD is scripted TV dialogue (*Friends*) with a laugh track and studio acoustics. Laptop-mic speech is quieter and differs in delivery. The loudness threshold in the prosody description was set from MELD clips.
- **Crude prosody for the LLM.** It only gets RMS loudness and duration. Pitch and speaking rate aren't used.
- **Rare classes.** Disgust and fear have F1 ≤ 0.24.
- **LLM behaviour.** Replies sometimes role-play as the other person in the story. They also follow the predicted label even when it's wrong.
- **Single-shot CLI.** Each run handles one turn and pays the model load and cold-start cost.
- **Minor preprocessing mismatch.** Training features cap clips at 15 s, but live inference doesn't.
- **Platform.** Apple Silicon only, because of MLX.

---

## Completed vs. intentionally left out

**Completed**
- A Text + Audio emotion classifier on MELD: text-only, audio-only and fusion models, compared on the official splits.
- An end-to-end local pipeline, audio → transcript + emotion state → streamed grounded reply, with `--replay` and `--record`.
- Measured parameter count, per-stage latency and peak memory on the target hardware (`scripts/benchmark.py`).
- Data verification of the MELD audio/label join (`scripts/verify_data.py`).

**Intentionally left out**
- **Vision / three-modality extension.** I focused on doing one track well.
- **Reinforcement learning.**
- **Continuous multi-turn session, voice-activity detection, text-to-speech.** The prototype handles one push-to-talk turn per run.
- **Fine-tuning the encoders, or training on Whisper transcripts.** These are the most likely next accuracy gains.
- **Dialogue context with better pooling.**

---

## Reproducing training

1. **Data.** Put these under `data/`, which is gitignored except `data/manifest.csv`:
   - MELD annotations from [declare-lab/MELD](https://github.com/declare-lab/MELD) → `data/labels/{train,dev,test}_sent_emo.csv`
   - Audio from [ajyy/MELD_audio](https://huggingface.co/datasets/ajyy/MELD_audio) (a lossless FLAC re-encode of the official mp4s, 16 kHz mono) → `data/audio/{train,dev,test}/dia{D}_utt{U}.flac`

   `python3 scripts/verify_data.py` checks the join and writes `data/manifest.csv`. 13,706 of 13,708 utterances are usable. `dia125_utt3` (train) and `dia110_utt7` (dev) have no audio and are dropped (both neutral). 141 unlabeled audio files, mostly duplicates, are ignored.
2. **Cache features** (frozen encoders, run once):
   ```bash
   python3 scripts/tokenize_data.py     # RoBERTa → cache/text.npy        [13706 × 768]
   python3 scripts/cache_audio.py       # WavLM   → cache/audio_wavlm_13L.npy [13706 × 19968]
   ```
3. **Train** in the notebooks: `Text_MLP.ipynb`, `AudioMLP.ipynb`, `Fusion_MLP.ipynb`. The last one writes `cache/fusion_model.pt`, which `main.py` uses.

---

## External components

| Component | Source | Used for |
|---|---|---|
| MELD | declare-lab | Training/evaluation data and labels |
| MELD audio (FLAC) | `ajyy/MELD_audio` on Hugging Face | Audio for training |
| RoBERTa-base | `roberta-base` | Frozen text encoder |
| WavLM-base-plus | `microsoft/wavlm-base-plus` | Frozen audio encoder |
| Whisper-base.en | `mlx-community/whisper-base.en-mlx` | Speech recognition |
| Qwen2.5-3B-Instruct (4-bit) | `mlx-community/Qwen2.5-3B-Instruct-4bit` | Response generation |

**Trained in this repo:** the fusion head (`cache/fusion_model.pt`) and the text-only and audio-only baselines.

**AI assistance:** Claude was used. I mainly used claude to help organize and clean a lot of my code, for example I just wrote this README in txt format and had claude convert all of it to markdown. Claude also helped in structuring the project and helping me think through design choices.
---

## Repository layout

```
scripts/
  main.py            end-to-end pipeline: --replay <wav> | --record [out.wav]
  transcribe.py      push-to-talk mic recorder (runs as its own process)
  models.py          TextMLP / AudioMLP / FusionMLP definitions
  benchmark.py       parameter, latency and memory report
  verify_data.py     MELD audio/label join check → data/manifest.csv
  tokenize_data.py   cache RoBERTa features
  cache_audio.py     cache WavLM features
notebooks/           training + analysis (Text_MLP, AudioMLP, Fusion_MLP, ...)
cache/               cached features + trained heads (fusion_model.pt)
results/             experiment log, benchmark output
samples/             example clips for --replay
```
