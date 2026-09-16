#!/usr/bin/env python3
"""
Verify the MELD audio/label join.

Checks, per split, that every row in the official annotation CSV has a
matching audio file, and reports anything that doesn't line up in either
direction. Writes a clean manifest of usable (row, audio) pairs.

Usage:
    python verify_data.py
    python verify_data.py --audio-dir data/audio --labels-dir data/labels
    python verify_data.py --manifest data/manifest.csv
    python verify_data.py --split dev
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

SPLITS = {"train": "train_sent_emo.csv", "dev": "dev_sent_emo.csv", "test": "test_sent_emo.csv"}
EXPECTED_ROWS = {"train": 9989, "dev": 1109, "test": 2610}
EMOTIONS = ["neutral", "joy", "surprise", "anger", "sadness", "disgust", "fear"]


def load_labels(path: Path) -> pd.DataFrame:
    """Read a MELD CSV, coping with the cp1252 encoding it ships in."""
    for enc in ("cp1252", "utf-8", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc)
            df.attrs["encoding"] = enc
            return df
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"could not decode {path} with cp1252/utf-8/latin-1")


def check_split(split: str, audio_dir: Path, labels_dir: Path, ext: str, n_show: int):
    csv_path = labels_dir / SPLITS[split]
    split_audio = audio_dir / split

    if not csv_path.exists():
        print(f"[{split}] MISSING CSV: {csv_path}")
        return None
    if not split_audio.is_dir():
        print(f"[{split}] MISSING AUDIO DIR: {split_audio}")
        return None

    df = load_labels(csv_path)
    df["stem"] = "dia" + df["Dialogue_ID"].astype(str) + "_utt" + df["Utterance_ID"].astype(str)

    want = set(df["stem"])
    have = {p.stem for p in split_audio.glob(f"*{ext}")}

    missing = want - have   # annotated rows with no audio -> must drop
    extra = have - want     # audio with no annotation     -> ignore
    usable = want & have

    dupes = len(df) - len(want)

    print(f"\n=== {split} ===")
    print(f"  csv encoding used : {df.attrs['encoding']}")
    print(f"  rows in csv       : {len(df)}" + (f"  (expected {EXPECTED_ROWS[split]})" if len(df) != EXPECTED_ROWS[split] else ""))
    if dupes:
        print(f"  DUPLICATE ids     : {dupes}  <-- same dia/utt appears more than once")
    print(f"  audio files found : {len(have)}")
    print(f"  usable pairs      : {len(usable)}")
    print(f"  missing audio     : {len(missing)}")
    print(f"  unlabelled audio  : {len(extra)}  (ignored)")

    if missing:
        shown = sorted(missing)[:n_show]
        print(f"  missing ids       : {', '.join(shown)}" + (" ..." if len(missing) > n_show else ""))
        lost = df[df["stem"].isin(missing)]
        print(f"  lost by emotion   : {lost['Emotion'].value_counts().to_dict()}")

    if extra:
        shown = sorted(extra)[:n_show]
        print(f"  extra ids         : {', '.join(shown)}" + (" ..." if len(extra) > n_show else ""))

    kept = df[df["stem"].isin(usable)].copy()
    kept["split"] = split
    kept["audio_path"] = kept["stem"].map(lambda s: str(split_audio / f"{s}{ext}"))

    dist = kept["Emotion"].value_counts()
    total = len(kept)
    print("  emotion distribution:")
    for emo in EMOTIONS:
        n = int(dist.get(emo, 0))
        print(f"    {emo:<9} {n:>5}  ({100 * n / total:5.2f}%)")

    return kept


def main():
    ap = argparse.ArgumentParser(description="Verify the MELD audio/label join.")
    ap.add_argument("--audio-dir", type=Path, default=Path("data/audio"))
    ap.add_argument("--labels-dir", type=Path, default=Path("data/labels"))
    ap.add_argument("--ext", default=".flac", help="audio file extension (default: .flac)")
    ap.add_argument("--split", choices=list(SPLITS), help="check one split only")
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"),
                    help="where to write the joined manifest")
    ap.add_argument("--no-manifest", action="store_true", help="verify only, write nothing")
    ap.add_argument("--n-show", type=int, default=10, help="how many offending ids to print")
    args = ap.parse_args()

    splits = [args.split] if args.split else list(SPLITS)

    frames, failed = [], False
    for split in splits:
        kept = check_split(split, args.audio_dir, args.labels_dir, args.ext, args.n_show)
        if kept is None:
            failed = True
        else:
            frames.append(kept)

    if not frames:
        print("\nnothing verified -- check --audio-dir and --labels-dir", file=sys.stderr)
        return 1

    manifest = pd.concat(frames, ignore_index=True)
    print(f"\ntotal usable utterances: {len(manifest)}")

    if not args.no_manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        cols = ["split", "stem", "audio_path", "Utterance", "Speaker", "Emotion",
                "Sentiment", "Dialogue_ID", "Utterance_ID", "Season", "Episode",
                "StartTime", "EndTime"]
        manifest[[c for c in cols if c in manifest.columns]].to_csv(args.manifest, index=False)
        print(f"manifest written to {args.manifest}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
