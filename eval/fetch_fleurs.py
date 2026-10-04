#!/usr/bin/env python3
"""Download the FLEURS Nepali test split and write the audio this benchmark expects.

    pip install datasets
    python eval/fetch_fleurs.py

Writes eval/fleurs_ne_audio/*.wav matching eval/fleurs_ne_test.jsonl.
"""
from __future__ import annotations

import json
from pathlib import Path

import soundfile as sf
from datasets import load_dataset

OUT = Path(__file__).parent / "fleurs_ne_audio"
MANIFEST = Path(__file__).parent / "fleurs_ne_test.jsonl"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    wanted = {json.loads(l)["id"] for l in open(MANIFEST, encoding="utf-8")}
    ds = load_dataset("google/fleurs", "ne_np", split="test")

    written = 0
    for i, row in enumerate(ds):
        name = f"fleurs_{i:05d}"
        if name not in wanted:
            continue
        audio = row["audio"]
        sf.write(OUT / f"{name}.wav", audio["array"], audio["sampling_rate"])
        written += 1
    print(f"{written}/{len(wanted)} files written to {OUT}")
    if written < len(wanted):
        print("Some ids were missing. The FLEURS release may have changed since "
              "this manifest was built.")


if __name__ == "__main__":
    main()
