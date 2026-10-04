#!/usr/bin/env python3
"""Transcribe Nepali audio with punctuation.

    python transcribe.py audio.wav
    python transcribe.py --device cpu one.wav two.wav
    python transcribe.py --hotwords "काठमाडौं,सगरमाथा" audio.wav

Requires transformers>=4.50,<5. Audio longer than 30 seconds must be chunked.
"""
from __future__ import annotations

import argparse
import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

DEFAULT_MODEL = "BlackfireAI/nepali-asr-aud8-325m"
MAX_SECONDS = 30


class NepaliASR:
    def __init__(self, model: str = DEFAULT_MODEL, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, trust_remote_code=True, dtype=torch.bfloat16,
            attn_implementation="eager").to(self.device).eval()

    @torch.no_grad()
    def __call__(self, path: str, hotwords: str | None = None) -> str:
        prompt = "Please transcribe this audio."
        if hotwords:
            prompt = f"{prompt} Expect these terms: {hotwords}."
        conversation = [{"role": "user", "content": [
            {"type": "audio", "path": path}, {"type": "text", "text": prompt}]}]
        batch = dict(self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, return_tensors="pt",
            sampling_rate=16000, audio_padding="longest",
            audio_max_length=MAX_SECONDS * 16000,
            text_kwargs={"padding": "longest", "truncation": True, "max_length": 1000}))
        batch = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in batch.items()}
        out = self.model.generate(**batch, max_new_tokens=256, do_sample=False,
                                  num_beams=5, no_repeat_ngram_size=4)
        return self.processor.decode(out[0, batch["input_ids"].shape[1]:],
                                     skip_special_tokens=True).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Transcribe Nepali speech with punctuation.")
    ap.add_argument("audio", nargs="+")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", choices=["cuda", "cpu"])
    ap.add_argument("--hotwords", help="comma separated terms to bias decoding toward")
    ap.add_argument("--json", help="write results to this file instead of stdout")
    a = ap.parse_args()

    asr = NepaliASR(a.model, a.device)
    results = []
    for path in a.audio:
        try:
            text = asr(path, a.hotwords)
        except Exception as e:
            print(f"{path}: ERROR {e}", file=sys.stderr)
            continue
        results.append({"audio": path, "text": text})
        if not a.json:
            print(f"{path}\t{text}" if len(a.audio) > 1 else text)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=1)
        print(f"{len(results)} transcripts -> {a.json}")


if __name__ == "__main__":
    main()
