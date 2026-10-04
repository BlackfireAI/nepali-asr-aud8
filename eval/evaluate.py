#!/usr/bin/env python3
"""Score the model on a test manifest and report CER and WER.

Manifest is JSON Lines with "audio_path" and "text" per line. An optional "src"
field splits the report into subsets.

    python eval/evaluate.py --manifest eval/fleurs_ne_test.jsonl
    python eval/evaluate.py --manifest test.jsonl --out results.json

Decoding uses num_beams=5 with no_repeat_ngram_size=4. Both are required. See the
Decoding section of the README for the measurements behind that.
"""
from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

PUNCT = re.compile(r"[।,.?!;:\"'’‘“”\-–—()\[\]{}/\\|]")
MAX_SECONDS = 30


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", PUNCT.sub(" ", text)).strip()


def edit_distance(ref: list, hyp: list) -> int:
    if len(ref) < len(hyp):
        ref, hyp = hyp, ref
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def score(refs: list[str], hyps: list[str]) -> dict:
    word_err = word_tot = char_err = char_tot = 0
    for r, h in zip(refs, hyps):
        word_err += edit_distance(r.split(), h.split())
        word_tot += len(r.split())
        char_err += edit_distance(list(r.replace(" ", "")), list(h.replace(" ", "")))
        char_tot += len(r.replace(" ", ""))
    return {"n": len(refs),
            "cer": round(char_err / max(char_tot, 1), 4),
            "wer": round(word_err / max(word_tot, 1), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", default="BlackfireAI/nepali-asr-aud8-325m")
    ap.add_argument("--device", choices=["cuda", "cpu"])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", help="write full results and hypotheses here")
    a = ap.parse_args()

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        torch.set_num_threads(a.threads)
    processor = AutoProcessor.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, trust_remote_code=True, dtype=torch.bfloat16,
        attn_implementation="eager").to(device).eval()

    manifest = Path(a.manifest)
    rows = [json.loads(l) for l in open(manifest, encoding="utf-8") if l.strip()]
    # Relative audio paths are resolved against the manifest, not the shell's working
    # directory, so the benchmark runs the same from anywhere in the repo.
    for row in rows:
        if not Path(row["audio_path"]).is_absolute():
            row["audio_path"] = str((manifest.parent / row["audio_path"]).resolve())
    print(f"{len(rows)} utterances on {device}", flush=True)

    refs, hyps, raws, srcs = [], [], [], []
    audio_seconds = 0.0
    start = time.time()
    with torch.no_grad():
        for i, row in enumerate(rows, 1):
            conversation = [{"role": "user", "content": [
                {"type": "audio", "path": row["audio_path"]},
                {"type": "text", "text": "Please transcribe this audio."}]}]
            batch = dict(processor.apply_chat_template(
                conversation, add_generation_prompt=True, return_tensors="pt",
                sampling_rate=16000, audio_padding="longest",
                audio_max_length=MAX_SECONDS * 16000,
                text_kwargs={"padding": "longest", "truncation": True, "max_length": 1000}))
            batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
            out = model.generate(**batch, max_new_tokens=256, do_sample=False,
                                 num_beams=5, no_repeat_ngram_size=4)
            raw = processor.decode(out[0, batch["input_ids"].shape[1]:],
                                   skip_special_tokens=True).strip()
            audio_seconds += row.get("duration", 0.0)
            raws.append(raw)
            hyps.append(normalise(raw))
            refs.append(normalise(row["text"]))
            srcs.append(row.get("src", "all"))
            if i % 25 == 0:
                print(f"  {i}/{len(rows)}  {time.time() - start:.0f}s", flush=True)

    elapsed = time.time() - start
    punctuated = sum(bool(re.search(r"[।,.?!]", r)) for r in raws)
    results = {"model": a.model, "manifest": a.manifest, "device": device,
               "decode": "beam5_norepeat4", "overall": score(refs, hyps),
               "rtf": round(elapsed / max(audio_seconds, 1), 4),
               "realtime_factor": round(audio_seconds / elapsed, 2),
               "pct_outputs_with_punctuation": round(100 * punctuated / len(raws), 1)}
    for s in sorted(set(srcs)):
        if s != "all":
            keep = [i for i, v in enumerate(srcs) if v == s]
            results[s] = score([refs[i] for i in keep], [hyps[i] for i in keep])
    print(json.dumps(results, indent=1))

    if a.out:
        payload = {"results": results,
                   "hypotheses": [{"src": s, "ref": r, "hyp": h, "raw": w}
                                  for s, r, h, w in zip(srcs, refs, hyps, raws)]}
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
        print(f"written to {a.out}")


if __name__ == "__main__":
    main()
