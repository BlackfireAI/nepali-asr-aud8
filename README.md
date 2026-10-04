<img src="assets/blackfire.svg" alt="Blackfire A.I." width="260">

# Nepali ASR with punctuation (aud8, 325M)

Speech recognition for Nepali that outputs punctuated text. A fine-tune of
[Edge0/Audio8-ASR-0.1B](https://huggingface.co/Edge0/Audio8-ASR-0.1B) on 66.9 hours of
Nepali and 9.7 hours of English, with a Devanagari-extended tokenizer and a decoder
adapted on 6.57M tokens of Nepali text.

325M parameters, 16 kHz mono input, 30 second hard cap per call.

## Which model do you want

We publish two Nepali ASR models. They are not ranked, they are different tools.

| | this model (aud8) | [nepali-asr-xlsr-300m](https://github.com/BlackfireAI/nepali-asr-xlsr) |
|---|---|---|
| punctuation | yes | no |
| English inside Nepali speech | partially handled | not at all |
| accuracy on plain Nepali | good | better |
| speed | needs a GPU, below realtime on CPU | fast, many times realtime on CPU |
| architecture | autoregressive, beam search | CTC, single forward pass |
| dependencies | `trust_remote_code`, `transformers<5` | stock transformers |
| length limit | 30 seconds per call | none |

**This model is the slower and less accurate of the two.** What it gives you in exchange
is readable text with sentence boundaries, which the XLS-R model cannot produce at all.

Pick this one when the output is going to be read by a person, and you have a GPU. Pick
XLS-R for throughput, CPU deployment, long audio, or when raw accuracy matters more than
formatting.

## Install

Python 3.9 or newer. A GPU is strongly recommended, this model runs below realtime on CPU.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

That installs the exact versions this model was tested with:

```
torch 2.13.0 · transformers 4.57.6 · soundfile 0.14.0 · librosa 1.0.0
```

**The transformers version matters.** This model must run on transformers 4.x. On
transformers 5.x it does not raise an error, it silently produces nonsense output. If you
already have transformers 5 installed in another environment, use a separate virtual
environment for this model rather than downgrading.

Check it works:

```bash
python transcribe.py your_audio.wav
```

The model downloads from Hugging Face on first use, about 640 MB, and is cached
afterwards. It loads custom model code from the repository, which is why
`trust_remote_code=True` appears in the examples.

## Use

```bash
python transcribe.py audio.wav
python transcribe.py --device cpu one.wav two.wav
python transcribe.py --hotwords "काठमाडौं,सगरमाथा" audio.wav
```

From Python:

```python
from transcribe import NepaliASR

asr = NepaliASR()
print(asr("audio.wav"))
```

## Demo

Unedited output on a FLEURS test clip. Note the sentence-final danda.

```
reference  एक क्षेत्रको यात्रा भर्चुअल रूपमा साझा गर्नु पनि यात्रा प्रतिबिम्बित गर्ने र
           भविष्यका कक्षाहरूसँग अनुभव साझा गर्नेका लागि एक राम्रो तरिका हो

output     एक क्षेत्रको यात्रा भर्चुअल रूपमा साझा गर्नु पनि यात्रा प्रतिबिम्बित गर्ने र।
           भविष्यका कक्षाहरूसँग अनुभव साझा गरेका लागि एक राम्रो तरिका हो।
```

Punctuation appears on essentially every output, but placement is the weakest part of it.
The danda after `गर्ने र` splits a clause that should have continued.

## Measuring it yourself

The evaluation script and test manifest are in `eval/`. We are not publishing our own
scores for this model. Run it on audio that resembles your use case, which will tell you
more than our numbers would:

```bash
pip install datasets
python eval/fetch_fleurs.py
python eval/evaluate.py --manifest eval/fleurs_ne_test.jsonl
```

The script reports CER and WER, and what fraction of outputs carried punctuation. It
works on any JSON Lines manifest with `audio_path` and `text` fields, so you can point it
at your own test set.

## Decoding

`num_beams=5` and `no_repeat_ngram_size=4` are not optional, and the second one matters
more than it looks.

Beam search on its own makes character error **worse** than greedy decoding. The
language-model-adapted decoder is fluent enough to fall into repetition loops, and beam
search amplifies them. One ungated output was 201 characters of `छाप` repeated against a
75 character reference. The n-gram block is what stops it.

Use both settings or neither.

## Limitations

**30 seconds maximum per call.** Longer audio must be chunked. There is no built in voice
activity detection.

**Slow.** Autoregressive decoding with beam search runs below realtime on CPU. This model
wants a GPU. The XLS-R model is many times faster and runs comfortably on a laptop.

**Less accurate than our XLS-R model** on plain Nepali, on both character and word error.

**Punctuation placement is imperfect.** Sentence boundaries are often right, mid-sentence
commas and dandas frequently are not.

**Weak on heavy code-switching.** It handles occasional English words inside Nepali
speech. Sustained English, or dense technical code-switching, is not reliable.

**Requires `trust_remote_code=True`.** The architecture ships as Python files rather than
being part of transformers.

**Read and narrated speech.** The training corpus is predominantly read, broadcast and
narrated. Spontaneous conversation is harder.

**Nepali orthographic variants** (ी against ि, श against स, ब against व) account for a
noticeable share of word error, and some of it is unwinnable because the reference
transcripts themselves are inconsistent.

## Training data

66.9 hours of Nepali and 9.7 hours of English speech, drawn from the same corpus described
in the [XLS-R model repository](https://github.com/BlackfireAI/nepali-asr-xlsr).

The Nepali fine-tuning did not damage English transcription, which improved slightly.

## What changed during fine-tuning

Ranked by effect. Four of six planned improvements made results worse and were dropped.

| change | effect |
|---|---|
| unfrozen audio encoder | the largest single win, and it helps rare words most |
| decoder adapted on 6.57M tokens of Nepali text | second largest, helps mid-frequency words |
| Devanagari tokenizer, only together with text adaptation | helps |
| beam 5 with 4-gram blocking | large win at decode time, no retraining |
| SpecAugment, speed and noise augmentation | worse, the model was underfit, not overfit |
| Devanagari tokenizer alone | worse, the new embeddings had too little text to learn from |
| pseudo-labelling | worse |

The diagnosis behind this: the model was missing a large share of words it had already
seen hundreds of times in training. That is not mishearing. The confusions were
near-homophones such as भन्ने against भने, which means the decoder was the problem rather
than the audio encoder. Its Nepali perplexity before adaptation was roughly 134,000.
Adapting it on Nepali text brought that down by four orders of magnitude, and the
mid-frequency error buckets moved with it.

## Licence

CC BY-NC 4.0, inherited from the base model. **Non-commercial use only.** This restriction
comes from upstream and cannot be lifted by us or by anyone downstream.

Required attribution for derivatives: based on Edge0/Audio8-ASR-0.1B (CC BY-NC 4.0),
fine-tuned for Nepali by Blackfire A.I. Pvt. Ltd.

## Credits

Built on [Audio8-ASR-0.1B](https://huggingface.co/Edge0/Audio8-ASR-0.1B) by the Audio8
team, released under CC BY-NC 4.0. Upstream project:
[github.com/AutoArk/open-audio-opd](https://github.com/AutoArk/open-audio-opd).

Audio8-ASR is itself built on the Qwen3-ASR-0.6B audio encoder and the
Ref-Pretrain-Qwen-104M language model backbone, and is described in
*Data-Efficient On-Policy Distillation for Automatic Speech Recognition*
([arXiv:2605.28139](https://arxiv.org/abs/2605.28139)).

If you use this model, please cite the upstream work as well as ours:

```bibtex
@article{audio8asr2026,
  title   = {Data-Efficient On-Policy Distillation for Automatic Speech Recognition},
  journal = {arXiv preprint arXiv:2605.28139},
  year    = {2026}
}
```

Evaluation uses [FLEURS](https://huggingface.co/datasets/google/fleurs).

## Citation

```bibtex
@misc{blackfire2026nepaliaud8,
  title  = {Nepali ASR with punctuation: an Audio8-ASR fine-tune},
  author = {Blackfire A.I. Pvt. Ltd.},
  year   = {2026},
  url    = {https://github.com/BlackfireAI/nepali-asr-aud8}
}
```

Maintained by [Blackfire A.I. Pvt. Ltd.](https://blackfire.com.np)
