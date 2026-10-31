# Fine-tuned Whisper-small for Singapore English (`whisper-small-mnsc-sg-15k-heldout`)

`openai/whisper-small` fine-tuned on conversational Singapore English from the IMDA National Speech Corpus (MNSC PART3). This is the transcription model used by the MVP app (`/api/transcribe`).

**File:** `whisper-small-mnsc-sg-15k-heldout.zip` (~1 GB)
**SHA256:** `F3ABBD3A0664634E528E666495768843D686DF874050B48980AFE8381A33753F`

## Install

Unzip into `mvp/models/whisper-small-mnsc-sg-15k-heldout/`, then start the app. `GET /api/health` should show `"asr": {"ready": true}`.

```powershell
curl.exe -fL "https://github.com/nuswohc/Singlish/releases/download/asr-whisper-15k-heldout-20261024/whisper-small-mnsc-sg-15k-heldout.zip" -o asr.zip
(Get-FileHash asr.zip -Algorithm SHA256).Hash   # must match the hash above
Expand-Archive asr.zip -DestinationPath mvp\models\whisper-small-mnsc-sg-15k-heldout
Remove-Item asr.zip
```

The zip contains six serving files: `config.json`, `generation_config.json`, `model.safetensors`, `processor_config.json`, `tokenizer.json` and `tokenizer_config.json`.

## Results (word error rate, lower is better)

Measured with the held-out model (`whisper-small-mnsc-sg-15k-heldout`) on speakers it never heard in training, with temperature fallback decoding. The 95% confidence intervals come from a cluster bootstrap (1,000 resamples, seed 42; resampling conversations for PART3, speakers for PART1).

| Test set | Speech type | Clips | WER [95% CI] | CER [95% CI] |
|---|---|---|---|---|
| MNSC PART1-Test | Read speech | 3,000 | 0.116 [0.108, 0.124] | 0.053 [0.049, 0.058] |
| PART3-Test, 122 held-out rows (40 conversations) | Conversation | 122 | 0.232 [0.203, 0.260] | 0.153 [0.133, 0.173] |
| Extra set (PART3-Train clips of the same 40 conversations) | Conversation | 1,969 | 0.225 [0.207, 0.242] | 0.147 [0.134, 0.159] |

Conversational speech is about twice as hard as read speech. Roughly 1 word in 4–5 is wrong on conversations, and 1 in 9 on read sentences.

Before fine-tuning, `openai/whisper-small` (same rows, same temperature fallback decoding) scored 0.337 [0.303, 0.370] on PART3-Test, 122 held-out rows and 0.335 [0.310, 0.359] on the extra set, so fine-tuning lowered conversational WER by 0.106 [0.085, 0.126] and 0.110 [0.100, 0.120]; on PART1 read speech the base model was slightly better (0.107 vs 0.116).

## What we did

1. **Data.** 14,918 PART3 conversation clips (13,507 train / 1,411 validation; 273 / 30 conversations, 546 / 60 voices), capped at 50 clips per speaker, seed 42. No PART1 read speech was mixed in.
2. **Unseen-speaker test split.** All 322 PART3-Test conversations also appear in PART3-Train, so a model trained on PART3-Train has already heard the test speakers. We held out 40 test conversations: their 80 voices (in either speaker role) and all PART1-Test voices were removed from training and validation. Training and validation were split by conversation, so no voice appears in both. An audit (`scripts/audit_test_speaker_overlap.py`) confirmed zero overlap.
3. **Training.** Whisper-small, learning rate 2e-5 (linear decay, no warmup), batch size 8 (2 per step × 4 gradient accumulation), fp16 with gradient checkpointing (fits on an 8 GB GPU). Up to 3 epochs, validation every half epoch, early stopping after 2 evaluations without improvement, best checkpoint chosen by validation loss. The best checkpoint was at epoch 2 (step 3376, eval loss 0.665). About 2 h on an RTX 3070 Laptop GPU.
4. **Decoding.** We compared decoding settings on the validation set, picked one, then ran the test set once with it. Temperature fallback was best (see below).
5. **Check against an earlier model.** The old 15k model, trained on all PART3 speakers, scored the same on the 40 held-out conversations (paired difference +0.002 WER, CI [−0.008, +0.011]). So having heard the test speakers didn't measurably inflate its score. We ship the held-out model because its numbers come from unseen speakers.

## Decoding: use temperature fallback

Plain greedy decoding sometimes gets stuck repeating a phrase ("and australia and australia …"). Temperature fallback decodes greedily first. If the output looks like a loop (token compression ratio > 1.35) or has low confidence (average log-probability < −1.0), it re-decodes with sampling at temperatures 0.2, 0.4 … 1.0.

| Decoding | Validation WER | PART3-Test, 122 held-out rows WER |
|---|---|---|
| Plain greedy | 0.297 | 0.336 |
| No-repeat 5-gram block | 0.242 | n/a |
| No-repeat 10-gram block | 0.241 | n/a |
| Temperature fallback + 5-gram block | 0.241 | n/a |
| **Temperature fallback (used)** | **0.237** | **0.232** |

The setting is **not** stored in `generation_config.json`. A plain `model.generate()` call is still greedy, so callers must pass the options. The MVP and our scripts do this by default (shared code: `scripts/whisper_decoding.py`):

```python
model.generate(input_features, language="en", task="transcribe",
               temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
               compression_ratio_threshold=1.35, logprob_threshold=-1.0)
```

Re-decoded clips use random sampling, so their output can vary slightly with the seed (42), GPU or library version.

## Compatibility

- Saved with transformers 5.16.1. Under transformers 4.57.6 (the MVP's pin), `WhisperProcessor.from_pretrained` fails on this tokenizer config. `mvp/transcription.py` contains the workaround, and with it, outputs are token-identical to 5.16.1.
- On CPU (float32), a 10 s clip takes about 3–5 s. The model adds about 1.2 GB of RAM (1.6 GB at peak). Measured in the MVP; see the Transcription section of `mvp/README.md`.

## Limitations

- **Domain:** trained and tested on MNSC recordings. Phone microphones, noise and other accents will differ, so treat the WER as a benchmark, not a guarantee.
- **Small held-out set:** the PART3 numbers come from 40 conversations, so the confidence intervals are about ±3 WER points. The two PART3 test sets share those conversations, so they are not independent.
- **Local terms:** on unseen speakers, the model recognises 76% of named Singapore terms (places, food, acronyms such as HDB and MRT) and 64% of Singlish particles (lah, leh, lor, meh). Frequent terms are mostly right. Errors cluster in rare names, for example COE → "ceoe" and Choa Chu Kang → "trajokang". Details: `docs/part2_terms_analysis.md`.
- **Inconsistent labels:** the training transcripts mix some spellings (lor/loh, hor/orh), which counts as errors in WER.
- **Silence:** Whisper can invent text on silence or noise. Trim silence before transcribing (the MVP does).
- **Long audio:** clips over 30 s need chunking (`scripts/transcribe.py` and the MVP use 30 s windows).

## Licence

Fine-tuned from `openai/whisper-small` (MIT) on the IMDA National Speech Corpus via `AudioLLMs/Multitask-National-Speech-Corpus-v1-extend`. Redistribution of the weights may be subject to the IMDA corpus licence terms.
