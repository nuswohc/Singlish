# Accent classifier dataset: Singaporean vs non-Singaporean English

Builds the final 3–6 s accent-classification dataset from four public sources.
No archive is ever unzipped: audio is decoded in memory and stored as FLAC bytes
inside parquet files.

| Class | Group | Source | Clips | Speakers |
|---|---|---|---|---|
| Singaporean | Singaporean | MNSC Extended PART1-Train (Hugging Face) | 5,000 | 1,032 |
| Non-Singaporean | American, British, South Asian | Common Voice 27.0 English | 800 each | 800 each |
| Non-Singaporean | Australian | Common Voice 27.0 English | 800 | 625 |
| Non-Singaporean | Filipino | Common Voice 27.0 English | 800 | 144 |
| Non-Singaporean | Hong Kong | Common Voice 27.0 English | 685 | 120 |
| Non-Singaporean | Malaysian | Common Voice 27.0 English | 672 | 105 |
| Non-Singaporean | Mainland Chinese | speechocean762 (adults only) | 800 | 127 |
| Non-Singaporean | Korean | Speech Accent Archive (HF mirror) | 329 | 52 |
| Out-of-domain check only | Singaporean | Common Voice 27.0 English | 554 | 81 |

Training classes: 5,000 Singaporean vs 6,486 non-Singaporean clips.
Hong Kong, Malaysian and the Common Voice Singaporean group use every
available speaker at the 15-clip cap, so they can't reach 800.

All clips: 16 kHz mono, 3–6 s after silence trimming, normalised, max 15 clips
per speaker. Train/val/test (70/15/15) is split by speaker, so no speaker
appears in more than one split.

The Common Voice "Singaporean" group is NOT training data. Train on MNSC
Singaporean, then check whether the model still recognises these Common Voice
Singaporean clips. If it doesn't, it learned the dataset rather than the accent.

## Requirements

- Windows/macOS/Linux, Python 3.14 (developed on 3.14.7)
- About 100 GB free disk (the Common Voice archive is 95 GB)
- A Mozilla Data Collective (MDC) account for Common Voice
- Internet access for the first run of each step

## Quick start with the shared data (no downloads)

If you were given `accent_dataset_3to6s.zip`, skip "Run from scratch":

1. `pip install -r requirements.txt`
2. Unzip it into this folder's `data\` so you get `data\cv_subset\`,
   `data\mnsc_subset\`, `data\speechocean_subset\`, `data\saa_subset\`.
   In PowerShell: `Expand-Archive accent_dataset_3to6s.zip -DestinationPath data`
3. Load it as shown in "Load the data" below.

The zip is for private use within the project team and course staff only.
Don't post it publicly (see Licenses).

## Run from scratch

Run every command from the `accent_classifier` folder. Each script creates its
own output folders and replaces its previous output, so re-running is safe.

### 0. Install

```powershell
pip install -r requirements.txt
```

### 1. Common Voice access (one-time, manual)

1. Sign in at https://mozilladatacollective.com and open
   "Common Voice Scripted Speech 27.0 - English". Accept the dataset terms.
2. Create an API key in your MDC dashboard.
3. Copy `.env.example` to `.env` and paste the key after `MDC_API_KEY=`.
   Never commit or share `.env`.

### 2. Download Common Voice English (~95 GB, ~2–3 h)

```powershell
python scripts\download_common_voice.py
```

Saved to `raw\`. If it stops (network drop, Ctrl+C), run the same command
again and it resumes. Keep the `.part` and `.checksum` files.

### 3. Scan metadata (seconds)

```powershell
python scripts\scan_common_voice_metadata.py
```

Reads only the `.tsv` metadata, maps accent labels to groups, and prints
clips/speakers per group to `data\cv_accent_counts.csv`. Accent rules are at
the top of the script (`ACCENT_RULES`).

### 4. Extract the Common Voice subset (~15 min)

```powershell
python scripts\extract_common_voice_subset.py --accent_groups Malaysian "Hong Kong" Filipino "South Asian" American British Australian Singaporean --max_duration 6 --oversample 1.6
```

Output: `data\cv_subset\`. Don't interrupt it: it deletes the previous subset
at the start.

`--oversample 1.6` selects 60% extra clips up front. Durations in the
metadata are measured before silence trimming, so about a third of the
selected clips end up outside 3–6 s and are dropped. The default 1.25 left
most groups short of 800.

### 5. Mainland Chinese and Korean (~1.5 GB download, a few minutes)

```powershell
python scripts\prepare_speechocean762.py --max_duration 6
python scripts\prepare_speech_accent_archive.py --max_duration 6 --chunk_target_min 4 --chunk_target_max 5.5
```

Output: `data\speechocean_subset\` and `data\saa_subset\`. Downloads go to
`raw\`. Add `--no_download` on later runs to reuse them.

### 6. Singaporean (MNSC)

```powershell
python scripts\prepare_mnsc_singaporean.py --max_duration 6 --oversample 2.0
```

Output: `data\mnsc_subset\`. Downloads 4 PART1-Train shards (~2 GB) into the
Hugging Face cache. `--oversample 2.0` is needed to reach 5,000 clips at 3–6 s;
the default 1.25 gives only ~3,700. The transcription project's test sets
(PART1-Test, PART3-Test) are not used. Speakers the fine-tuned Whisper model
trained on are forced into train (0 overlap with PART1).

## Load the data

```python
import sys; sys.path.insert(0, "scripts")
from cv_utils import load_accent_data

df = load_accent_data()   # all sources, all splits

# IMPORTANT: hold out the Common Voice Singaporean clips (out-of-domain check)
ood = (df.source == "common_voice") & (df.accent_group == "Singaporean")
check = df[ood]                       # evaluate on these only, never train on them
df    = df[~ood]

train = df[df.split == "train"]
val   = df[df.split == "val"]
test  = df[df.split == "test"]
```

Each row has `audio` (float32 array, 16 kHz), `is_singaporean` (the label),
`accent_group`, `source`, `speaker_id`, `split`, `duration_sec`, `sentence`.
`load_accent_data(sources=["mnsc"])` or `accent_groups=["Korean"]` filters.

## Known limitations

- Korean (329 clips, 8 test speakers) is smaller than the other groups; use
  class weights and treat its test score as indicative.
- Mainland Chinese and Korean each come from a single corpus, and every Korean
  speaker reads the same paragraph. Report accuracy per source to check the
  model isn't learning recording conditions or text.
- About half the Korean chunk boundaries fall back to the quietest point
  rather than a detected pause, so some cuts may land mid-word.

## Licenses

- Common Voice: CC0
- speechocean762: CC BY 4.0 (Zhang et al., 2021)
- Speech Accent Archive: non-commercial use (Weinberger, 2015); third-party HF mirror
- MNSC: derived from IMDA Singapore's National Speech Corpus, redistributed on
  Hugging Face by AudioLLMs. The HF card states no license; check IMDA's NSC terms
  (https://www.imda.gov.sg/how-we-can-help/national-speech-corpus).

This dataset is for a non-commercial school project.

## Tests

```powershell
python -m unittest discover -s tests
```
