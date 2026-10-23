# Accent classifier dataset: Singaporean vs non-Singaporean English

Builds a 30,000-clip accent-classification dataset from four public sources.
Audio is decoded in memory and stored as FLAC bytes inside parquet files.

| Class | Group | Source | Clips |
|---|---|---|---|
| Singaporean | Singaporean | MNSC Extended PART1-Train (Hugging Face) | 7,500 |
| Singaporean | Singaporean | MNSC Extended PART2-Train (Hugging Face) | 7,500 |
| Non-Singaporean | American, British, Australian | Common Voice English | 2,369 each |
| Non-Singaporean | Filipino, South Asian | Common Voice English | 2,368 each |
| Non-Singaporean | Hong Kong | Common Voice English | 2,028 |
| Non-Singaporean | Mainland Chinese | speechocean762 (adults only) | 800 |
| Non-Singaporean | Korean | Speech Accent Archive (HF mirror) | 329 |

Training classes: 15,000 Singaporean vs 15,000 non-Singaporean clips.

All clips: 16 kHz mono, 3–6 s after silence trimming, normalised.
Train/val/test (70/15/15) is split by speaker, so no speaker appears in more
than one split, including across MNSC parts. Sampling spreads clips across
speakers; caps are 30 per MNSC speaker and 200 per Common Voice speaker.
MNSC speaker splits use seed 42.

## Requirements

- Windows/macOS/Linux, Python 3.14 (developed on 3.14.7)
- About 150 GB free disk (the Common Voice archive is about 95 GB)
- A Mozilla Data Collective (MDC) account for Common Voice
- Internet access for the first run of each step

## Quick start with the shared data

Download [accent_dataset_3to6s_v2.zip](https://drive.google.com/file/d/1GFQ-ooGlfThktEJdG1hTwq-ruCs9xUzS/view?usp=drive_link)
from Google Drive, then skip "Run from scratch":

1. `pip install -r requirements.txt`
2. Unzip it into this folder's `data\` so you get `data\cv_subset\`,
   `data\mnsc_subset\`, `data\speechocean_subset\`, `data\saa_subset\`.
   In PowerShell: `Expand-Archive accent_dataset_3to6s_v2.zip -DestinationPath data`
3. Run `python scripts\verify_dataset.py` and load it as shown below.

The zip is for private use within the project team and course staff only.
Don't post it publicly (see Licenses).

## Run from scratch

Run every command from the `accent_classifier_data` folder. Each script creates
its own output folders and replaces its previous output.

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

### 2. Download Common Voice English

```powershell
python scripts\download_common_voice.py
```

Saved to `raw\`. If interrupted, run the same command again to resume.
Keep the `.part` and `.checksum` files.

### 3. Scan metadata

```powershell
python scripts\scan_common_voice_metadata.py
```

Reads the `.tsv` metadata, maps accents to groups, and writes
`data\cv_accent_counts.csv`. Accent rules are in `ACCENT_RULES` in the script.

### 4. Extract the Common Voice subset

```powershell
python scripts\extract_common_voice_subset.py
```

Output: `data\cv_subset\`, with 13,871 clips across the six groups above.
The defaults select extra candidates to allow for clips dropped after silence
trimming. If a group is short, the script reports it; increase `--oversample`
or `--max_clips_per_speaker` and rerun.

### 5. Mainland Chinese and Korean

```powershell
python scripts\prepare_speechocean762.py
python scripts\prepare_speech_accent_archive.py
```

Output: `data\speechocean_subset\` and `data\saa_subset\`. Downloads go to
`raw\`. Add `--no_download` on later runs to reuse them.

### 6. Singaporean (MNSC)

```powershell
python scripts\prepare_mnsc_singaporean.py
```

Output: `data\mnsc_subset\`, with 7,500 PART1-Train and 7,500 PART2-Train clips.
The script downloads the required shards into the Hugging Face cache and
includes the extended PART1 selection automatically. No separate project or
local data folder is required. Test configurations are excluded.

### 7. Check the dataset

```powershell
python scripts\verify_dataset.py
```

Checks counts, labels and speaker separation, and decodes every clip to check
duplicates, duration and preprocessing outputs. Reports are saved in
`data\verification\`. Boundary-trimming warnings need a raw-input check.

## Load the data

```python
import sys; sys.path.insert(0, "scripts")
from cv_utils import load_accent_data

df = load_accent_data()
train = df[df.split == "train"]
val   = df[df.split == "val"]
test  = df[df.split == "test"]
```

Each row has `audio` (float32 array, 16 kHz), `is_singaporean` (the label),
`accent_group`, `source`, `speaker_id`, `split`, `duration_sec`, `sentence`.
`load_accent_data(sources=["mnsc"])` or `accent_groups=["Korean"]` filters.
Use `decode_audio=False` to load encoded audio with less memory.

## Known limitations

- Korean (329 clips) is smaller than the other groups; treat its test score
  as indicative.
- Mainland Chinese and Korean each come from a single corpus, and every Korean
  speaker reads the same paragraph. Report accuracy per source to check the
  model isn't learning recording conditions or text.
- Some Korean chunk boundaries fall at the quietest point rather than a
  detected pause, so some cuts may land mid-word.

## Licenses

- Common Voice: CC0
- speechocean762: CC BY 4.0 (Zhang et al., 2021)
- Speech Accent Archive: non-commercial use (Weinberger, 2015); third-party HF mirror
- MNSC: derived from IMDA Singapore's National Speech Corpus, redistributed on
  Hugging Face by AudioLLMs. The HF card states no license; check IMDA's NSC terms
  (https://www.imda.gov.sg/how-we-can-help/national-speech-corpus).

This dataset is for a non-commercial school project.
