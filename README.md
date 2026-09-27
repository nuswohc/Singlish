# Singlish POC

Proof of concept for Singapore English accent classification.

The repository contains simple Colab notebooks for each experiments.

Audio, feature caches, trained models and notebook outputs stay in each member's Google
Drive.

## Colab notebooks

- `notebooks/E1_01_svm_beginner.ipynb`
- `notebooks/E1_02_random_forest_beginner.ipynb`
- `notebooks/E1_03_logistic_regression_beginner.ipynb`
- `notebooks/E2_01_whisper_classifier_beginner.ipynb`
- `notebooks/E3_01_unsupervised_learning_beginner.ipynb`

Open a notebook in Colab and configure the Drive path and audio limit:

```python
PROJECT_ROOT = Path('/content/drive/MyDrive/Singlish')
MAX_CLIPS_PER_CLASS = None  # Use all audio, or set a positive limit such as 100.
```

The E1 and E2 notebooks choose up to this many clips from each class, spreading
the selection across speakers with a fixed random seed. Use the same value in
each notebook to compare the same recordings. E3 reads the E2 embedding cache
for the matching limit, so run E2 through its cache cell first. If you add or
remove audio that changes E2's selected file list, set `REBUILD_CACHE = True`
in E2 for that run.

## Google Drive layout

Each team member uploads the prepared `Singlish` folder to `MyDrive`:

```text
Singlish/
├── data/
│   └── audio/
│       ├── singapore/speaker_###/*.wav
│       └── non_singapore/speaker_###/*.wav
```

Only `data/audio` must be uploaded. A notebook creates its own output folder
when it runs; for example, the E1 notebooks create `Singlish/outputs/e1`.

Use the same audio for every experiment. E1 uses acoustic features, E2 creates
Whisper embeddings, E3 clusters the embeddings, and E4 combines the
supervised model probabilities. The beginner notebooks create their own
speaker-disjoint split with the same random seed.

'Singlish' folder shared at following Google drive link:
  https://drive.google.com/drive/folders/1rg_NL97iBC9lwSBO9bz7idZ2tkMJHWqi?usp=drive_link

## Simple team workflow

All three members may use `main`. Pull before editing, avoid editing the same
notebook simultaneously, clear large notebook outputs, and commit one coherent
change at a time.

Do not commit audio, models, credentials or Google Drive outputs.
