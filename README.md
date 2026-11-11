# Kreyòl Captioner

A Python prototype that turns Haitian Creole audio or video into editable captions,
entirely on your computer. No paid APIs, account, or cloud hosting is required.

## What works in this first version

- Multilingual Whisper transcription with the language fixed to Haitian Creole (`ht`).
- Audio and video input through PyAV; no separate FFmpeg executable is needed for transcription.
- SRT, WebVTT, plain text, and JSON exports, including word timestamps.
- A local web interface with media preview and editable caption text/timing.
- WER/CER evaluation against human reference transcripts.
- Human-labeling CSVs and a trainable **traditional logistic regression** error detector.
- Separate original and reviewed exports. Silence keeps its position in the timeline.

The speech model is pretrained. No Haitian Creole fine-tuning has been performed.
An initial 15-second public-speech check produced substantial spelling and word
errors with both `base` and `small`; language support does not imply usable accuracy.
Treat the current outputs as drafts needing substantial human correction until a
representative evaluation and model adaptation demonstrate improvement.
Until you train a review classifier, review suggestions use explicitly identified
heuristics. There is no trained review model bundled with the project.
This is an offline batch prototype, not live transcription. Video rendering with
burned-in captions, automatic sound descriptions, diarization, and ASR fine-tuning
are future work.

## Quick start

Use **Python 3.10 or later** (3.12 recommended). From this project directory:

The environment and the `base`/`small` model caches have already been installed in
this workspace. To start this existing installation, run `.venv/bin/kreyol-captioner app`.
For a fresh installation elsewhere:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[ui,ml]"
kreyol-captioner app
```

On Windows, activate with `.venv\Scripts\activate` instead.
Open **http://127.0.0.1:7860**. Choose a recording, generate captions, edit the table,
then click **Save reviewed captions**. Start with `base`; compare `small` on your
recordings before choosing a model. The default is CPU INT8 inference.

The first transcription downloads model weights (roughly 150 MB for base).
Allow extra disk space for Python packages, audio/video uploads, and run exports.
Downloads need internet access; cached-model transcription can run offline:

```bash
kreyol-captioner app --offline
```

The app listens only on localhost and does not create a public sharing tunnel.
The interface uploads files to the local Python process; audio is not sent to a
transcription API. Gradio analytics are disabled. Model downloads contact Hugging
Face. Do not expose this prototype directly to the public internet.

### Command line

```bash
kreyol-captioner transcribe /path/to/recording.mp3 --model base
kreyol-captioner transcribe /path/to/video.mp4 --model small --output runs
kreyol-captioner transcribe /path/to/recording.wav --model base --offline
```

Every run gets a unique directory under `runs/` containing:

| File | Purpose |
| --- | --- |
| `captions.srt`, `captions.vtt` | Draft subtitle files |
| `transcript.txt` | Plain transcript |
| `transcript.json` | Original segments, words, scores, settings, and source hash |
| `review.csv` | Feature rows with **empty human labels** |
| `captions.reviewed.srt`, `.vtt` | Edited captions, after saving in the app |
| `transcript.reviewed.json` | Edited text and times; original ASR records are preserved |

Supported extensions include WAV, MP3, M4A, FLAC, OGG, AAC, OPUS, MP4, MOV, MKV,
WebM, and AVI. A listed extension does not guarantee every codec is decodable.
Browser video preview supports MP4/WebM; other video formats can still be transcribed.
No captions are produced if speech is not detected.

Whisper word timings are estimates. Defaults favor two lines of about 42 characters
and speech spans of at most six seconds. Pauses, unusually long words, missing word
timings, and manual edits can exceed these preferences. Review synchronization and
readability before publishing. VAD may miss quiet speech; Whisper can invent text.
For French/English code-switching, check that the actual spoken words are preserved.

## Evaluate recognition

Create a human reference `.txt` containing exactly the spoken words, or use a reviewed
JSON transcript. Compare it with the matching prediction:

```bash
kreyol-captioner evaluate \
  --reference /path/to/reference.txt \
  --hypothesis runs/your-run/transcript.json \
  --output evaluation.json
```

By default evaluation uses Unicode NFC, case folding, punctuation replaced with
spaces, and collapsed whitespace. Haitian Creole accents are retained. CER excludes
whitespace. Add `--literal` to retain case and punctuation. Rates are fractions
(0.20 means 20%). An empty reference with nonempty predictions has an undefined rate
(`null`) and explicit insertion counts. Timing and accessibility require separate review.

## Train the traditional ML error detector

1. Transcribe representative recordings and inspect every segment you label.
2. In each `review.csv`, fill `label` with `0` for acceptable text or `1` for text
   requiring correction. Leave unaudited rows empty. **Do not copy `suggested_review`
   into `label`: that would train the model to repeat its own heuristics.** Use one
   consistent policy for substantial errors, names, numbers, and meaning changes.
3. Set `group_id` to the same speaker ID for all recordings from that speaker. The
   default is a source-file SHA-256, which groups repeated runs of an identical file,
   but cannot identify a speaker across different recordings. Group related sources
   together when speakers are unknown.
4. Train on the labeled CSVs:

```bash
kreyol-captioner train-review \
  runs/first-run/review.csv runs/second-run/review.csv runs/third-run/review.csv \
  --output models/review-model.json

kreyol-captioner app --review-model models/review-model.json
```

The trainer standardizes six numeric features and fits class-balanced logistic
regression. It reserves 25% of groups for development validation, reports precision,
recall, F1, review fraction and a confusion matrix, and saves a portable JSON model.
Inference never unpickles arbitrary model files. Exact duplicate recording/segment
rows are skipped. Minimum guards require 20 labeled segments, both classes, and
three groups; a useful classifier needs substantially more representative data.

The default threshold is 0.5; `--threshold` changes it. Select a threshold using
development results and collect more data if validation has only one class. Keep a
separate final test set that is never used for model or threshold selection.
Scores are **not calibrated confidence percentages**. This component prioritizes
human review; it does not automatically rewrite the ASR output or improve raw WER.

## Data and next development stages

Start by auditing the [CMU Haitian Creole speech corpus mirror](https://huggingface.co/datasets/Frostie08/cmu_haitian_creole_speech).
It lists 6,385 examples, speaker metadata, and CMU's original license conditions.
`kreyol-captioner prepare-cmu` downloads it only when you run it (see the log below).
The mirror's single train partition is not suitable for speaker-independent evaluation
as-is.

Recommended sequence:

1. Audit 50–100 CMU examples for transcript quality, language, and speaker metadata.
2. Establish Whisper base/small results on a separate, manually verified sample of
   the kinds of recordings you want to support.
3. Build speaker/source-separated train, development, and final-test manifests.
   Remove duplicate recordings across mirrors and partitions.
4. Collect human review labels and evaluate the traditional classifier.
5. Add optional LoRA fine-tuning using PyTorch/Transformers/PEFT only after baseline
   errors and available free GPU resources justify it. Fine-tuning is not implemented
   in this first version.

The public Wikitongues [Castelline video](https://commons.wikimedia.org/wiki/File:WIKITONGUES-_Castelline_speaking_Haitian_Creole.webm)
and its [Haitian Creole subtitles](https://commons.wikimedia.org/wiki/TimedText:WIKITONGUES-_Castelline_speaking_Haitian_Creole.webm.ht.srt)
are a small end-to-end example. The video is CC BY-SA 4.0; follow the applicable
attribution and license requirements. Community subtitles need checking against
audio; do not assume they are error-free benchmark labels.

## CMU dataset preparation log (November 10, 2026)

Steps taken to build speaker-separated train/dev/test sets from the CMU mirror
(step 3 of the sequence above). Nothing was transcribed or trained in this stage.

1. **Checked readiness for training.** All six existing `runs/*/review.csv` files had
   empty `label` columns and only 5 segments in total (the trainer needs 20+ labeled
   segments, both classes, 3+ groups). `train-review` was therefore not run.
2. **Read the dataset card.** 6,385 rows in a single `train` split, five parquet shards
   (about 2 GB), no validation/test splits, `gender` is "Unknown", and `speaker_id`
   just repeats `audio_id`. CMU's notice requires keeping the copyright notice,
   marking modifications, and acknowledging the source in any published work.
3. **Installed `pyarrow`** and added an optional `data` extra to `pyproject.toml`
   (`pyarrow`, `huggingface_hub`). Re-ran `pip install -e ".[ui,ml,data]"` because
   `.venv` was still pointing at an older copy of the project in `~/Documents/Codex/`.
4. **Added `src/kreyol_captioner/data.py` and the `prepare-cmu` command.** It downloads
   the parquet shards, writes one WAV per utterance (16 kHz mono), skips byte-identical
   audio and empty transcripts, assigns whole speaker groups to a single split, then
   removes dev/test rows whose normalized text also appears in train.
5. **Tested on shard 0 only**, which exposed a problem: the first speaker rule
   (text before `_`) made 1,919 "speakers" across the full set because IDs come in
   several formats (`BLMW_4D14`, `BCMN0215`, `new-Y08`). The rule now uses the leading
   letters of the ID. This is an **inferred group, not a verified speaker label**.
6. **Added a unit test** for speaker separation, ID parsing, and text-overlap removal
   (20 tests pass; ruff clean).
7. **Ran the full preparation** with seed 42:

```bash
kreyol-captioner prepare-cmu --output data/cmu
```

| Split | Utterances | Groups | Words |
| --- | --- | --- | --- |
| train | 4,831 | 97 | 92,396 |
| dev | 146 | 6 | 3,175 |
| test | 73 | 9 | 1,481 |

1,011 dev/test rows were removed because the same sentence occurs in train. No group
appears in more than one split.

**Output** (`data/cmu/`, git-ignored): `audio/*.wav`, `references/<audio_id>.txt`
(usable with `evaluate --reference`), `train.jsonl`, `dev.jsonl`, `test.jsonl`, and
`summary.json`.

**Known limits**
- Groups are very uneven (`BCMN` has 1,446 train utterances; dev is 111 `BAMW`), so the
  dev and test sets are small and dominated by a few groups. Treat WER from them as
  rough. Changing `--seed`, `--dev`, or `--test` gives different splits.
- Many sentences are Bible-style passages, so results may not carry over to
  conversational speech like the Castelline video.
- The transcripts have not been audited by a Haitian Creole speaker (sequence step 1).
- The shards also remain in the Hugging Face cache (`~/.cache/huggingface`, about 1.9 GB
  of blobs); that copy can be deleted once `data/cmu` is built.
- Cite CMU as the source and keep its notice when publishing anything using this data.

**Baseline (done):** `kreyol-captioner benchmark --split dev --model small` measures
corpus WER/CER with the app's decoding settings and writes results to `evaluations/`.
Pretrained `small` scored about 83% WER / 27% CER on dev and 84% / 28% on test;
`base` was about 100% WER. See [VALIDATION.md](VALIDATION.md) for details and limits.

## Data cleaning and fine-tuning setup log (November 10, 2026)

Steps taken to prepare fine-tuning. **No full fine-tuning run has been done yet**; it
needs a GPU. See [training/README.md](training/README.md) for how to run it.

1. **Checked for corrupted audio.** All 6,061 WAV files decode, and all are 16 kHz.
   The damage is inside the audio: some files end early and are padded with digital
   zeros (mostly the `BLMW` set), some are almost silent, some exceed Whisper's 30-second
   window, and some are too short or too long for their transcript.
2. **Added `kreyol-captioner clean-cmu`** (`src/kreyol_captioner/clean.py`). It applies
   fixed rules to every split and writes `<split>.clean.jsonl`, `<split>.rejected.jsonl`
   (with reasons) and `clean_summary.json`. Text is NFC-normalized, straight and curly
   double quotes are removed, `’` becomes `'`, and whitespace is collapsed. Numbers are left
   as written (466 transcripts contain digits).

| Split | Kept | Rejected | Main reasons |
| --- | --- | --- | --- |
| train | 4,775 | 56 | truncated 33, speech rate doesn't fit text 35, near-silent 8, all-zero 6, over 30 s 3, control characters 2 |
| dev | 145 | 1 | under 1 s |
| test | 72 | 1 | speech too fast for the text |

   One file can have several reasons.
3. **Started a transcript-mismatch screen** for train: pretrained `small` transcribes every
   kept train file, and `clean-cmu --asr-results ...` then drops train items with CER ≥ 0.8.
   This is for train only, so held-out sets are not filtered by model difficulty.
4. **Added `training/finetune_whisper_lora.py`** (LoRA on `openai/whisper-small` with
   Transformers and PEFT; dependencies in `training/requirements-train.txt`, kept out of the
   app). It evaluates dev WER during training, keeps the best adapter, stops early, refuses
   test rows, merges the adapter, and converts to a faster-whisper model.
5. **Smoke-tested the training script on this Mac** with `whisper-tiny`, 32 train and 8 dev
   clips, in a separate environment. It trained, evaluated, saved, merged and converted,
   and the converted model ran in `kreyol-captioner benchmark`. This tests only the code,
   **not accuracy**. The test found and fixed a crash in DataLoader workers on macOS and
   a converter call that used the wrong Python environment.
6. **Added `app --model <folder>`** to show a fine-tuned model in the app's model choices,
   and fixed `benchmark` file names for local model paths.
7. 24 tests pass; ruff passes for `src/`, `tests/` and `training/`.

## Development

See [VALIDATION.md](VALIDATION.md) for the benchmark results, observed accuracy
limitations, and tested environment. `requirements.txt`
records the third-party versions used for that environment; it is not a universal
lockfile for every operating system.

```bash
python -m unittest discover -s tests -v
```

Tests cover subtitle timing and Unicode, WER/CER, preserved originals, empty labels,
JSON model validation, and training/inference on explicitly synthetic test fixtures.
They do not download models or claim Haitian Creole recognition accuracy.

The code uses MIT licensing. Whisper uses MIT; datasets and third-party dependencies
retain their own licenses. Existing hardware, electricity, internet, disk space, and
human annotation time are needed, but there are no required paid services.
