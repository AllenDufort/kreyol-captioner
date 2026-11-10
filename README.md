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
This project does not download the full corpus automatically or assume the mirror's
train partition is suitable for speaker-independent evaluation.

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

## Development

See [VALIDATION.md](VALIDATION.md) for the initial real-media and browser checks,
observed accuracy limitations, and tested environment. `requirements.txt`
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
