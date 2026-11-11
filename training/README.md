# Fine-tuning Whisper for Haitian Creole

LoRA fine-tuning of `openai/whisper-small` on the cleaned CMU splits. Training needs a
CUDA GPU (a free Colab or Kaggle T4 is enough); the app itself does not.

## 1. Clean the data (on your computer)

```bash
kreyol-captioner prepare-cmu --output data/cmu          # if not done yet
kreyol-captioner clean-cmu                              # audio/text rules only

# Optional but recommended: screen train for transcripts that don't match the audio.
# Slow on CPU (about 2-3 hours for small); fast on a GPU.
kreyol-captioner benchmark --split train.clean --model small --output evaluations/train-screen
kreyol-captioner clean-cmu --asr-results evaluations/train-screen/train.clean-small.utterances.jsonl
```

`clean-cmu` writes `<split>.clean.jsonl`, `<split>.rejected.jsonl` (with reasons) and
`clean_summary.json`. Listen to some rejected files before trusting the filter.
The ASR screen is applied to **train only**: removing hard dev/test items with a model
would make the model look better than it is.

## 2. Train (on a GPU)

Upload the project folder (code plus `data/cmu/audio` and the `*.clean.jsonl` files) to
Colab or Kaggle. Then:

```bash
pip install -r training/requirements-train.txt
python training/finetune_whisper_lora.py --data data/cmu --output finetuned/small-ht
```

Useful options: `--epochs 4`, `--batch-size 16`, `--lr 1e-3`, `--speed-perturb`
(augmentation), `--max-train 64 --max-dev 16` (quick smoke test).
The script evaluates dev WER (greedy decoding) every 200 steps, keeps the best adapter,
stops early when dev stops improving, and never reads the test split.

Outputs in `finetuned/small-ht/`: `best-adapter/`, `merged/`, `ctranslate2/`
(the faster-whisper model), `history.json` and `summary.json`.

## 3. Use and measure (on your computer)

Download `finetuned/small-ht/ctranslate2/`, then:

```bash
kreyol-captioner benchmark --split dev.clean --model finetuned/small-ht/ctranslate2
kreyol-captioner app --model finetuned/small-ht/ctranslate2
kreyol-captioner transcribe file.wav --model finetuned/small-ht/ctranslate2
```

Compare against pretrained `small` on the same `dev.clean` split. Run the test split
**once**, after every choice is final:

```bash
kreyol-captioner benchmark --split test.clean --model finetuned/small-ht/ctranslate2
```

The model is converted with float16 weights; the app's default `--compute-type int8`
quantizes them at load time.
