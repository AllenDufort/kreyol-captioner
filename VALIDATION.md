# Validation — November 10, 2026

Benchmark of the pretrained models on the speaker-separated CMU splits built by
`prepare-cmu` (see the README's "CMU dataset preparation log").

Environment: Python 3.12.14, macOS ARM64, CPU, INT8. faster-whisper 1.2.1,
ctranslate2 4.8.5. Pretrained Whisper `base` and `small` only: **no fine-tuning, no
trained review classifier.** Decoding is identical to the app (`DECODE_OPTIONS` in
`transcribe.py`: language `ht`, beam 5, VAD on, no previous-text conditioning).

Reproduce:

```bash
kreyol-captioner prepare-cmu --output data/cmu        # seed 42
kreyol-captioner benchmark --split dev  --model base  # also: small; test
kreyol-captioner benchmark --split test --model small
```

Results are written to `evaluations/<split>-<model>.json` (summary and per-group) and
`.utterances.jsonl` (every reference/hypothesis pair).

## Software checks

- **22 automated tests pass** (`python -m unittest discover -s tests -v`), including
  new tests for speaker-group splitting, text-overlap removal, ID parsing, pooled
  WER arithmetic, and a benchmark run with a fake model.
- Ruff passes for `src/` and `tests/`.
- `prepare-cmu` ran on all five shards: speaker leakage across splits was **zero**.
- The benchmark smoke test (10 utterances) and all four full runs completed with no
  empty hypotheses.

## Recognition accuracy (corpus-level)

Corpus-level means total errors divided by total reference words (or characters), not
an average of per-utterance rates. Text is NFC-normalized, case-folded, and
punctuation-stripped; accents are kept. WER above 100% is possible because of insertions.

| Model | Split | Utterances | Audio | WER | CER | Sub / Del / Ins (words) | Utterances with WER ≥ 100% | Speed (× real time) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | dev | 146 | 24.9 min | **100.0%** | 42.1% | 2439 / 528 / 193 | 85 | 0.19 |
| small | dev | 146 | 24.9 min | **83.1%** | 26.7% | 2077 / 440 / 111 | 27 | 0.24 |
| base | test | 73 | 11.0 min | **103.9%** | 43.2% | 1104 / 256 / 169 | 40 | 0.17 |
| small | test | 73 | 11.0 min | **84.3%** | 28.0% | 989 / 210 / 41 | 13 | 0.30 |

Speed is processing time divided by audio duration on this laptop CPU, after the model
was loaded (lower is faster; 0.24 means one minute of audio took about 14 seconds).

### What this shows

- **Neither pretrained model is usable for unedited captions.** Roughly four of every
  five words are wrong with `small`; `base` is worse than outputting nothing by WER.
- `small` is clearly better than `base` on both splits (about 17–20 points lower
  WER, and fewer insertions). Dev and test agree, which is some evidence the
  difference is real. Larger models and fine-tuning are untested.
- **CER (27–28%) is much lower than WER (83–84%).** Many words are close
  phonetically but misspelled, so a character-level view shows the models hear much of
  the speech but do not write it in Haitian Creole orthography.
- Observed error patterns in `evaluations/*.utterances.jsonl`:
  - **French-like spelling and words.** Hypotheses contain French function words
    (le, la, de, est, que…) at 9–10% of words with `base` and 4% with `small`, versus 2% in the
    references. Example: reference "leplisouvan, se anba tonè elèv yo travay." became
    "Les plus souvent c'est un bateau na il est vieux à le travail."
  - **Missing diacritics.** 13–15% of reference words contain ò/è/à; about 0–1% of hypothesis
    words do. Because accents are kept in scoring, this alone counts as an error on
    many words ("poutèt" vs "poutet"). A lenient accent-insensitive score was **not**
    computed, so the size of this effect is unknown.
  - **Numerals vs spelled-out numbers.** References spell numbers out
    ("dizuit novanm lanne diznèf san katreven uit"); hypotheses use digits ("18 novem lanei 1948").
  - **Word splitting and merging** ("lekòl" → "li kol", "poujwem coble").
  - **No empty outputs and no invented captions on silence were seen in this data**
    (0 empty hypotheses), but the CMU clips are all speech, so hallucination on silence
    was not tested here.

## Limits of this benchmark

- **Small, uneven sets.** Dev has 146 utterances and 6 groups, of which one (`BAMW`) is 111
  utterances. Test has 73 utterances and 9 groups, of which two (`EMMW`, `CWMN`) are 45. Per-group
  rates in the JSON files rest on 1–25 utterances each, and the dev and test numbers should be
  read as ±several points at best. No confidence intervals were computed.
- **Groups are inferred, not verified.** CMU's `speaker_id` repeats the utterance ID, so
  the group is the ID's leading letters. Speaker overlap across splits is therefore
  not ruled out; it is only avoided at the group level.
- **Duplicate-sentence filtering changed the sets.** 1,011 dev/test rows were removed
  because the same sentence is in train, which makes the held-out sets harder and less
  like the train set. Only exact normalized matches were removed, not near-duplicates.
- **Reference quality is unaudited.** The transcripts have not been checked by a
  Haitian Creole speaker. Some reference errors, inconsistent spelling conventions, or
  stray quote marks would count against the model.
- **Domain.** Much of the text is read, Bible-style or news speech recorded for a
  corpus. These numbers do not describe conversational audio such as the Castelline video,
  noisy phone audio, or French/English code-switching.
- **The test split was used once to report numbers, and only with pretrained models;
  it has not been used for any tuning.** If you tune decoding or thresholds, do that on dev and
  keep test for a single final report, or rebuild test with a new seed.
- **Not measured:** caption timing accuracy, subtitle readability, word-timestamp
  error, review-flag precision/recall, and reviewer time savings.

## Not yet validated

- No review classifier has been trained on real labels (all existing `review.csv`
  labels are empty), so its usefulness is unknown.
- No Whisper fine-tuning exists. The numbers above are the baseline to beat.
- Larger pretrained models (`medium`, `large-v3`) and decoding changes (beam size,
  VAD, prompting) have not been compared.
