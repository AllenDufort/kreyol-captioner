# Initial validation — November 8, 2026

Environment: Python 3.12.14, macOS ARM64, CPU inference. Dependencies were
installed in the project's `.venv`; `requirements.txt` records the resolved
third-party versions for this environment. A separate FFmpeg executable was not
installed or required for transcription.

## Completed checks

- **19 automated tests passed** using `python -m unittest discover -s tests -v`.
- Ruff checks passed for `src/` and `tests/`.
- `pip check` reported no broken dependency requirements.
- Whisper `base` and `small` loaded on CPU using INT8, without a paid service.
- A 15-second WAV excerpt of Wikitongues' Castelline recording produced SRT,
  VTT, text, JSON, and a labeling CSV. Base-model inference was also run with
  cached weights in offline mode.
- The complete 237.761-second WebM video was transcribed with cached `base`
  weights: **37 speech segments and 48 subtitle cues** were exported, with
  29 segments suggested for review. These are workflow counts, not accuracy measures.
- A three-second silent WAV produced zero segments and no invented captions.
- In the actual browser, uploaded the audio excerpt, generated captions, edited
  a caption, saved reviewed files, and checked that the edited text was present
  on disk while the original transcript remained unchanged.
- Restarted the current app and verified it renders at `http://127.0.0.1:7860`.

## Findings and limits

Both pretrained `base` and `small` produced substantial word and spelling errors
on the short Haitian Creole sample, including French-like spellings. The short
sample's errors were not reliably caught by the preliminary review rules.
**The current speech output needs human correction.** Model language support
does not establish useful recognition accuracy.

No representative human-verified benchmark or final test evaluation has been
completed. No real-data review classifier has been trained. The classifier's
automated training test uses explicitly synthetic fixtures only. No Whisper
fine-tuning has been implemented or performed.

The next priority is to audit CMU data, assemble an independent reference set,
measure recognition accuracy, collect consistent human labels, and then evaluate
adaptation. Word timing, difficult audio, code-switching, sound descriptions,
and subtitle readability still need human evaluation.

An audio-decoding compatibility failure was found with PyAV 19 during real-media
testing. The project now constrains PyAV to versions 14–16; testing succeeded
with PyAV 16.1.0. Sample-video credits and license links are in the README and
`runs/ATTRIBUTION.md`.
