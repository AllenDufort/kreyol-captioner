"""Objective data-quality filters for prepared splits (audio, text, optional ASR mismatch).

Audio and text rules are applied to train, dev and test alike, so held-out sets lose
only clearly broken items. The ASR-mismatch rule is applied to train only, because
using a model's output to remove hard dev/test items would flatter the model.
"""

import json
import re
import unicodedata
from pathlib import Path

import numpy as np

MIN_SECONDS, MAX_SECONDS = 1.0, 30.0       # Whisper's window is 30 s
MIN_RMS = 0.002                             # essentially silent
MAX_CLIPPED = 0.005                         # share of samples at full scale
MAX_INTERIOR_ZERO_SECONDS = 0.3             # digital dropout inside speech
MAX_TRAILING_ZERO_SECONDS = 0.5             # truncated file padded with zeros
MIN_WORDS_PER_SECOND, MAX_WORDS_PER_SECOND = 0.7, 5.0   # transcript/audio mismatch
MIN_WORDS = 2
TRIM_LEVEL = 0.01                           # |x| below this counts as edge silence


def clean_text(text: str) -> tuple[str, list[str]]:
    """Return NFC text with quotes removed and whitespace collapsed, plus problems."""
    problems = []
    text = unicodedata.normalize("NFC", text)
    if any(unicodedata.category(c) in ("Cc", "Cf", "Co", "Cs") and c not in "\t\n " for c in text):
        problems.append("control_character")
    if "©" in text:
        problems.append("copyright_symbol")
    text = text.replace("’", "'")
    text = re.sub(r"[\"“”«»]", "", text)
    text = " ".join(text.split())
    if len(text.split()) < MIN_WORDS:
        problems.append("too_few_words")
    return text, problems


def _edge_trim(samples: np.ndarray) -> np.ndarray:
    loud = np.flatnonzero(np.abs(samples) >= TRIM_LEVEL)
    return samples[loud[0]:loud[-1] + 1] if loud.size else samples[:0]


def _longest_zero_run(samples: np.ndarray) -> int:
    zero = np.concatenate(([0], (samples == 0).view(np.int8), [0]))
    edges = np.diff(zero)
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return int((ends - starts).max()) if starts.size else 0


def audio_problems(samples: np.ndarray, sample_rate: int, n_words: int) -> list[str]:
    """Samples are floats in [-1, 1]. Edge digital silence is ignored for dropouts and rate."""
    problems = []
    if sample_rate != 16000:
        problems.append("sample_rate_not_16k")
    seconds = len(samples) / sample_rate if sample_rate else 0.0
    if seconds < MIN_SECONDS:
        problems.append("too_short")
    if seconds > MAX_SECONDS:
        problems.append("too_long")
    if len(samples) == 0:
        return problems + ["empty"]
    if float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))) < MIN_RMS:
        problems.append("near_silent")
    if float((np.abs(samples) >= 0.999).mean()) > MAX_CLIPPED:
        problems.append("clipped")
    nonzero = np.flatnonzero(samples != 0)
    if not nonzero.size:
        return problems + ["all_zero"]
    if (len(samples) - 1 - nonzero[-1]) / sample_rate > MAX_TRAILING_ZERO_SECONDS:
        problems.append("truncated")  # audio ends, rest of file is exact digital zeros
    speech = _edge_trim(samples)
    if _longest_zero_run(speech) / sample_rate > MAX_INTERIOR_ZERO_SECONDS:
        problems.append("dropout")
    spoken_seconds = len(speech) / sample_rate
    if spoken_seconds > 0:
        rate = n_words / spoken_seconds
        if rate > MAX_WORDS_PER_SECOND:
            problems.append("speech_rate_too_fast_for_text")
        elif rate < MIN_WORDS_PER_SECOND:
            problems.append("speech_rate_too_slow_for_text")
    return problems


def load_samples(path: Path) -> tuple[np.ndarray, int]:
    import av

    with av.open(str(path)) as container:
        rate = container.streams.audio[0].rate
        parts = [frame.to_ndarray().reshape(-1) for frame in container.decode(audio=0)]
    samples = np.concatenate(parts) if parts else np.zeros(0)
    if np.issubdtype(samples.dtype, np.integer):
        samples = samples.astype(np.float32) / np.iinfo(samples.dtype).max
    return samples.astype(np.float32), rate


def clean_split(data_dir: Path, split: str, *, asr_results: Path | None = None,
                max_cer: float = 0.8) -> tuple[list[dict], list[dict]]:
    manifest = data_dir / f"{split}.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    cer = {}
    if asr_results:
        for line in asr_results.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                cer[item["audio_id"]] = item["char"]["rate"]
    kept, rejected = [], []
    for row in rows:
        text, problems = clean_text(row["text"])
        try:
            samples, rate = load_samples(data_dir / row["audio"])
            problems += audio_problems(samples, rate, len(text.split()))
        except Exception as error:  # undecodable file
            problems.append(f"undecodable:{type(error).__name__}")
        if asr_results:
            value = cer.get(row["audio_id"])
            if value is None:
                problems.append("no_asr_result")
            elif value >= max_cer:
                problems.append(f"asr_cer>={max_cer}")
        if problems:
            rejected.append({**row, "problems": problems})
        else:
            kept.append({**row, "text": text, "text_original": row["text"]})
    return kept, rejected
