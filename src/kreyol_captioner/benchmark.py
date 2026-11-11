"""Corpus-level WER/CER over a prepared split, using the same decoding as the app."""

import json
import time
from collections import defaultdict
from pathlib import Path

from .evaluation import edit_counts, normalize
from .transcribe import DECODE_OPTIONS, get_model


def _rate(errors: int, units: int):
    return errors / units if units else None


def _pool(rows: list[dict]) -> dict:
    """Corpus-level rates: total errors over total reference units, not a mean of rates."""
    out = {"utterances": len(rows)}
    for kind in ("word", "char"):
        errors = sum(r[kind]["errors"] for r in rows)
        units = sum(r[kind]["reference_units"] for r in rows)
        out[kind] = {"errors": errors, "reference_units": units, "rate": _rate(errors, units),
                     "substitutions": sum(r[kind]["substitutions"] for r in rows),
                     "deletions": sum(r[kind]["deletions"] for r in rows),
                     "insertions": sum(r[kind]["insertions"] for r in rows)}
    return out


def run_benchmark(data_dir: Path, split: str, model_name: str, output: Path, *, limit: int | None = None,
                  device: str = "cpu", compute_type: str = "int8",
                  cache_dir: Path = Path(".cache/models"), offline: bool = False) -> dict:
    manifest = data_dir / f"{split}.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"No examples in {manifest}")
    model = get_model(model_name, device, compute_type, str(cache_dir.resolve()), offline)
    results, audio_seconds = [], 0.0
    started = time.perf_counter()
    for index, row in enumerate(rows, 1):
        segments, info = model.transcribe(str(data_dir / row["audio"]), **DECODE_OPTIONS)
        hypothesis = " ".join(s.text.strip() for s in segments if s.text.strip())
        ref, hyp = normalize(row["text"]), normalize(hypothesis)
        audio_seconds += float(info.duration)
        results.append({
            "audio_id": row["audio_id"], "group": row["speaker"], "duration": float(info.duration),
            "reference": row["text"], "hypothesis": hypothesis, "empty": not hyp,
            "word": edit_counts(ref.split(), hyp.split()),
            "char": edit_counts(list("".join(ref.split())), list("".join(hyp.split()))),
        })
        if index % 25 == 0 or index == len(rows):
            print(f"{model_name} {split}: {index}/{len(rows)}", flush=True)
    elapsed = time.perf_counter() - started
    by_group = defaultdict(list)
    for r in results:
        by_group[r["group"]].append(r)
    summary = {
        "model": model_name, "split": split, "device": device, "compute_type": compute_type,
        "decode_options": DECODE_OPTIONS, "normalization": "NFC, casefold, punctuation removed",
        "overall": _pool(results),
        "empty_hypotheses": sum(r["empty"] for r in results),
        "audio_seconds": audio_seconds, "processing_seconds": elapsed,
        "real_time_factor": elapsed / audio_seconds if audio_seconds else None,
        "by_group": {g: _pool(rs) for g, rs in sorted(by_group.items())},
        "utterances_wer_ge_1": sum((r["word"]["rate"] or 0) >= 1 for r in results),
    }
    output.mkdir(parents=True, exist_ok=True)
    stem = f"{split}-{Path(model_name).name if Path(model_name).exists() else model_name}"
    (output / f"{stem}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output / f"{stem}.utterances.jsonl").open("w", encoding="utf-8") as handle:
        for r in results:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")
    return summary
