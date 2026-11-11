"""Local Whisper inference and reproducible run exports."""

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

from . import __version__
from .models import Segment, Transcript, Word
from .review import annotate, load_review_model, write_label_template
from .subtitles import make_captions, validate_captions, write_subtitles

EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".wav", ".mp3", ".m4a",
              ".flac", ".ogg", ".aac", ".opus", ".aiff", ".wma"}
# Shared by captioning and benchmarking so measured accuracy matches the app.
DECODE_OPTIONS = {"language": "ht", "task": "transcribe", "beam_size": 5, "word_timestamps": True,
                  "vad_filter": True, "condition_on_previous_text": False}


@lru_cache(maxsize=1)
def get_model(name: str, device: str, compute_type: str, cache_dir: str, offline: bool):
    try:
        import onnxruntime
        from faster_whisper import WhisperModel
    except ImportError as error:
        raise RuntimeError("Install the project first: pip install -e .") from error
    onnxruntime.disable_telemetry_events()
    return WhisperModel(name, device=device, compute_type=compute_type,
                        download_root=cache_dir, local_files_only=offline)


def source_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(
    source: Path, output_root: Path, *, model_name: str = "base", device: str = "cpu",
    compute_type: str = "int8", cache_dir: Path = Path(".cache/models"),
    offline: bool = False, review_model: Path | None = None, progress=None,
) -> tuple[Transcript, Path, list]:
    source = source.expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"Input does not exist: {source}")
    if source.suffix.lower() not in EXTENSIONS:
        raise ValueError("Unsupported audio/video extension.")
    if source.stat().st_size == 0:
        raise ValueError("The input file is empty.")
    classifier = load_review_model(review_model) if review_model else None
    if progress:
        progress(0.05, desc="Loading speech recognition model")
    started = time.perf_counter()
    model = get_model(model_name, device, compute_type, str(cache_dir.resolve()), offline)
    predicted, info = model.transcribe(str(source), **DECODE_OPTIONS)
    segments = []
    for raw in predicted:
        if not raw.text.strip() or raw.end <= raw.start:
            continue
        segments.append(Segment(
            id=len(segments), start=float(raw.start), end=float(raw.end), text=raw.text.strip(),
            words=[Word(float(w.start), float(w.end), w.word) for w in (raw.words or [])],
            avg_logprob=float(raw.avg_logprob), no_speech_prob=float(raw.no_speech_prob),
            compression_ratio=float(raw.compression_ratio),
        ))
        if progress:
            progress(min(0.9, 0.1 + 0.8 * raw.end / max(info.duration, 0.01)),
                     desc=f"Transcribed {raw.end:.0f} of {info.duration:.0f} seconds")
    annotate(segments, classifier)
    captions = make_captions(segments)
    validate_captions(captions, float(info.duration))
    elapsed = time.perf_counter() - started
    now = datetime.now(timezone.utc)
    recording_id = f"{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    transcript = Transcript(
        source=str(source), recording_id=recording_id, duration=float(info.duration), segments=segments,
        metadata={
            "created_utc": now.isoformat(), "app_version": __version__,
            "faster_whisper_version": version("faster-whisper"), "model": model_name,
            "device": device, "compute_type": compute_type, "beam_size": 5,
            "task": "transcribe", "language": "ht", "word_timestamps": True,
            "vad_filter": True, "condition_on_previous_text": False,
            "review_method": "trained_logistic_regression" if classifier else "heuristic",
            "review_model_sha256": source_hash(review_model) if review_model else None,
            "source_sha256": source_hash(source), "processing_seconds_including_model_load": elapsed,
            "real_time_factor_including_model_load": elapsed / info.duration if info.duration else None,
        },
    )
    stem = re.sub(r"[^\w-]+", "-", source.stem, flags=re.UNICODE).strip("-")[:60] or "recording"
    directory = output_root.expanduser().resolve() / f"{stem}-{recording_id}"
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "transcript.json").write_text(
        json.dumps(transcript.to_dict(), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (directory / "transcript.txt").write_text(transcript.text + "\n", encoding="utf-8")
    write_subtitles(captions, directory)
    write_label_template(transcript, directory / "review.csv")
    if progress:
        progress(1.0, desc="Captions ready")
    return transcript, directory, captions
