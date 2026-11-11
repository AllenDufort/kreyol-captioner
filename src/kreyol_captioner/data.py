"""Download the CMU Haitian Creole mirror and build speaker-separated splits.

Requires the optional ``data`` extra (pyarrow, huggingface_hub). Nothing here runs
unless the ``prepare-cmu`` command is invoked.
"""

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from .evaluation import normalize

REPO = "Frostie08/cmu_haitian_creole_speech"
SHARDS = 5


def speaker_of(audio_id: str) -> str:
    """The mirror's speaker_id just repeats audio_id, so infer a group from the ID.

    IDs look like BLMW_4D14, BCMN0215, BAMW0966_x01 or new-Y08. The leading letters
    (BLMW, BCMN, new) recur and are used as the speaker/session group. This is an
    assumption: a coarse group may hold several speakers, which is conservative for
    leakage but not a verified speaker label.
    """
    match = re.match(r"[A-Za-z]+", audio_id)
    return match.group(0).upper() if match else audio_id


def _extension(name: str | None, payload: bytes) -> str:
    if payload[:4] == b"RIFF":
        return ".wav"
    if payload[:4] == b"fLaC":
        return ".flac"
    if payload[:4] == b"OggS":
        return ".ogg"
    if payload[:3] == b"ID3" or payload[:2] in (b"\xff\xfb", b"\xff\xf3"):
        return ".mp3"
    return Path(name).suffix if name and Path(name).suffix else ".wav"


def extract(output: Path, shards: list[int], cache_dir: Path | None = None) -> list[dict]:
    """Download parquet shards and write one audio file per unique utterance."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    audio_dir = output / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    records, seen_audio = [], set()
    for index in shards:
        name = f"data/train-{index:05d}-of-{SHARDS:05d}.parquet"
        print(f"Fetching {name} ...", flush=True)
        path = hf_hub_download(REPO, name, repo_type="dataset", cache_dir=cache_dir)
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
            for row in batch.to_pylist():
                audio = row["audio"] or {}
                payload = audio.get("bytes")
                if not payload or not str(row.get("text") or "").strip():
                    continue
                digest = hashlib.sha256(payload).hexdigest()
                if digest in seen_audio:
                    continue
                seen_audio.add(digest)
                audio_id = str(row["audio_id"])
                file = audio_dir / f"{audio_id}{_extension(audio.get('path'), payload)}"
                file.write_bytes(payload)
                records.append({
                    "audio_id": audio_id, "speaker": speaker_of(audio_id),
                    "audio": str(file.relative_to(output)), "sha256": digest,
                    "text": row["text"].strip(),
                })
    return records


def split_by_speaker(records: list[dict], dev: float, test: float, seed: int) -> dict[str, list[dict]]:
    """Assign whole speakers to one split, aiming at the requested utterance fractions."""
    by_speaker = defaultdict(list)
    for record in records:
        by_speaker[record["speaker"]].append(record)
    speakers = sorted(by_speaker)
    random.Random(seed).shuffle(speakers)
    total = len(records)
    targets = {"test": test * total, "dev": dev * total}
    splits: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    # Fill test, then dev, with whole speakers; the rest is train.
    for name in ("test", "dev"):
        for speaker in list(speakers):
            if len(splits[name]) + len(by_speaker[speaker]) <= targets[name] * 1.25:
                splits[name].extend(by_speaker[speaker])
                speakers.remove(speaker)
            if len(splits[name]) >= targets[name]:
                break
    for speaker in speakers:
        splits["train"].extend(by_speaker[speaker])
    return splits


def drop_text_overlap(splits: dict[str, list[dict]]) -> int:
    """Remove dev/test rows whose normalized text also appears in train."""
    train_text = {normalize(r["text"]) for r in splits["train"]}
    dropped = 0
    for name in ("dev", "test"):
        kept = [r for r in splits[name] if normalize(r["text"]) not in train_text]
        dropped += len(splits[name]) - len(kept)
        splits[name] = kept
    return dropped


def write_splits(output: Path, splits: dict[str, list[dict]], dropped: int, seed: int) -> dict:
    refs = output / "references"
    refs.mkdir(parents=True, exist_ok=True)
    summary = {"seed": seed, "dropped_text_overlap": dropped, "splits": {}}
    owner = {}
    for name, rows in splits.items():
        with (output / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                row["split"] = name
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                owner.setdefault(row["speaker"], set()).add(name)
                (refs / f"{row['audio_id']}.txt").write_text(row["text"] + "\n", encoding="utf-8")
        summary["splits"][name] = {
            "utterances": len(rows), "speakers": len({r["speaker"] for r in rows}),
            "words": sum(len(r["text"].split()) for r in rows),
        }
    summary["speaker_leakage"] = sorted(s for s, names in owner.items() if len(names) > 1)
    counts = Counter(r["speaker"] for rows in splits.values() for r in rows)
    summary["utterances_per_speaker"] = {
        "min": min(counts.values(), default=0), "max": max(counts.values(), default=0),
        "speakers": len(counts),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def prepare(output: Path, *, shards: list[int] | None = None, dev: float = 0.1, test: float = 0.1,
            seed: int = 42, keep_text_overlap: bool = False, cache_dir: Path | None = None) -> dict:
    if not (0 < dev < 0.5 and 0 < test < 0.5):
        raise ValueError("--dev and --test must each be between 0 and 0.5.")
    shards = list(range(SHARDS)) if shards is None else shards
    if any(i < 0 or i >= SHARDS for i in shards):
        raise ValueError(f"Shard numbers must be 0-{SHARDS - 1}.")
    output.mkdir(parents=True, exist_ok=True)
    records = extract(output, shards, cache_dir)
    if not records:
        raise ValueError("No usable examples were found.")
    splits = split_by_speaker(records, dev, test, seed)
    dropped = 0 if keep_text_overlap else drop_text_overlap(splits)
    return write_splits(output, splits, dropped, seed)
