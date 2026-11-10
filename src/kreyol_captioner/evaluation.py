"""WER and CER with explicit normalization and no model dependencies."""

import json
import unicodedata
from pathlib import Path


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).casefold()
    text = "".join(" " if unicodedata.category(c).startswith("P") else c for c in text)
    return " ".join(text.split())


def edit_counts(reference: list[str], hypothesis: list[str]) -> dict:
    """Levenshtein counts using linear memory and deterministic tie breaking."""
    # Each cell stores (total errors, substitutions, deletions, insertions).
    previous = [(i, 0, 0, i) for i in range(len(hypothesis) + 1)]
    for i, ref in enumerate(reference, 1):
        current = [(i, 0, i, 0)]
        for j, hyp in enumerate(hypothesis, 1):
            if ref == hyp:
                current.append(previous[j - 1])
            else:
                diagonal, above, left = previous[j - 1], previous[j], current[-1]
                candidates = [
                    (diagonal[0] + 1, diagonal[1] + 1, diagonal[2], diagonal[3]),
                    (above[0] + 1, above[1], above[2] + 1, above[3]),
                    (left[0] + 1, left[1], left[2], left[3] + 1),
                ]
                current.append(min(candidates, key=lambda cell: cell[0]))
        previous = current
    errors, substitutions, deletions, insertions = previous[-1]
    return {
        "reference_units": len(reference), "substitutions": substitutions,
        "deletions": deletions, "insertions": insertions, "errors": errors,
        "rate": errors / len(reference) if reference else (0.0 if not hypothesis else None),
    }


def evaluate(reference: str, hypothesis: str, *, normalized: bool = True) -> dict:
    if normalized:
        reference, hypothesis = normalize(reference), normalize(hypothesis)
    word = edit_counts(reference.split(), hypothesis.split())
    char = edit_counts(list("".join(reference.split())), list("".join(hypothesis.split())))
    return {"normalized": normalized, "wer": word["rate"], "cer": char["rate"],
            "words": word, "characters": char}


def read_text(path: Path) -> str:
    content = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() != ".json":
        return content
    data = json.loads(content)
    if isinstance(data.get("segments"), list):
        return " ".join(segment["text"] for segment in data["segments"])
    if isinstance(data.get("text"), str):
        return data["text"]
    raise ValueError("JSON must contain a text field or a segments list.")
