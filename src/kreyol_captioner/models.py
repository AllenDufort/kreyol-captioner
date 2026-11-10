"""Dependency-free, serializable transcription records."""

import math
from dataclasses import asdict, dataclass, field
from typing import Any


def check_times(start: float, end: float, *, allow_zero: bool = False) -> None:
    if not math.isfinite(start) or not math.isfinite(end):
        raise ValueError("Timestamps must be finite numbers.")
    if start < 0 or end < start or (end == start and not allow_zero):
        raise ValueError("Timestamps require 0 <= start < end.")


@dataclass
class Word:
    """ASR word token; text retains leading whitespace and attached punctuation."""

    start: float
    end: float
    text: str

    def __post_init__(self) -> None:
        check_times(self.start, self.end, allow_zero=True)


@dataclass
class Segment:
    id: int
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)
    avg_logprob: float = -1.0
    no_speech_prob: float = 0.0
    compression_ratio: float = 1.0
    review_score: float | None = None
    needs_review: bool = False
    review_reasons: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        check_times(self.start, self.end)


@dataclass
class Caption:
    start: float
    end: float
    text: str

    def __post_init__(self) -> None:
        check_times(self.start, self.end)
        if not self.text.strip():
            raise ValueError("Captions cannot contain empty text.")


@dataclass
class Transcript:
    source: str
    recording_id: str
    duration: float
    segments: list[Segment]
    language: str = "ht"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments).strip()

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, **asdict(self), "text": self.text}
