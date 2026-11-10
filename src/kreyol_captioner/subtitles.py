"""Caption segmentation and UTF-8 subtitle export."""

import math
import textwrap
from itertools import pairwise
from pathlib import Path

from .models import Caption, Segment


def wrap_text(text: str, width: int = 42) -> str:
    return "\n".join(textwrap.wrap(
        " ".join(text.split()), width=width,
        break_long_words=False, break_on_hyphens=False,
    ))


def make_captions(
    segments: list[Segment], *, line_width: int = 42, max_duration: float = 6.0,
) -> list[Caption]:
    """Split on word boundaries; never fabricate word timestamps.

    A segment without word timings stays intact. Extremely long individual
    words, pauses, or slow words can exceed the preferred layout limits.
    """
    if line_width < 1 or max_duration <= 0:
        raise ValueError("Caption width and duration must be positive.")
    captions: list[Caption] = []
    for segment in segments:
        words = [w for w in segment.words if w.text.strip()]
        if not words:
            if segment.text.strip():
                captions.append(Caption(segment.start, segment.end,
                                        wrap_text(segment.text, line_width)))
            continue
        group = []

        def flush(group=group, segment=segment) -> None:
            if group:
                start = max(segment.start, min(group[0].start, segment.end - 0.001))
                end = min(segment.end, max(w.end for w in group))
                if end <= start:
                    # Whisper occasionally emits a zero-duration word.
                    end = min(segment.end, start + 0.02)
                if end > start:
                    captions.append(Caption(start, end, wrap_text(
                        "".join(w.text for w in group).strip(), line_width)))
                else:
                    raise ValueError("Speech segment is too short to export caption timing.")
                group.clear()

        for word in words:
            candidate = "".join(w.text for w in [*group, word]).strip()
            if group and (
                len(wrap_text(candidate, line_width).splitlines()) > 2
                or word.end - group[0].start > max_duration
                or word.start - group[-1].end > 0.8
            ):
                flush()
            group.append(word)
            if word.text.rstrip().endswith((".", "!", "?")) and word.end - group[0].start >= 1:
                flush()
        flush()

    # Small timestamp overlaps can occur in ASR output. Resolve them without
    # shifting every later caption, which would desynchronize the video.
    for previous, current in pairwise(captions):
        if current.start < previous.end:
            if current.start > previous.start:
                previous.end = current.start
            else:
                raise ValueError("ASR emitted captions with incompatible timestamps.")
    return captions


def timestamp(seconds: float, *, vtt: bool = False) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Subtitle timestamps must be finite and nonnegative.")
    milliseconds = round(seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    whole_seconds, milliseconds = divmod(milliseconds, 1000)
    separator = "." if vtt else ","
    return f"{hours:02}:{minutes:02}:{whole_seconds:02}{separator}{milliseconds:03}"


def validate_captions(captions: list[Caption], duration: float | None = None) -> None:
    previous_end = 0.0
    for caption in captions:
        # Revalidate mutable records and ensure timestamps survive millisecond export.
        caption.__post_init__()
        if round(caption.end * 1000) <= round(caption.start * 1000):
            raise ValueError("Caption duration must be at least one millisecond.")
        if caption.start < previous_end - 1e-6:
            raise ValueError("Captions must be in order and must not overlap.")
        if duration is not None and caption.end > duration + 0.001:
            raise ValueError("A caption extends past the end of the recording.")
        previous_end = caption.end


def subtitle_text(captions: list[Caption], *, vtt: bool = False) -> str:
    validate_captions(captions)
    blocks = []
    for index, caption in enumerate(captions, 1):
        timing = f"{timestamp(caption.start, vtt=vtt)} --> {timestamp(caption.end, vtt=vtt)}"
        # Prevent edited caption text from introducing extra subtitle cues or VTT markup.
        text = caption.text.replace("\r", "")
        text = "\n".join(line for line in text.splitlines() if line.strip())
        if vtt:
            text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        blocks.append(f"{index}\n{timing}\n{text}")
    prefix = "WEBVTT\n\n" if vtt else ""
    return prefix + "\n\n".join(blocks) + ("\n" if blocks else "")


def write_subtitles(captions: list[Caption], directory: Path, stem: str = "captions") -> tuple[Path, Path]:
    srt, vtt = directory / f"{stem}.srt", directory / f"{stem}.vtt"
    srt.write_text(subtitle_text(captions), encoding="utf-8")
    vtt.write_text(subtitle_text(captions, vtt=True), encoding="utf-8")
    return srt, vtt
