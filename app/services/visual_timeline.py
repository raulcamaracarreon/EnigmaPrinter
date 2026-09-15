from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable


@dataclass(frozen=True)
class NarrationSpan:
    index: int
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["duration"] = self.duration
        return data


@dataclass(frozen=True)
class SemanticScene:
    index: int
    start: float
    end: float
    narration_text: str
    narration_indices: tuple[int, ...]
    narration_spans: tuple[NarrationSpan, ...]
    alignment_spans: tuple[NarrationSpan, ...] = ()

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def narration_boundaries(self) -> tuple[float, ...]:
        """Safe internal boundaries where a narration unit begins."""
        if len(self.narration_spans) <= 1:
            return ()
        boundaries: list[float] = []
        for span in self.narration_spans[1:]:
            boundary = float(span.start)
            if self.start < boundary < self.end:
                boundaries.append(boundary)
        return tuple(boundaries)

    @property
    def alignment_boundaries(self) -> tuple[float, ...]:
        """Finer provider-aligned cut candidates, usually word boundaries."""
        if len(self.alignment_spans) <= 1:
            return ()
        boundaries: list[float] = []
        for span in self.alignment_spans[1:]:
            boundary = float(span.start)
            if self.start < boundary < self.end:
                boundaries.append(boundary)
        return tuple(boundaries)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "narration_text": self.narration_text,
            "narration_indices": list(self.narration_indices),
            "narration_boundaries": list(self.narration_boundaries),
            "alignment_boundaries": list(self.alignment_boundaries),
            "narration_spans": [span.to_dict() for span in self.narration_spans],
            "alignment_spans": [span.to_dict() for span in self.alignment_spans],
        }


@dataclass(frozen=True)
class VisualTimeline:
    audio_duration: float
    timing_source: str
    segmentation_strategy: str
    segments: tuple[SemanticScene, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "audio_duration": self.audio_duration,
            "timing_source": self.timing_source,
            "segmentation_strategy": self.segmentation_strategy,
            "segments": [segment.to_dict() for segment in self.segments],
        }


_ONLY_CLOSING_PUNCTUATION_RE = re.compile(r"^[\]\)}»”’\"'.,;:!?…]+$")
_OPENING_CHARS = "([{«“‘\"'"
_WEAK_ENDINGS = (",", ";", ":", "—", "–", "-")
_STRONG_ENDINGS = (".", "?", "!", "…")
_CLOSING_CHARS = ")] }»”’\"'".replace(" ", "")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(number):
        return default
    return number


def _smart_join(parts: Iterable[str]) -> str:
    result = ""
    for raw in parts:
        piece = str(raw or "").strip()
        if not piece:
            continue
        if not result:
            result = piece
            continue
        if _ONLY_CLOSING_PUNCTUATION_RE.fullmatch(piece) or piece[0] in ",.;:!?…)]}»”’":
            result += piece
        elif result[-1] in _OPENING_CHARS:
            result += piece
        else:
            result += " " + piece

    result = re.sub(r"\s+([,.;:!?…\]\)}»”’])", r"\1", result)
    return result.strip()


def _strip_closing_chars(text: str) -> str:
    cleaned = str(text or "").rstrip()
    while cleaned and cleaned[-1] in _CLOSING_CHARS:
        cleaned = cleaned[:-1].rstrip()
    return cleaned


def _has_weak_ending(text: str) -> bool:
    cleaned = _strip_closing_chars(text)
    return bool(cleaned) and cleaned.endswith(_WEAK_ENDINGS)


def _has_strong_ending(text: str) -> bool:
    cleaned = _strip_closing_chars(text)
    if not cleaned:
        return False
    if cleaned.endswith(_WEAK_ENDINGS):
        return False
    if cleaned.endswith(_STRONG_ENDINGS):
        return True
    return True


def _coerce_spans(narration_segments: Iterable[Any]) -> list[NarrationSpan]:
    spans: list[NarrationSpan] = []
    pending_prefix: list[tuple[int, float, float, str]] = []

    for position, raw in enumerate(narration_segments, start=1):
        if hasattr(raw, "to_dict"):
            raw = raw.to_dict()
        if not isinstance(raw, dict):
            continue

        text = str(raw.get("text", "") or "").strip()
        if not text:
            continue
        start = max(0.0, _safe_float(raw.get("start"), 0.0))
        end = max(start, _safe_float(raw.get("end"), start))
        try:
            index = int(raw.get("index", position))
        except (TypeError, ValueError, OverflowError):
            index = position

        if not any(char.isalnum() for char in text):
            if spans:
                previous = spans[-1]
                spans[-1] = replace(
                    previous,
                    end=max(previous.end, end),
                    text=_smart_join((previous.text, text)),
                )
            else:
                pending_prefix.append((index, start, end, text))
            continue

        if pending_prefix:
            prefix_text = [item[3] for item in pending_prefix]
            start = min(start, min(item[1] for item in pending_prefix))
            text = _smart_join((*prefix_text, text))
            pending_prefix.clear()

        spans.append(NarrationSpan(index=index, start=start, end=end, text=text))

    if pending_prefix and spans:
        previous = spans[-1]
        spans[-1] = replace(
            previous,
            end=max(previous.end, max(item[2] for item in pending_prefix)),
            text=_smart_join((previous.text, *(item[3] for item in pending_prefix))),
        )
    return spans


def _group_semantic_spans(spans: list[NarrationSpan]) -> list[tuple[int, int]]:
    """Group weakly connected clauses into scene candidates.

    This layer deliberately ignores Clip Duration. Duration limits belong to the
    media-shot planner, so changing a generation ceiling cannot rewrite semantic
    story boundaries.
    """
    if not spans:
        return []

    groups: list[tuple[int, int]] = []
    group_start = 0
    for index, span in enumerate(spans):
        if _has_weak_ending(span.text):
            continue
        if _has_strong_ending(span.text):
            groups.append((group_start, index))
            group_start = index + 1

    if group_start < len(spans):
        groups.append((group_start, len(spans) - 1))
    return groups


def _scene_window(
    spans: list[NarrationSpan],
    start_index: int,
    end_index: int,
    audio_duration: float,
) -> tuple[float, float]:
    start = 0.0 if start_index == 0 else spans[start_index].start
    if end_index >= len(spans) - 1:
        end = audio_duration
    else:
        end = spans[end_index + 1].start
    return max(0.0, start), max(start, min(end, audio_duration))


def _spans_overlapping_window(
    spans: list[NarrationSpan], start: float, end: float
) -> tuple[NarrationSpan, ...]:
    return tuple(
        span
        for span in spans
        if span.end > start + 1e-9 and span.start < end - 1e-9
    )


def build_visual_timeline(narration_timeline: Any) -> VisualTimeline:
    """Build semantic scenes and preserve fine alignment as optional cut data."""
    if hasattr(narration_timeline, "to_dict"):
        narration_timeline = narration_timeline.to_dict()
    if not isinstance(narration_timeline, dict):
        raise ValueError("narration_timeline must be a mapping or expose to_dict()")

    audio_duration = _safe_float(narration_timeline.get("audio_duration"), 0.0)
    if audio_duration <= 0:
        raise ValueError("narration timeline audio_duration must be positive")

    spans = _coerce_spans(narration_timeline.get("segments", []) or [])
    if not spans:
        raise ValueError("narration timeline must contain at least one usable segment")
    alignment_spans = _coerce_spans(narration_timeline.get("alignment_units", []) or [])

    timing_source = str(narration_timeline.get("timing_source") or "estimated")
    groups = _group_semantic_spans(spans)

    scenes: list[SemanticScene] = []
    for scene_index, (start_index, end_index) in enumerate(groups, start=1):
        start, end = _scene_window(spans, start_index, end_index, audio_duration)
        source_spans = tuple(spans[start_index : end_index + 1])
        scenes.append(
            SemanticScene(
                index=scene_index,
                start=start,
                end=end,
                narration_text=_smart_join(span.text for span in source_spans),
                narration_indices=tuple(span.index for span in source_spans),
                narration_spans=source_spans,
                alignment_spans=_spans_overlapping_window(alignment_spans, start, end),
            )
        )

    return VisualTimeline(
        audio_duration=audio_duration,
        timing_source=timing_source,
        segmentation_strategy="narration_semantic_boundaries_v2",
        segments=tuple(scenes),
    )
