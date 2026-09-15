from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from html import unescape
from typing import Any, Iterable

from app.utils import utils


TIMING_NATIVE = "native"
TIMING_DERIVED = "derived"
TIMING_ESTIMATED = "estimated"
_VALID_TIMING_SOURCES = {TIMING_NATIVE, TIMING_DERIVED, TIMING_ESTIMATED}


@dataclass(frozen=True)
class NarrationSegment:
    index: int
    start: float
    end: float
    text: str
    timing_source: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["duration"] = self.duration
        return data


@dataclass(frozen=True)
class NarrationAlignmentUnit:
    """Fine-grained provider timing unit used only as a safe cut candidate.

    Edge/Azure native timing normally exposes word-level units. Other providers
    may expose coarser derived units. The media planner treats these as candidate
    boundaries, never as semantic scene boundaries by themselves.
    """

    index: int
    start: float
    end: float
    text: str
    timing_source: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["duration"] = self.duration
        return data


@dataclass(frozen=True)
class NarrationTimeline:
    audio_duration: float
    timing_source: str
    segments: tuple[NarrationSegment, ...]
    alignment_units: tuple[NarrationAlignmentUnit, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "audio_duration": self.audio_duration,
            "timing_source": self.timing_source,
            "segments": [segment.to_dict() for segment in self.segments],
            "alignment_units": [unit.to_dict() for unit in self.alignment_units],
        }


def _safe_seconds(value: Any) -> float | None:
    try:
        if hasattr(value, "total_seconds"):
            value = value.total_seconds()
        seconds = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(seconds) or seconds < 0:
        return None
    return seconds


def _split_script_lines(text: str) -> list[str]:
    normalized = str(text or "").strip()
    if not normalized:
        return []
    lines = utils.split_string_by_punctuations(normalized, keep_punctuation=True)
    cleaned = [line.strip() for line in lines if str(line).strip()]
    return cleaned or [normalized]


def _normalize_for_match(text: str) -> str:
    """Normalize TTS cue text without changing the text shown to the user."""
    normalized = unicodedata.normalize("NFKC", unescape(str(text or ""))).casefold()
    normalized = re.sub(r"[\u064b-\u065f\u0670\u06d6-\u06ed\u0640]", "", normalized)
    for source, target in (
        ("أإآٱ", "ا"),
        ("ىئ", "ي"),
        ("ة", "ه"),
        ("ؤ", "و"),
    ):
        for char in source:
            normalized = normalized.replace(char, target)
    return "".join(char for char in normalized if char.isalnum())


def _matched_script_line(script_lines: list[str], current_text: str, index: int) -> str:
    if index >= len(script_lines):
        return ""
    target = script_lines[index]
    if str(current_text) == target:
        return target
    current_norm = _normalize_for_match(current_text)
    target_norm = _normalize_for_match(target)
    if current_norm and current_norm == target_norm:
        return target
    return ""


def _timeline_from_cues(
    script_lines: list[str],
    cues: Iterable[Any],
    timing_source: str,
) -> list[NarrationSegment]:
    segments: list[NarrationSegment] = []
    current_text = ""
    current_start: float | None = None
    current_end: float | None = None
    script_index = 0

    for cue in cues:
        cue_text = unescape(str(getattr(cue, "content", "") or ""))
        cue_start = _safe_seconds(getattr(cue, "start", None))
        cue_end = _safe_seconds(getattr(cue, "end", None))
        if cue_start is None or cue_end is None or cue_end < cue_start:
            continue
        if current_start is None:
            current_start = cue_start
        current_end = cue_end
        current_text += cue_text

        matched = _matched_script_line(script_lines, current_text, script_index)
        if not matched:
            continue

        segments.append(
            NarrationSegment(
                index=script_index + 1,
                start=current_start,
                end=current_end,
                text=matched,
                timing_source=timing_source,
            )
        )
        script_index += 1
        current_text = ""
        current_start = None
        current_end = None

    if len(segments) != len(script_lines):
        return []
    return segments


def _alignment_units_from_cues(
    cues: Iterable[Any], timing_source: str
) -> list[NarrationAlignmentUnit]:
    units: list[NarrationAlignmentUnit] = []
    for cue in cues:
        text = unescape(str(getattr(cue, "content", "") or "")).strip()
        start = _safe_seconds(getattr(cue, "start", None))
        end = _safe_seconds(getattr(cue, "end", None))
        if not text or start is None or end is None or end < start:
            continue
        units.append(
            NarrationAlignmentUnit(
                index=len(units) + 1,
                start=start,
                end=end,
                text=text,
                timing_source=timing_source,
            )
        )
    return units




def _alignment_units_from_explicit_buffer(
    sub_maker: Any, timing_source: str
) -> list[NarrationAlignmentUnit]:
    """Read native fine timing captured before subtitle aggregation.

    ``voice.py`` stores Edge/Azure word boundaries in a private provider-neutral
    list so this timeline layer does not need to know the TTS SDK event schema.
    """
    raw_units = list(getattr(sub_maker, "_mpt_alignment_units", []) or [])
    units: list[NarrationAlignmentUnit] = []
    for raw in raw_units:
        if not isinstance(raw, dict):
            continue
        text = unescape(str(raw.get("text", "") or "")).strip()
        start = _safe_seconds(raw.get("start"))
        end = _safe_seconds(raw.get("end"))
        if not text or start is None or end is None or end < start:
            continue
        units.append(
            NarrationAlignmentUnit(
                index=len(units) + 1,
                start=start,
                end=end,
                text=text,
                timing_source=timing_source,
            )
        )
    return units

def _timeline_from_legacy_offsets(
    script_lines: list[str],
    sub_maker: Any,
    timing_source: str,
) -> list[NarrationSegment]:
    offsets = list(getattr(sub_maker, "offset", []) or [])
    subs = list(getattr(sub_maker, "subs", []) or [])
    if not offsets or not subs:
        return []

    segments: list[NarrationSegment] = []
    current_text = ""
    current_start_100ns: int | float | None = None
    current_end_100ns: int | float | None = None
    script_index = 0

    for offset, sub in zip(offsets, subs):
        try:
            start_100ns, end_100ns = offset
            start_100ns = float(start_100ns)
            end_100ns = float(end_100ns)
        except (TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(start_100ns) or not math.isfinite(end_100ns):
            continue
        if end_100ns < start_100ns:
            continue

        if current_start_100ns is None:
            current_start_100ns = start_100ns
        current_end_100ns = end_100ns
        current_text += unescape(str(sub or ""))

        matched = _matched_script_line(script_lines, current_text, script_index)
        if not matched:
            continue

        segments.append(
            NarrationSegment(
                index=script_index + 1,
                start=max(0.0, float(current_start_100ns) / 10_000_000.0),
                end=max(0.0, float(current_end_100ns) / 10_000_000.0),
                text=matched,
                timing_source=timing_source,
            )
        )
        script_index += 1
        current_text = ""
        current_start_100ns = None
        current_end_100ns = None

    if len(segments) != len(script_lines):
        return []
    return segments


def _alignment_units_from_legacy_offsets(
    sub_maker: Any, timing_source: str
) -> list[NarrationAlignmentUnit]:
    offsets = list(getattr(sub_maker, "offset", []) or [])
    subs = list(getattr(sub_maker, "subs", []) or [])
    units: list[NarrationAlignmentUnit] = []
    for offset, sub in zip(offsets, subs):
        try:
            start_100ns, end_100ns = offset
            start_100ns = float(start_100ns)
            end_100ns = float(end_100ns)
        except (TypeError, ValueError, OverflowError):
            continue
        if (
            not math.isfinite(start_100ns)
            or not math.isfinite(end_100ns)
            or end_100ns < start_100ns
        ):
            continue
        text = unescape(str(sub or "")).strip()
        if not text:
            continue
        units.append(
            NarrationAlignmentUnit(
                index=len(units) + 1,
                start=max(0.0, start_100ns / 10_000_000.0),
                end=max(0.0, end_100ns / 10_000_000.0),
                text=text,
                timing_source=timing_source,
            )
        )
    return units


def _estimated_timeline(
    script_lines: list[str],
    audio_duration: float,
    timing_source: str,
) -> list[NarrationSegment]:
    if not script_lines:
        return []
    weights = [max(len(_normalize_for_match(line)), 1) for line in script_lines]
    total_weight = sum(weights)
    if total_weight <= 0:
        weights = [1 for _ in script_lines]
        total_weight = len(weights)

    segments: list[NarrationSegment] = []
    start = 0.0
    for index, (line, weight) in enumerate(zip(script_lines, weights), start=1):
        if index == len(script_lines):
            end = audio_duration
        else:
            end = min(audio_duration, start + audio_duration * (weight / total_weight))
        end = max(start, end)
        segments.append(
            NarrationSegment(
                index=index,
                start=start,
                end=end,
                text=line,
                timing_source=timing_source,
            )
        )
        start = end
    return segments


def build_narration_timeline(
    *,
    script: str,
    audio_duration: float,
    sub_maker: Any | None = None,
    provider_timing_source: str = TIMING_DERIVED,
) -> NarrationTimeline:
    """Build a provider-neutral narration timeline.

    The primary ``segments`` remain sentence/clause-level narration units. When
    the active TTS exposes finer boundaries, ``alignment_units`` preserves those
    provider timings separately so later media planning can choose safe cut points
    without turning every word into a semantic scene.
    """
    duration = _safe_seconds(audio_duration)
    if duration is None or duration <= 0:
        raise ValueError("audio_duration must be a positive finite number")

    script_lines = _split_script_lines(script)
    if not script_lines:
        raise ValueError("script must contain narration text")

    source = str(provider_timing_source or TIMING_DERIVED).strip().lower()
    if source not in _VALID_TIMING_SOURCES:
        source = TIMING_DERIVED

    explicit_alignment: list[NarrationAlignmentUnit] = []
    if sub_maker is not None:
        explicit_alignment = _alignment_units_from_explicit_buffer(sub_maker, source)

    if sub_maker is not None:
        cues = list(getattr(sub_maker, "cues", []) or [])
        if cues:
            segments = _timeline_from_cues(script_lines, cues, source)
            if segments:
                return NarrationTimeline(
                    audio_duration=duration,
                    timing_source=source,
                    segments=tuple(segments),
                    alignment_units=tuple(
                        explicit_alignment or _alignment_units_from_cues(cues, source)
                    ),
                )

        segments = _timeline_from_legacy_offsets(script_lines, sub_maker, source)
        if segments:
            return NarrationTimeline(
                audio_duration=duration,
                timing_source=source,
                segments=tuple(segments),
                alignment_units=tuple(
                    explicit_alignment
                    or _alignment_units_from_legacy_offsets(sub_maker, source)
                ),
            )

    segments = _estimated_timeline(script_lines, duration, TIMING_ESTIMATED)
    return NarrationTimeline(
        audio_duration=duration,
        timing_source=TIMING_ESTIMATED,
        segments=tuple(segments),
        alignment_units=(),
    )


def provider_timing_source(tts_server: str | None) -> str:
    """Return the timing quality currently exposed by MoneyPrinterTurbo providers."""
    server = str(tts_server or "").strip().lower()
    if server in {"azure-tts-v1", "azure-tts-v2"}:
        return TIMING_NATIVE
    return TIMING_DERIVED
