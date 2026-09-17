from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from uuid import uuid4

from app.services import narration_timeline, subtitle, voice
from app.utils import utils


class NarrationAlignmentError(ValueError):
    """Raised when uploaded narration cannot be aligned safely to its source script."""


@dataclass(frozen=True)
class RecognizedWord:
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class _ScriptToken:
    text: str
    normalized: str
    segment_index: int


@dataclass(frozen=True)
class ExternalNarrationAlignment:
    timeline: narration_timeline.NarrationTimeline
    script_word_count: int
    recognized_word_count: int
    matched_word_count: int
    script_coverage: float
    recognized_coverage: float

    def to_dict(self) -> dict:
        data = asdict(self)
        data["timeline"] = self.timeline.to_dict()
        return data


_WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)


def _normalize_word(text: str) -> str:
    value = unicodedata.normalize("NFKD", str(text or "")).casefold()
    value = "".join(char for char in value if not unicodedata.combining(char))
    return "".join(char for char in value if char.isalnum())


def _script_segments(script: str) -> list[str]:
    normalized = str(script or "").strip()
    if not normalized:
        return []
    segments = utils.split_string_by_punctuations(normalized, keep_punctuation=True)
    return [str(segment).strip() for segment in segments if str(segment).strip()]


def _script_tokens(segments: list[str]) -> list[_ScriptToken]:
    tokens: list[_ScriptToken] = []
    for segment_index, segment in enumerate(segments):
        for match in _WORD_RE.finditer(segment):
            text = match.group(0)
            normalized = _normalize_word(text)
            if normalized:
                tokens.append(
                    _ScriptToken(
                        text=text,
                        normalized=normalized,
                        segment_index=segment_index,
                    )
                )
    return tokens


def _parse_srt_seconds(value: str) -> float:
    raw = str(value or "").strip().replace(".", ",")
    try:
        hours_text, minutes_text, rest = raw.split(":", 2)
        seconds_text, millis_text = rest.split(",", 1)
        seconds = (
            int(hours_text) * 3600
            + int(minutes_text) * 60
            + int(seconds_text)
            + int(millis_text[:3].ljust(3, "0")) / 1000.0
        )
    except (TypeError, ValueError):
        raise NarrationAlignmentError(f"invalid Whisper timestamp: {value!r}") from None
    if not math.isfinite(seconds) or seconds < 0:
        raise NarrationAlignmentError(f"invalid Whisper timestamp: {value!r}")
    return seconds


def _recognized_words_from_srt(subtitle_file: str) -> list[RecognizedWord]:
    words: list[RecognizedWord] = []
    for _, times, text in subtitle.file_to_subtitles(subtitle_file):
        try:
            start_text, end_text = str(times).split(" --> ", 1)
        except ValueError:
            continue
        start = _parse_srt_seconds(start_text)
        end = _parse_srt_seconds(end_text)
        cleaned = str(text or "").strip()
        if not cleaned or end < start:
            continue
        words.append(RecognizedWord(text=cleaned, start=start, end=end))
    return words


def transcribe_alignment_words(audio_file: str) -> list[RecognizedWord]:
    """Transcribe an uploaded narration with the project's existing Whisper model.

    The subtitle service already owns the lazy ``faster-whisper`` model and its VAD
    configuration. Reusing its word-level SRT path avoids loading a second Whisper
    model into memory solely for narration alignment.
    """
    if not audio_file or not Path(audio_file).is_file():
        raise NarrationAlignmentError("uploaded narration audio file does not exist")

    temp_dir = Path(utils.storage_dir("temp", create=True))
    subtitle_file = temp_dir / f"narration-alignment-{uuid4().hex}.srt"
    try:
        result = subtitle.create(
            audio_file=audio_file,
            subtitle_file=str(subtitle_file),
            word_level=True,
        )
        if result == "" or not subtitle_file.is_file():
            raise NarrationAlignmentError(
                "Whisper is unavailable or did not produce word timestamps"
            )
        words = _recognized_words_from_srt(str(subtitle_file))
        if not words:
            raise NarrationAlignmentError(
                "Whisper did not recognize any timestamped narration words"
            )
        return words
    finally:
        try:
            subtitle_file.unlink(missing_ok=True)
        except OSError:
            pass


def _word_similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right, autojunk=False).ratio()


def _pair_score(left: str, right: str) -> float:
    similarity = _word_similarity(left, right)
    if similarity >= 0.999:
        return 3.0
    if similarity >= 0.84:
        return 2.0
    if similarity >= 0.68:
        return 0.75
    return -2.0


def _align_token_indices(
    script_tokens: list[_ScriptToken],
    recognized_words: list[RecognizedWord],
) -> list[tuple[int, int]]:
    """Globally align script tokens to recognized words while preserving order."""
    observed = [_normalize_word(word.text) for word in recognized_words]
    script = [token.normalized for token in script_tokens]
    n = len(script)
    m = len(observed)
    gap_penalty = -1.25

    scores = [[0.0] * (m + 1) for _ in range(n + 1)]
    moves = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        scores[i][0] = scores[i - 1][0] + gap_penalty
        moves[i][0] = "up"
    for j in range(1, m + 1):
        scores[0][j] = scores[0][j - 1] + gap_penalty
        moves[0][j] = "left"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diagonal = scores[i - 1][j - 1] + _pair_score(script[i - 1], observed[j - 1])
            up = scores[i - 1][j] + gap_penalty
            left = scores[i][j - 1] + gap_penalty
            best = max(diagonal, up, left)
            scores[i][j] = best
            if math.isclose(best, diagonal):
                moves[i][j] = "diag"
            elif math.isclose(best, up):
                moves[i][j] = "up"
            else:
                moves[i][j] = "left"

    pairs: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        move = moves[i][j]
        if move == "diag" and i > 0 and j > 0:
            similarity = _word_similarity(script[i - 1], observed[j - 1])
            if similarity >= 0.68:
                pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif move == "up" and i > 0:
            i -= 1
        elif j > 0:
            j -= 1
        else:
            i -= 1

    pairs.reverse()
    return pairs


def _build_timeline(
    *,
    segments: list[str],
    script_tokens: list[_ScriptToken],
    recognized_words: list[RecognizedWord],
    pairs: list[tuple[int, int]],
    audio_duration: float,
) -> narration_timeline.NarrationTimeline:
    pair_by_script = {script_index: word_index for script_index, word_index in pairs}
    segment_matches: list[list[tuple[_ScriptToken, RecognizedWord]]] = [
        [] for _ in segments
    ]
    segment_token_counts = [0 for _ in segments]
    for token in script_tokens:
        segment_token_counts[token.segment_index] += 1
    alignment_units: list[narration_timeline.NarrationAlignmentUnit] = []

    for script_index, token in enumerate(script_tokens):
        word_index = pair_by_script.get(script_index)
        if word_index is None:
            continue
        word = recognized_words[word_index]
        segment_matches[token.segment_index].append((token, word))
        alignment_units.append(
            narration_timeline.NarrationAlignmentUnit(
                index=len(alignment_units) + 1,
                start=max(0.0, min(float(word.start), audio_duration)),
                end=max(0.0, min(float(word.end), audio_duration)),
                text=token.text,
                timing_source=narration_timeline.TIMING_DERIVED,
            )
        )

    narration_segments: list[narration_timeline.NarrationSegment] = []
    for segment_index, (segment_text, matches) in enumerate(
        zip(segments, segment_matches), start=1
    ):
        token_count = segment_token_counts[segment_index - 1]
        segment_coverage = len(matches) / token_count if token_count else 0.0
        if not matches or segment_coverage < 0.60:
            raise NarrationAlignmentError(
                f"script segment {segment_index} could not be aligned reliably "
                f"(segment coverage {segment_coverage:.0%}, required 60%)"
            )
        start = max(0.0, min(matches[0][1].start, audio_duration))
        end = max(start, min(matches[-1][1].end, audio_duration))
        narration_segments.append(
            narration_timeline.NarrationSegment(
                index=segment_index,
                start=start,
                end=end,
                text=segment_text,
                timing_source=narration_timeline.TIMING_DERIVED,
            )
        )

    return narration_timeline.NarrationTimeline(
        audio_duration=audio_duration,
        timing_source=narration_timeline.TIMING_DERIVED,
        segments=tuple(narration_segments),
        alignment_units=tuple(alignment_units),
    )


def align_external_narration(
    *,
    audio_file: str,
    script: str,
    audio_duration: float | None = None,
    recognized_words: list[RecognizedWord] | None = None,
    minimum_script_coverage: float = 0.75,
    minimum_recognized_coverage: float = 0.60,
) -> ExternalNarrationAlignment:
    """Strictly align uploaded narration audio to the user-provided source script.

    The script remains the textual authority. Whisper supplies only timing evidence.
    Unlike ``build_narration_timeline`` this path never falls back to proportional
    estimated timing: materially mismatched audio fails explicitly before an Active
    Shot Plan can appear valid.
    """
    segments = _script_segments(script)
    if not segments:
        raise NarrationAlignmentError("script must contain narration text")
    script_tokens = _script_tokens(segments)
    if not script_tokens:
        raise NarrationAlignmentError("script does not contain alignable words")

    duration = audio_duration
    if duration is None:
        duration = voice.get_audio_duration(audio_file)
    try:
        duration = float(duration)
    except (TypeError, ValueError, OverflowError):
        raise NarrationAlignmentError("uploaded narration duration is unavailable") from None
    if not math.isfinite(duration) or duration <= 0:
        raise NarrationAlignmentError("uploaded narration duration is unavailable")

    words = list(recognized_words or transcribe_alignment_words(audio_file))
    words = [
        word
        for word in words
        if _normalize_word(word.text)
        and math.isfinite(float(word.start))
        and math.isfinite(float(word.end))
        and 0 <= float(word.start) <= float(word.end)
    ]
    if not words:
        raise NarrationAlignmentError("uploaded narration contains no alignable words")

    pairs = _align_token_indices(script_tokens, words)
    matched = len(pairs)
    script_coverage = matched / len(script_tokens)
    recognized_coverage = matched / len(words)

    if script_coverage < float(minimum_script_coverage):
        raise NarrationAlignmentError(
            "uploaded narration does not match the script closely enough "
            f"(script coverage {script_coverage:.0%}, required {minimum_script_coverage:.0%})"
        )
    if recognized_coverage < float(minimum_recognized_coverage):
        raise NarrationAlignmentError(
            "uploaded narration contains too much speech outside the script "
            f"(recognized coverage {recognized_coverage:.0%}, required {minimum_recognized_coverage:.0%})"
        )

    timeline = _build_timeline(
        segments=segments,
        script_tokens=script_tokens,
        recognized_words=words,
        pairs=pairs,
        audio_duration=duration,
    )
    return ExternalNarrationAlignment(
        timeline=timeline,
        script_word_count=len(script_tokens),
        recognized_word_count=len(words),
        matched_word_count=matched,
        script_coverage=script_coverage,
        recognized_coverage=recognized_coverage,
    )
