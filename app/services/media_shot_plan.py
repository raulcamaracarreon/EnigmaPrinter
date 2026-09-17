from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable


BOUNDARY_SCENE = "scene_boundary"
BOUNDARY_NARRATION = "narration_boundary"
BOUNDARY_ALIGNED = BOUNDARY_NARRATION  # backward-compatible alias
BOUNDARY_ALIGNMENT = "alignment_boundary"
BOUNDARY_BALANCED = "balanced_internal_cut"

DEFAULT_MINIMUM_SHOT_DURATION = 3.0

_ONLY_CLOSING_PUNCTUATION_RE = re.compile(r"^[\]\)}»”’\"'.,;:!?…]+$")
_OPENING_CHARS = "([{«“‘\"'"
_SOURCE_PRIORITY = {
    BOUNDARY_SCENE: 0,
    BOUNDARY_NARRATION: 1,
    BOUNDARY_ALIGNMENT: 2,
    BOUNDARY_BALANCED: 3,
}
_STRONG_SENTENCE_END_RE = re.compile(
    r'[.?!…][\]\)}»”’"\']*\s*$'
)


@dataclass(frozen=True)
class MediaShot:
    index: int
    scene_index: int
    scene_indices: tuple[int, ...]
    start: float
    end: float
    narration_text: str
    narration_indices: tuple[int, ...]
    alignment_indices: tuple[int, ...]
    start_boundary_source: str
    end_boundary_source: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def uses_internal_estimate(self) -> bool:
        return (
            self.start_boundary_source == BOUNDARY_BALANCED
            or self.end_boundary_source == BOUNDARY_BALANCED
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["duration"] = self.duration
        data["uses_internal_estimate"] = self.uses_internal_estimate
        data["scene_indices"] = list(self.scene_indices)
        data["narration_indices"] = list(self.narration_indices)
        data["alignment_indices"] = list(self.alignment_indices)
        return data


@dataclass(frozen=True)
class MediaShotPlan:
    audio_duration: float
    max_clip_duration: float
    min_clip_duration: float
    timing_source: str
    shots: tuple[MediaShot, ...]

    @property
    def undersized_shot_count(self) -> int:
        return sum(
            1 for shot in self.shots if shot.duration < self.min_clip_duration - 1e-6
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "audio_duration": self.audio_duration,
            "max_clip_duration": self.max_clip_duration,
            "min_clip_duration": self.min_clip_duration,
            "timing_source": self.timing_source,
            "undersized_shot_count": self.undersized_shot_count,
            "shots": [shot.to_dict() for shot in self.shots],
        }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(number):
        return default
    return number


def _as_mapping(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if not isinstance(value, dict):
        raise ValueError("visual_timeline must be a mapping or expose to_dict()")
    return value


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


def _unique_sorted(values: Iterable[Any], *, start: float, end: float) -> list[float]:
    result: list[float] = []
    for value in values:
        number = _safe_float(value, math.nan)
        if not math.isfinite(number) or not (start < number < end):
            continue
        result.append(number)
    return sorted({round(value, 9) for value in result})


def _nearest_valid_boundary(
    values: list[float], *, lower: float, upper: float, desired: float
) -> float | None:
    valid = [boundary for boundary in values if lower <= boundary <= upper]
    if not valid:
        return None
    return min(valid, key=lambda boundary: abs(boundary - desired))


def _scene_cuts(
    *,
    start: float,
    end: float,
    narration_boundaries: list[float],
    alignment_boundaries: list[float],
    max_clip_duration: float,
) -> tuple[list[float], dict[float, str]]:
    duration = max(0.0, end - start)
    if duration <= max_clip_duration + 1e-9:
        return [start, end], {start: BOUNDARY_SCENE, end: BOUNDARY_SCENE}

    shot_count = max(1, int(math.ceil(duration / max_clip_duration)))
    even_duration = duration / shot_count
    cuts = [start]
    cut_sources: dict[float, str] = {start: BOUNDARY_SCENE, end: BOUNDARY_SCENE}
    previous = start

    for cut_number in range(1, shot_count):
        remaining_shots = shot_count - cut_number
        desired = start + even_duration * cut_number
        lower = max(previous + 1e-6, end - max_clip_duration * remaining_shots)
        upper = min(previous + max_clip_duration, end - 1e-6)

        chosen = _nearest_valid_boundary(
            narration_boundaries, lower=lower, upper=upper, desired=desired
        )
        if chosen is not None:
            source = BOUNDARY_NARRATION
        else:
            chosen = _nearest_valid_boundary(
                alignment_boundaries, lower=lower, upper=upper, desired=desired
            )
            if chosen is not None:
                source = BOUNDARY_ALIGNMENT
            else:
                if lower > upper:
                    chosen = min(max(desired, previous + 1e-6), end - 1e-6)
                else:
                    chosen = min(max(desired, lower), upper)
                source = BOUNDARY_BALANCED

        cuts.append(chosen)
        cut_sources[chosen] = source
        previous = chosen

    cuts.append(end)
    return cuts, cut_sources


def _preferred_scene_cuts(
    *,
    start: float,
    end: float,
    preferred: Iterable[Any],
    narration_boundaries: list[float],
    alignment_boundaries: list[float],
    max_clip_duration: float,
) -> tuple[list[float], dict[float, str]]:
    cuts = [start, *_unique_sorted(preferred, start=start, end=end), end]
    for left, right in zip(cuts, cuts[1:]):
        if right - left > max_clip_duration + 1e-6:
            raise ValueError("preferred media cuts would exceed max_clip_duration")

    sources: dict[float, str] = {start: BOUNDARY_SCENE, end: BOUNDARY_SCENE}
    for cut in cuts[1:-1]:
        if any(math.isclose(cut, value, abs_tol=1e-6) for value in narration_boundaries):
            sources[cut] = BOUNDARY_NARRATION
        elif any(math.isclose(cut, value, abs_tol=1e-6) for value in alignment_boundaries):
            sources[cut] = BOUNDARY_ALIGNMENT
        else:
            raise ValueError("preferred media cut is not a validated narration/alignment boundary")
    return cuts, sources


def _overlapping_spans(
    spans: list[Any], start: float, end: float
) -> tuple[str, tuple[int, ...]]:
    texts: list[str] = []
    indices: list[int] = []
    for raw in spans:
        if not isinstance(raw, dict):
            continue
        span_start = _safe_float(raw.get("start"), start)
        span_end = _safe_float(raw.get("end"), span_start)
        if span_end > start + 1e-9 and span_start < end - 1e-9:
            text = str(raw.get("text", "") or "").strip()
            if text:
                texts.append(text)
            try:
                index = int(raw.get("index"))
            except (TypeError, ValueError, OverflowError):
                continue
            if index not in indices:
                indices.append(index)
    return _smart_join(texts), tuple(indices)


def _shot_text_and_indices(
    scene: dict[str, Any], start: float, end: float
) -> tuple[str, tuple[int, ...], tuple[int, ...]]:
    narration_text, narration_indices = _overlapping_spans(
        list(scene.get("narration_spans", []) or []), start, end
    )
    alignment_text, alignment_indices = _overlapping_spans(
        list(scene.get("alignment_spans", []) or []), start, end
    )
    text = narration_text or alignment_text
    if not text:
        text = str(scene.get("narration_text", "") or "").strip()
    return text, narration_indices, alignment_indices


def _time_key(value: float) -> float:
    return round(float(value), 9)


def _set_boundary_source(source_map: dict[float, str], value: float, source: str) -> None:
    key = _time_key(value)
    current = source_map.get(key)
    if current is None or _SOURCE_PRIORITY.get(source, 99) < _SOURCE_PRIORITY.get(current, 99):
        source_map[key] = source


def _scene_indices_for_window(
    scenes: list[dict[str, Any]], start: float, end: float
) -> tuple[int, ...]:
    result: list[int] = []
    for position, scene in enumerate(scenes, start=1):
        scene_start = _safe_float(scene.get("start"), 0.0)
        scene_end = _safe_float(scene.get("end"), scene_start)
        if scene_end <= start + 1e-9 or scene_start >= end - 1e-9:
            continue
        try:
            scene_index = int(scene.get("index", position))
        except (TypeError, ValueError, OverflowError):
            scene_index = position
        if scene_index not in result:
            result.append(scene_index)
    return tuple(result)


def _window_text_and_indices(
    scenes: list[dict[str, Any]], start: float, end: float
) -> tuple[str, tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    text_parts: list[str] = []
    narration_indices: list[int] = []
    alignment_indices: list[int] = []
    scene_indices: list[int] = []

    for position, scene in enumerate(scenes, start=1):
        scene_start = _safe_float(scene.get("start"), 0.0)
        scene_end = _safe_float(scene.get("end"), scene_start)
        if scene_end <= start + 1e-9 or scene_start >= end - 1e-9:
            continue
        try:
            scene_index = int(scene.get("index", position))
        except (TypeError, ValueError, OverflowError):
            scene_index = position
        if scene_index not in scene_indices:
            scene_indices.append(scene_index)

        text, narration, alignment = _shot_text_and_indices(scene, start, end)
        if text:
            text_parts.append(text)
        for index in narration:
            if index not in narration_indices:
                narration_indices.append(index)
        for index in alignment:
            if index not in alignment_indices:
                alignment_indices.append(index)

    return (
        _smart_join(text_parts),
        tuple(narration_indices),
        tuple(alignment_indices),
        tuple(scene_indices),
    )


def _collect_boundary_catalog(
    scenes: list[dict[str, Any]],
    base_shots: list[MediaShot],
    *,
    audio_duration: float,
) -> tuple[list[float], dict[float, str], set[float]]:
    sources: dict[float, str] = {}
    scene_boundaries: set[float] = set()

    _set_boundary_source(sources, 0.0, BOUNDARY_SCENE)
    _set_boundary_source(sources, audio_duration, BOUNDARY_SCENE)
    scene_boundaries.update({_time_key(0.0), _time_key(audio_duration)})

    for scene in scenes:
        start = max(0.0, _safe_float(scene.get("start"), 0.0))
        end = min(audio_duration, max(start, _safe_float(scene.get("end"), start)))
        _set_boundary_source(sources, start, BOUNDARY_SCENE)
        _set_boundary_source(sources, end, BOUNDARY_SCENE)
        scene_boundaries.update({_time_key(start), _time_key(end)})
        for value in scene.get("narration_boundaries", []) or []:
            _set_boundary_source(sources, _safe_float(value, start), BOUNDARY_NARRATION)
        for value in scene.get("alignment_boundaries", []) or []:
            _set_boundary_source(sources, _safe_float(value, start), BOUNDARY_ALIGNMENT)

    # Base cuts include balanced fallbacks, guaranteeing that a hard-max path
    # remains available even when a provider exposes only coarse timing.
    for shot in base_shots:
        _set_boundary_source(sources, shot.start, shot.start_boundary_source)
        _set_boundary_source(sources, shot.end, shot.end_boundary_source)

    points = sorted(
        key for key in sources if -1e-9 <= key <= audio_duration + 1e-9
    )
    return points, sources, scene_boundaries


def _edge_crossed_scene_count(
    scene_boundaries: set[float], start: float, end: float
) -> int:
    return sum(1 for value in scene_boundaries if start + 1e-9 < value < end - 1e-9)


def _crosses_locked_boundary(
    locked: tuple[float, ...], start: float, end: float
) -> bool:
    return any(start + 1e-9 < value < end - 1e-9 for value in locked)


def _normalize_minimum_duration(
    timeline: dict[str, Any],
    base_shots: list[MediaShot],
    *,
    audio_duration: float,
    minimum: float,
    maximum: float,
    locked_boundaries: Iterable[float] = (),
) -> list[MediaShot]:
    """Repartition at validated boundaries to avoid very short shots when possible.

    The maximum is hard. The minimum is a strong preference: the dynamic planner
    first minimizes the number and total shortfall of undersized shots, then
    preserves semantic scene boundaries and higher-quality timing boundaries.
    Existing balanced fallback cuts remain candidates so coverage can never become
    less robust than the base deterministic plan.
    """
    scenes = [scene for scene in list(timeline.get("segments", []) or []) if isinstance(scene, dict)]
    if not scenes or not base_shots:
        return base_shots

    points, source_map, scene_boundaries = _collect_boundary_catalog(
        scenes, base_shots, audio_duration=audio_duration
    )
    if len(points) < 2:
        return base_shots

    locked = tuple(sorted({_time_key(value) for value in locked_boundaries}))
    midpoint = (minimum + maximum) / 2.0

    # Cost tuple, lexicographically minimized:
    # short shots -> shortfall -> skipped semantic boundaries -> low-quality cuts
    # -> synthetic cuts -> shot count -> duration balance tie-breaker.
    best: list[tuple[Any, ...] | None] = [None] * len(points)
    previous: list[int | None] = [None] * len(points)
    best[0] = (0, 0.0, 0, 0, 0, 0, 0.0)

    for right_index in range(1, len(points)):
        right = points[right_index]
        right_source = source_map.get(_time_key(right), BOUNDARY_BALANCED)
        source_penalty = _SOURCE_PRIORITY.get(right_source, 9)
        for left_index in range(right_index - 1, -1, -1):
            left = points[left_index]
            duration = right - left
            if duration > maximum + 1e-6:
                break
            if duration <= 1e-6 or best[left_index] is None:
                continue
            if _crosses_locked_boundary(locked, left, right):
                continue

            short = 1 if duration < minimum - 1e-6 else 0
            shortfall = max(0.0, minimum - duration)
            skipped_scenes = _edge_crossed_scene_count(scene_boundaries, left, right)
            balanced_cut = 1 if right_source == BOUNDARY_BALANCED and right < audio_duration - 1e-6 else 0
            balance_penalty = abs(duration - midpoint)

            edge = (
                short,
                round(shortfall, 6),
                skipped_scenes,
                source_penalty,
                balanced_cut,
                1,
                round(balance_penalty, 6),
            )
            candidate = tuple(a + b for a, b in zip(best[left_index], edge))
            if best[right_index] is None or candidate < best[right_index]:
                best[right_index] = candidate
                previous[right_index] = left_index

    if best[-1] is None:
        return base_shots

    path_indices: list[int] = []
    cursor: int | None = len(points) - 1
    while cursor is not None:
        path_indices.append(cursor)
        if cursor == 0:
            break
        cursor = previous[cursor]
    if not path_indices or path_indices[-1] != 0:
        return base_shots
    path_indices.reverse()
    cuts = [points[index] for index in path_indices]

    shots: list[MediaShot] = []
    for start, end in zip(cuts, cuts[1:]):
        text, narration_indices, alignment_indices, scene_indices = _window_text_and_indices(
            scenes, start, end
        )
        if not scene_indices:
            scene_indices = _scene_indices_for_window(scenes, start, end)
        scene_index = scene_indices[0] if scene_indices else 1
        shots.append(
            MediaShot(
                index=len(shots) + 1,
                scene_index=scene_index,
                scene_indices=scene_indices or (scene_index,),
                start=start,
                end=end,
                narration_text=text,
                narration_indices=narration_indices,
                alignment_indices=alignment_indices,
                start_boundary_source=source_map.get(_time_key(start), BOUNDARY_BALANCED),
                end_boundary_source=source_map.get(_time_key(end), BOUNDARY_BALANCED),
            )
        )
    return shots or base_shots


def _validate_final_plan(
    shots: list[MediaShot], *, audio_duration: float, max_clip_duration: float
) -> None:
    if not shots:
        raise ValueError("visual timeline did not produce any media shots")
    if not math.isclose(shots[0].start, 0.0, abs_tol=1e-6):
        raise ValueError("media shot plan must start at 0 seconds")
    if not math.isclose(shots[-1].end, audio_duration, abs_tol=1e-6):
        raise ValueError("media shot plan must cover the full audio duration")
    for shot in shots:
        if shot.duration > max_clip_duration + 1e-6:
            raise ValueError("media shot planner produced a shot longer than max_clip_duration")
    for previous_shot, current in zip(shots, shots[1:]):
        if not math.isclose(previous_shot.end, current.start, abs_tol=1e-6):
            raise ValueError("media shot plan contains a gap or overlap")


def build_media_shot_plan(
    visual_timeline: Any,
    *,
    max_clip_duration: float = 5.0,
    minimum_shot_duration: float = DEFAULT_MINIMUM_SHOT_DURATION,
    preferred_scene_cuts: dict[int, Iterable[Any]] | None = None,
) -> MediaShotPlan:
    """Build a safe shot plan with a hard maximum and preferred minimum.

    Long semantic scenes are first split using narration boundaries, finer provider
    alignment, then a balanced temporal fallback. A second deterministic pass may
    merge/repartition across semantic-scene boundaries to avoid shots shorter than
    ``minimum_shot_duration``. The minimum is soft; the maximum, full coverage,
    temporal order, and any LLM-selected validated boundaries remain hard.
    """
    timeline = _as_mapping(visual_timeline)
    audio_duration = _safe_float(timeline.get("audio_duration"), 0.0)
    if audio_duration <= 0:
        raise ValueError("visual timeline audio_duration must be positive")

    maximum = _safe_float(max_clip_duration, 0.0)
    if maximum <= 0:
        raise ValueError("max_clip_duration must be a positive finite number")
    requested_minimum = _safe_float(minimum_shot_duration, DEFAULT_MINIMUM_SHOT_DURATION)
    if requested_minimum <= 0:
        requested_minimum = DEFAULT_MINIMUM_SHOT_DURATION
    minimum = min(requested_minimum, maximum)

    timing_source = str(timeline.get("timing_source") or "estimated")
    scenes = list(timeline.get("segments", []) or [])
    if not scenes:
        raise ValueError("visual timeline must contain at least one semantic scene")

    preferred_scene_cuts = preferred_scene_cuts or {}
    base_shots: list[MediaShot] = []
    locked_boundaries: list[float] = []

    # A strong sentence ending is a hard visual cut.
    # The minimum shot duration must never merge across it.
    for scene in scenes[:-1]:
        if not isinstance(scene, dict):
            continue

        narration_text = str(
            scene.get("narration_text", "") or ""
        ).strip()

        if _STRONG_SENTENCE_END_RE.search(narration_text):
            locked_boundaries.append(
                _safe_float(scene.get("end"), 0.0)
            )

    for scene_position, raw_scene in enumerate(scenes, start=1):
        if not isinstance(raw_scene, dict):
            continue
        scene_index = int(raw_scene.get("index", scene_position))
        start = max(0.0, _safe_float(raw_scene.get("start"), 0.0))
        end = max(start, min(audio_duration, _safe_float(raw_scene.get("end"), start)))
        if end <= start:
            continue

        narration_candidates = _unique_sorted(
            raw_scene.get("narration_boundaries", []) or [], start=start, end=end
        )
        alignment_candidates = _unique_sorted(
            raw_scene.get("alignment_boundaries", []) or [], start=start, end=end
        )

        if scene_index in preferred_scene_cuts:
            preferred = list(preferred_scene_cuts[scene_index])
            cuts, sources = _preferred_scene_cuts(
                start=start,
                end=end,
                preferred=preferred,
                narration_boundaries=narration_candidates,
                alignment_boundaries=alignment_candidates,
                max_clip_duration=maximum,
            )
            locked_boundaries.extend(_unique_sorted(preferred, start=start, end=end))
        else:
            cuts, sources = _scene_cuts(
                start=start,
                end=end,
                narration_boundaries=narration_candidates,
                alignment_boundaries=alignment_candidates,
                max_clip_duration=maximum,
            )

        for left, right in zip(cuts, cuts[1:]):
            narration_text, narration_indices, alignment_indices = _shot_text_and_indices(
                raw_scene, left, right
            )
            base_shots.append(
                MediaShot(
                    index=len(base_shots) + 1,
                    scene_index=scene_index,
                    scene_indices=(scene_index,),
                    start=left,
                    end=right,
                    narration_text=narration_text,
                    narration_indices=narration_indices,
                    alignment_indices=alignment_indices,
                    start_boundary_source=sources.get(left, BOUNDARY_BALANCED),
                    end_boundary_source=sources.get(right, BOUNDARY_BALANCED),
                )
            )

    shots = _normalize_minimum_duration(
        timeline,
        base_shots,
        audio_duration=audio_duration,
        minimum=minimum,
        maximum=maximum,
        locked_boundaries=locked_boundaries,
    )
    # Normalize indices after any cross-scene merging/repartitioning.
    shots = [
        MediaShot(
            index=index,
            scene_index=shot.scene_index,
            scene_indices=shot.scene_indices,
            start=shot.start,
            end=shot.end,
            narration_text=shot.narration_text,
            narration_indices=shot.narration_indices,
            alignment_indices=shot.alignment_indices,
            start_boundary_source=shot.start_boundary_source,
            end_boundary_source=shot.end_boundary_source,
        )
        for index, shot in enumerate(shots, start=1)
    ]

    _validate_final_plan(shots, audio_duration=audio_duration, max_clip_duration=maximum)
    return MediaShotPlan(
        audio_duration=audio_duration,
        max_clip_duration=maximum,
        min_clip_duration=minimum,
        timing_source=timing_source,
        shots=tuple(shots),
    )
