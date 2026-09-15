from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable


TIMELINE_AWARE_IMAGE_SOURCES = frozenset(
    {"openai_image", "comfyui_t2i", "comfyui_mage"}
)
TIMELINE_AWARE_VIDEO_SOURCES = frozenset({"comfyui_video"})
TIMELINE_AWARE_GENERATED_SOURCES = frozenset(
    set(TIMELINE_AWARE_IMAGE_SOURCES) | set(TIMELINE_AWARE_VIDEO_SOURCES)
)
AUDIO_DURATION_TOLERANCE_SECONDS = 0.20
TIMELINE_OUTPUT_SHORTFALL_TOLERANCE_SECONDS = 0.08
TIMING_EPSILON = 1e-6

HYBRID_MEDIA_IMAGE_PROVIDERS = frozenset({"openai_image", "comfyui_t2i"})
HYBRID_MEDIA_VIDEO_PROVIDERS = frozenset({"comfyui_video"})
HYBRID_MEDIA_PROVIDERS = frozenset(
    set(HYBRID_MEDIA_IMAGE_PROVIDERS) | set(HYBRID_MEDIA_VIDEO_PROVIDERS)
)
HYBRID_IMAGE_MOTIONS = frozenset({"static", "zoom_in"})


class TimelineMediaError(ValueError):
    """Raised when a locked shot timeline is unsafe to use for media generation."""


@dataclass(frozen=True)
class LockedMediaShot:
    index: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
        }


def is_timeline_aware_image_source(source: str | None) -> bool:
    return str(source or "").strip() in TIMELINE_AWARE_IMAGE_SOURCES


def is_timeline_aware_video_source(source: str | None) -> bool:
    return str(source or "").strip() in TIMELINE_AWARE_VIDEO_SOURCES


def is_timeline_aware_generated_source(source: str | None) -> bool:
    return str(source or "").strip() in TIMELINE_AWARE_GENERATED_SOURCES


def _finite_float(value: Any, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TimelineMediaError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise TimelineMediaError(f"{name} must be finite")
    return number


def _as_shot_mapping(raw: Any) -> dict[str, Any]:
    if hasattr(raw, "to_dict"):
        raw = raw.to_dict()
    if not isinstance(raw, dict):
        raise TimelineMediaError("every media shot timeline item must be a mapping")
    return raw


def normalize_locked_shot_timeline(
    raw_timeline: Iterable[Any] | None,
    *,
    audio_duration: float,
    max_clip_duration: float,
    expected_count: int | None = None,
) -> list[dict[str, float | int]]:
    """Validate and normalize a locked media-shot timeline.

    The timeline is authoritative only when it covers the real narration audio from
    0 to the end, is contiguous, ordered, and never exceeds the configured hard
    maximum.  Supplied ``duration`` fields are ignored; duration is always derived
    from start/end so callers cannot create internally inconsistent windows.
    """
    duration = _finite_float(audio_duration, name="audio_duration")
    maximum = _finite_float(max_clip_duration, name="max_clip_duration")
    if duration <= 0:
        raise TimelineMediaError("audio_duration must be positive")
    if maximum <= 0:
        raise TimelineMediaError("max_clip_duration must be positive")

    items = list(raw_timeline or [])
    if not items:
        raise TimelineMediaError("media shot timeline is empty")
    if expected_count is not None and len(items) != int(expected_count):
        raise TimelineMediaError(
            f"media shot timeline has {len(items)} shots; expected {int(expected_count)}"
        )

    shots: list[LockedMediaShot] = []
    previous_end = 0.0
    for position, raw in enumerate(items, start=1):
        data = _as_shot_mapping(raw)
        try:
            index = int(data.get("index", position))
        except (TypeError, ValueError, OverflowError) as exc:
            raise TimelineMediaError("media shot timeline contains an invalid index") from exc
        if index != position:
            raise TimelineMediaError("media shot timeline indices must be consecutive and ordered")

        start = _finite_float(data.get("start"), name=f"shot {position} start")
        end = _finite_float(data.get("end"), name=f"shot {position} end")
        if start < -TIMING_EPSILON or end <= start + TIMING_EPSILON:
            raise TimelineMediaError(f"media shot {position} has invalid timing")
        if position == 1:
            if not math.isclose(start, 0.0, abs_tol=TIMING_EPSILON):
                raise TimelineMediaError("media shot timeline must start at 0 seconds")
        elif not math.isclose(start, previous_end, abs_tol=TIMING_EPSILON):
            raise TimelineMediaError("media shot timeline contains a gap or overlap")

        shot = LockedMediaShot(index=index, start=max(0.0, start), end=end)
        if shot.duration > maximum + TIMING_EPSILON:
            raise TimelineMediaError(
                f"media shot {position} lasts {shot.duration:.3f}s, exceeding the "
                f"{maximum:.3f}s hard maximum"
            )
        shots.append(shot)
        previous_end = end

    if not math.isclose(
        shots[-1].end,
        duration,
        abs_tol=AUDIO_DURATION_TOLERANCE_SECONDS,
    ):
        raise TimelineMediaError(
            "media shot timeline does not match the real narration duration: "
            f"timeline={shots[-1].end:.3f}s, audio={duration:.3f}s"
        )

    # Preserve the locked timeline end rather than stretching it to a rounded
    # duration.  The final compositor trims to the real audio file duration.
    return [shot.to_dict() for shot in shots]


def pair_prompts_with_timeline(
    search_terms: Iterable[str],
    shot_timeline: Iterable[Any],
    *,
    audio_duration: float,
    max_clip_duration: float,
) -> list[dict[str, Any]]:
    terms = [str(term or "").strip() for term in search_terms if str(term or "").strip()]
    shots = normalize_locked_shot_timeline(
        shot_timeline,
        audio_duration=audio_duration,
        max_clip_duration=max_clip_duration,
        expected_count=len(terms),
    )
    return [
        {
            **shot,
            "prompt": term,
        }
        for shot, term in zip(shots, terms)
    ]


def normalize_hybrid_media_plan(
    raw_plan: Iterable[Any] | None,
    shot_timeline: Iterable[Any] | None,
    search_terms: Iterable[str],
    *,
    audio_duration: float,
    max_clip_duration: float,
) -> list[dict[str, Any]]:
    """Validate one explicit provider decision for every locked timeline shot.

    Hybrid generation is only safe when planner rows, prompts and authoritative timing
    still describe the same Visual Shot Plan. Timing and prompts are therefore rebuilt
    from the locked/current inputs rather than trusted from editable UI payload fields.
    """
    prompts = [
        str(term or "").strip()
        for term in search_terms
        if str(term or "").strip()
    ]
    shots = normalize_locked_shot_timeline(
        shot_timeline,
        audio_duration=audio_duration,
        max_clip_duration=max_clip_duration,
        expected_count=len(prompts),
    )
    items = list(raw_plan or [])
    if len(items) != len(shots):
        raise TimelineMediaError(
            f"media plan has {len(items)} rows; expected {len(shots)} locked shots"
        )

    normalized: list[dict[str, Any]] = []
    for position, (raw, shot, prompt) in enumerate(zip(items, shots, prompts), start=1):
        data = _as_shot_mapping(raw)
        try:
            plan_index = int(data.get("shot_index", data.get("index", position)))
        except (TypeError, ValueError, OverflowError) as exc:
            raise TimelineMediaError(f"media plan shot {position} has an invalid index") from exc
        if plan_index != position:
            raise TimelineMediaError("media plan shot indices must be consecutive and ordered")

        for field in ("start", "end"):
            if field in data and data.get(field) is not None:
                planned_value = _finite_float(
                    data.get(field), name=f"media plan shot {position} {field}"
                )
                if not math.isclose(
                    planned_value, float(shot[field]), abs_tol=TIMING_EPSILON
                ):
                    raise TimelineMediaError(
                        f"media plan shot {position} timing no longer matches the locked timeline"
                    )

        planned_prompt = str(data.get("visual_prompt", data.get("prompt", "")) or "").strip()
        if planned_prompt and planned_prompt != prompt:
            raise TimelineMediaError(
                f"media plan shot {position} prompt no longer matches the applied Visual Shot Plan"
            )

        resource_type = str(data.get("resource_type", "") or "").strip().lower()
        provider = str(data.get("provider", "") or "").strip()
        if provider not in HYBRID_MEDIA_PROVIDERS:
            raise TimelineMediaError(
                f"media plan shot {position} uses unsupported provider: {provider or '<empty>'}"
            )
        expected_type = (
            "image" if provider in HYBRID_MEDIA_IMAGE_PROVIDERS else "video"
        )
        if resource_type != expected_type:
            raise TimelineMediaError(
                f"media plan shot {position} type/provider mismatch: "
                f"type={resource_type or '<empty>'}, provider={provider}"
            )

        if resource_type == "image":
            motion = str(data.get("motion", "static") or "static").strip()
            if motion not in HYBRID_IMAGE_MOTIONS:
                raise TimelineMediaError(
                    f"media plan shot {position} uses unsupported image motion: {motion}"
                )
        else:
            motion = ""

        normalized.append(
            {
                "index": position,
                "start": float(shot["start"]),
                "end": float(shot["end"]),
                "duration": float(shot["duration"]),
                "prompt": prompt,
                "resource_type": resource_type,
                "provider": provider,
                "motion": motion,
                "auto_rule": str(data.get("auto_rule", "") or ""),
                "narration_text": str(data.get("narration_text", "") or ""),
            }
        )

    return normalized


def hybrid_media_plan_providers(raw_plan: Iterable[Any] | None) -> set[str]:
    providers: set[str] = set()
    for raw in list(raw_plan or []):
        if isinstance(raw, dict):
            provider = str(raw.get("provider", "") or "").strip()
            if provider:
                providers.add(provider)
    return providers


def source_duration_for_timeline_shot(target_duration: float, clip_speed: float) -> float:
    """Return source duration that yields the locked target after speed scaling."""
    target = _finite_float(target_duration, name="target_duration")
    speed = _finite_float(clip_speed, name="clip_speed")
    if target <= 0:
        raise TimelineMediaError("target_duration must be positive")
    if speed <= 0:
        raise TimelineMediaError("clip_speed must be positive")
    return target * speed


def generation_seconds_for_timeline_video_shot(
    target_duration: float, clip_speed: float
) -> int:
    """Return the minimum whole-second T2V request for a locked shot.

    ComfyUI video workflows commonly expose integer second controls. The generated
    source must remain long enough after MoneyPrinterTurbo applies clip speed, so
    round upward rather than to the nearest integer.
    """
    source_duration = source_duration_for_timeline_shot(target_duration, clip_speed)
    return max(1, int(math.ceil(source_duration - TIMING_EPSILON)))


def timeline_video_source_is_long_enough(
    actual_duration: float, target_duration: float, clip_speed: float
) -> bool:
    """Return whether a generated T2V source can cover the locked shot window."""
    actual = _finite_float(actual_duration, name="actual_duration")
    target = _finite_float(target_duration, name="target_duration")
    speed = _finite_float(clip_speed, name="clip_speed")
    if speed <= 0:
        raise TimelineMediaError("clip_speed must be positive")
    # Compare in final-playback seconds. This keeps the same tolerance whether the
    # user selects 0.5x, 1x, or 2x source playback speed.
    available_after_speed = actual / speed
    return (
        available_after_speed + TIMELINE_OUTPUT_SHORTFALL_TOLERANCE_SECONDS
        >= target
    )
