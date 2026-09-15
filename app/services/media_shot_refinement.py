from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from app.services import media_shot_plan


@dataclass(frozen=True)
class RefinementResult:
    plan: media_shot_plan.MediaShotPlan
    eligible_scene_count: int
    refined_scene_count: int
    rejected_choice_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan.to_dict(),
            "eligible_scene_count": self.eligible_scene_count,
            "refined_scene_count": self.refined_scene_count,
            "rejected_choice_count": self.rejected_choice_count,
        }


def _as_mapping(value: Any, name: str) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping or expose to_dict()")
    return value


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(result):
        return default
    return result


def _strip_code_fence(text: str) -> str:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = _strip_code_fence(text)
    try:
        value = json.loads(cleaned)
    except Exception:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise ValueError("LLM refinement response does not contain a JSON object")
        value = json.loads(match.group())
    if not isinstance(value, dict):
        raise ValueError("LLM refinement response is not a JSON object")
    return value


def _smart_join(parts: Iterable[str]) -> str:
    result = ""
    for raw in parts:
        piece = str(raw or "").strip()
        if not piece:
            continue
        if not result:
            result = piece
        elif piece[0] in ",.;:!?…)]}»”’":
            result += piece
        elif result[-1:] in "([{«“‘\"'":
            result += piece
        else:
            result += " " + piece
    return re.sub(r"\s+([,.;:!?…\]\)}»”’])", r"\1", result).strip()


def _text_for_window(scene: dict[str, Any], start: float, end: float) -> str:
    spans = list(scene.get("alignment_spans", []) or [])
    if not spans:
        spans = list(scene.get("narration_spans", []) or [])
    parts: list[str] = []
    for span in spans:
        if not isinstance(span, dict):
            continue
        span_start = _safe_float(span.get("start"), start)
        span_end = _safe_float(span.get("end"), span_start)
        if span_end > start + 1e-9 and span_start < end - 1e-9:
            text = str(span.get("text", "") or "").strip()
            if text:
                parts.append(text)
    return _smart_join(parts) or str(scene.get("narration_text", "") or "").strip()


def _dedupe_boundaries(values: Iterable[Any], *, start: float, end: float) -> list[float]:
    result: list[float] = []
    for raw in values:
        value = _safe_float(raw, math.nan)
        if not math.isfinite(value) or not (start < value < end):
            continue
        if not any(math.isclose(value, existing, abs_tol=1e-6) for existing in result):
            result.append(value)
    return sorted(result)


def _shots_by_scene(plan: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = {}
    for shot in list(plan.get("shots", []) or []):
        if not isinstance(shot, dict):
            continue
        try:
            scene_index = int(shot.get("scene_index"))
        except (TypeError, ValueError, OverflowError):
            continue
        result.setdefault(scene_index, []).append(shot)
    return result


def build_refinement_candidates(
    visual_timeline: Any,
    base_plan: Any,
    *,
    max_scenes: int = 12,
    max_candidates_per_scene: int = 7,
) -> list[dict[str, Any]]:
    """Return only low-risk one-cut scenes where an LLM can choose a better boundary.

    The LLM never receives authority to create timestamps. Each returned candidate
    is an existing narration/alignment boundary that already satisfies the hard
    clip-duration ceiling on both sides.
    """
    timeline = _as_mapping(visual_timeline, "visual_timeline")
    plan = _as_mapping(base_plan, "base_plan")
    maximum = _safe_float(plan.get("max_clip_duration"), 0.0)
    if maximum <= 0:
        raise ValueError("base_plan max_clip_duration must be positive")
    minimum = _safe_float(
        plan.get("min_clip_duration"), media_shot_plan.DEFAULT_MINIMUM_SHOT_DURATION
    )
    minimum = min(maximum, max(0.0, minimum))

    scene_shots = _shots_by_scene(plan)
    result: list[dict[str, Any]] = []
    for raw_scene in list(timeline.get("segments", []) or []):
        if len(result) >= max_scenes:
            break
        if not isinstance(raw_scene, dict):
            continue
        try:
            scene_index = int(raw_scene.get("index"))
        except (TypeError, ValueError, OverflowError):
            continue
        shots = scene_shots.get(scene_index, [])
        if len(shots) != 2:
            continue
        # Minimum-duration normalization may create a shot spanning multiple
        # semantic scenes. Such shots are already a cross-scene compromise and
        # are deliberately excluded from optional LLM refinement.
        if any(len(list(shot.get("scene_indices", []) or [scene_index])) != 1 for shot in shots):
            continue

        # Narration-boundary cuts are already semantically strong; ask the LLM only
        # when the deterministic plan had to use a finer alignment or a fallback.
        current_source = str(shots[0].get("end_boundary_source") or "")
        if current_source == media_shot_plan.BOUNDARY_NARRATION:
            continue

        start = _safe_float(raw_scene.get("start"), 0.0)
        end = _safe_float(raw_scene.get("end"), start)
        current_cut = _safe_float(shots[0].get("end"), start)
        if end <= start:
            continue

        narration_boundaries = _dedupe_boundaries(
            raw_scene.get("narration_boundaries", []) or [], start=start, end=end
        )
        alignment_boundaries = _dedupe_boundaries(
            raw_scene.get("alignment_boundaries", []) or [], start=start, end=end
        )
        all_boundaries = _dedupe_boundaries(
            [*narration_boundaries, *alignment_boundaries], start=start, end=end
        )
        valid = [
            boundary
            for boundary in all_boundaries
            if boundary - start <= maximum + 1e-6
            and end - boundary <= maximum + 1e-6
            and boundary - start >= minimum - 1e-6
            and end - boundary >= minimum - 1e-6
        ]
        if len(valid) < 2:
            continue

        narration_set = narration_boundaries
        preferred = sorted(
            valid,
            key=lambda boundary: (
                0
                if any(math.isclose(boundary, n, abs_tol=1e-6) for n in narration_set)
                else 1,
                abs(boundary - current_cut),
            ),
        )[:max_candidates_per_scene]
        preferred = sorted(preferred)

        candidates: list[dict[str, Any]] = []
        for position, boundary in enumerate(preferred, start=1):
            boundary_id = f"S{scene_index}_B{position}"
            boundary_type = (
                "narration"
                if any(math.isclose(boundary, n, abs_tol=1e-6) for n in narration_set)
                else "alignment"
            )
            candidates.append(
                {
                    "id": boundary_id,
                    "boundary": boundary,
                    "boundary_type": boundary_type,
                    "current": math.isclose(boundary, current_cut, abs_tol=1e-6),
                    "left_duration": round(boundary - start, 3),
                    "right_duration": round(end - boundary, 3),
                    "before": _text_for_window(raw_scene, start, boundary),
                    "after": _text_for_window(raw_scene, boundary, end),
                }
            )

        result.append(
            {
                "scene_index": scene_index,
                "scene_text": str(raw_scene.get("narration_text", "") or "").strip(),
                "candidates": candidates,
            }
        )
    return result


def _build_prompt(candidate_scenes: list[dict[str, Any]]) -> str:
    safe_payload = []
    for scene in candidate_scenes:
        safe_payload.append(
            {
                "scene_index": scene["scene_index"],
                "scene_text": scene["scene_text"],
                "candidates": [
                    {
                        "id": item["id"],
                        "boundary_type": item["boundary_type"],
                        "current": item["current"],
                        "left_duration": item["left_duration"],
                        "right_duration": item["right_duration"],
                        "before": item["before"],
                        "after": item["after"],
                    }
                    for item in scene["candidates"]
                ],
            }
        )

    return f"""
# Role: Media Shot Boundary Editor

Choose the best existing cut boundary for each supplied scene.
This is a constrained editing task: you may ONLY choose one of the candidate IDs
listed for that scene. Do not create timestamps, new IDs, new scenes, or new text.

Prefer, in order:
1. a complete semantic/action idea on each side of the cut;
2. a natural phrase boundary rather than breaking a noun or verb phrase;
3. two visually useful and non-redundant shot descriptions;
4. the current candidate when no alternative is clearly better.

Return ONLY valid JSON in this exact shape:
{{"choices":[{{"scene_index":1,"boundary_id":"S1_B2"}}]}}
Include at most one choice per scene. Include every supplied scene.

Candidate scenes:
{json.dumps(safe_payload, ensure_ascii=False, separators=(",", ":"))}
""".strip()


def refine_media_shot_plan(
    visual_timeline: Any,
    base_plan: Any,
    *,
    response_generator: Callable[[str], str],
) -> RefinementResult:
    """Optionally refine low-risk internal cuts, then rebuild through the validator.

    Any unknown or malformed choice is ignored. The final plan is always rebuilt by
    ``build_media_shot_plan`` using validated existing boundaries only.
    """
    candidate_scenes = build_refinement_candidates(visual_timeline, base_plan)
    if not candidate_scenes:
        plan_mapping = _as_mapping(base_plan, "base_plan")
        rebuilt = media_shot_plan.build_media_shot_plan(
            visual_timeline,
            max_clip_duration=_safe_float(plan_mapping.get("max_clip_duration"), 5.0),
            minimum_shot_duration=_safe_float(
                plan_mapping.get("min_clip_duration"),
                media_shot_plan.DEFAULT_MINIMUM_SHOT_DURATION,
            ),
        )
        return RefinementResult(
            plan=rebuilt,
            eligible_scene_count=0,
            refined_scene_count=0,
            rejected_choice_count=0,
        )

    response = response_generator(_build_prompt(candidate_scenes))
    data = _parse_json_object(response)
    raw_choices = data.get("choices")
    if not isinstance(raw_choices, list):
        raise ValueError("LLM refinement response must contain a choices array")

    candidate_map: dict[tuple[int, str], float] = {}
    eligible_scene_ids: set[int] = set()
    for scene in candidate_scenes:
        scene_index = int(scene["scene_index"])
        eligible_scene_ids.add(scene_index)
        for item in scene["candidates"]:
            candidate_map[(scene_index, str(item["id"]))] = float(item["boundary"])

    preferred: dict[int, list[float]] = {}
    rejected = 0
    seen_scenes: set[int] = set()
    for raw in raw_choices:
        if not isinstance(raw, dict):
            rejected += 1
            continue
        try:
            scene_index = int(raw.get("scene_index"))
        except (TypeError, ValueError, OverflowError):
            rejected += 1
            continue
        boundary_id = str(raw.get("boundary_id") or "").strip()
        key = (scene_index, boundary_id)
        if scene_index in seen_scenes or key not in candidate_map:
            rejected += 1
            continue
        preferred[scene_index] = [candidate_map[key]]
        seen_scenes.add(scene_index)

    base_mapping = _as_mapping(base_plan, "base_plan")
    maximum = _safe_float(base_mapping.get("max_clip_duration"), 0.0)
    refined = media_shot_plan.build_media_shot_plan(
        visual_timeline,
        max_clip_duration=maximum,
        minimum_shot_duration=_safe_float(
            base_mapping.get("min_clip_duration"),
            media_shot_plan.DEFAULT_MINIMUM_SHOT_DURATION,
        ),
        preferred_scene_cuts=preferred,
    )

    # Extra defensive invariant: this refinement changes only boundary choice,
    # never scene count or shot count.
    base_shots = list(base_mapping.get("shots", []) or [])
    if len(refined.shots) != len(base_shots):
        raise ValueError("LLM refinement changed the number of media shots")

    return RefinementResult(
        plan=refined,
        eligible_scene_count=len(eligible_scene_ids),
        refined_scene_count=len(preferred),
        rejected_choice_count=rejected,
    )
