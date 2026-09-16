from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Callable

from loguru import logger


MAX_VISUAL_PROMPT_CHARS = 1200
MAX_SHOTS_PER_REQUEST = 60
MAX_VISUAL_SHOT_ATTEMPTS = 3
VISUAL_SHOT_BATCH_SIZE = 8


@dataclass(frozen=True)
class VisualShot:
    index: int
    start: float
    end: float
    duration: float
    scene_indices: tuple[int, ...]
    narration_indices: tuple[int, ...]
    narration_text: str
    shared_narration_with_previous: bool
    continues_same_scene: bool
    visual_prompt: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["scene_indices"] = list(self.scene_indices)
        data["narration_indices"] = list(self.narration_indices)
        return data


@dataclass(frozen=True)
class VisualShotPlan:
    source_shot_count: int
    shared_narration_continuation_count: int
    shots: tuple[VisualShot, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_shot_count": self.source_shot_count,
            "shared_narration_continuation_count": self.shared_narration_continuation_count,
            "shots": [shot.to_dict() for shot in self.shots],
            "visual_prompts": [shot.visual_prompt for shot in self.shots],
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


def _int_tuple(values: Any) -> tuple[int, ...]:
    result: list[int] = []
    for raw in list(values or []):
        try:
            value = int(raw)
        except (TypeError, ValueError, OverflowError):
            continue
        if value not in result:
            result.append(value)
    return tuple(result)


def _strip_code_fence(text: str) -> str:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = _strip_code_fence(text)
    candidate = cleaned

    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise ValueError(
                "visual shot planner response does not contain a JSON object"
            )

        candidate = match.group()

        try:
            value = json.loads(candidate)
        except json.JSONDecodeError as exc:
            start = max(0, exc.pos - 200)
            end = min(len(candidate), exc.pos + 200)
            context = candidate[start:end].replace("\n", "\\n")

            raise ValueError(
                "visual shot planner returned invalid JSON at "
                f"line {exc.lineno}, column {exc.colno}, char {exc.pos}; "
                f"context={context!r}"
            ) from exc

    if not isinstance(value, dict):
        raise ValueError("visual shot planner response is not a JSON object")

    return value


def _normalize_prompt_for_duplicate_check(text: str) -> str:
    normalized = str(text or "").strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"[^\w\s]+", "", normalized, flags=re.UNICODE)
    return normalized.strip()


def build_visual_shot_payload(base_plan: Any) -> list[dict[str, Any]]:
    plan = _as_mapping(base_plan, "base_plan")
    raw_shots = list(plan.get("shots", []) or [])
    if not raw_shots:
        raise ValueError("base_plan contains no media shots")
    if len(raw_shots) > MAX_SHOTS_PER_REQUEST:
        raise ValueError(
            f"visual shot planner currently supports at most {MAX_SHOTS_PER_REQUEST} shots per request"
        )

    payload: list[dict[str, Any]] = []
    previous_narration: set[int] = set()
    previous_scenes: set[int] = set()

    for position, raw in enumerate(raw_shots, start=1):
        if not isinstance(raw, dict):
            raise ValueError("every media shot must be a mapping")
        try:
            shot_index = int(raw.get("index", position))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("media shot index is invalid") from exc
        if shot_index != position:
            raise ValueError("media shot indices must be consecutive and ordered")

        start = _safe_float(raw.get("start"), math.nan)
        end = _safe_float(raw.get("end"), math.nan)
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            raise ValueError(f"media shot {shot_index} has invalid timing")

        narration_indices = _int_tuple(raw.get("narration_indices", []))
        raw_scene_indices = raw.get("scene_indices", []) or []
        if not raw_scene_indices and raw.get("scene_index") is not None:
            raw_scene_indices = [raw.get("scene_index")]
        scene_indices = _int_tuple(raw_scene_indices)
        narration_set = set(narration_indices)
        scene_set = set(scene_indices)
        shared_narration = bool(position > 1 and narration_set & previous_narration)
        continues_same_scene = bool(position > 1 and scene_set & previous_scenes)

        payload.append(
            {
                "shot_index": shot_index,
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(end - start, 3),
                "scene_indices": list(scene_indices),
                "narration_indices": list(narration_indices),
                "narration_text": str(raw.get("narration_text", "") or "").strip(),
                "shared_narration_with_previous": shared_narration,
                "continues_same_scene": continues_same_scene,
                "end_boundary_source": str(raw.get("end_boundary_source", "") or ""),
            }
        )
        previous_narration = narration_set
        previous_scenes = scene_set

    return payload


def _build_prompt(payload: list[dict[str, Any]], *, video_subject: str = "") -> str:
    subject = str(video_subject or "").strip()
    subject_block = subject if subject else "(not provided)"
    return f"""
# Role: Visual Shot Planner

Convert the locked Media Shot Plan below into exactly one unique visual prompt for each shot.
The timing, shot count, shot order, narration-unit mapping, and scene mapping are authoritative and MUST NOT be changed.

## Critical rules
1. Return exactly {len(payload)} shots, in the same order, with shot_index 1 through {len(payload)} exactly once each.
2. Return ONLY a visual_prompt for each shot. Do not return or invent timestamps, durations, narration units, scene IDs, or new shots.
3. Write every visual_prompt in English as a concrete cinematic description of what should be visible during that specific shot.
4. Do NOT merely paraphrase the narration. Translate narrative meaning into visible action, subject, framing, environment, and transient state.
5. When shared_narration_with_previous is true, the current shot covers a different temporal portion of the SAME narration unit. Do not repeat the previous visual composition or the same full prompt. Split the idea into a natural visual progression, for example establishing view -> character action -> object/detail/reaction, while staying faithful to the narration.
6. When continues_same_scene is true, preserve continuity of characters, location, important objects, and story state. You may vary framing, focus, or the next visible action, but do not reset the scene.
7. Do not invent plot events, characters, locations, props, or outcomes that are unsupported by the supplied narration.
8. Avoid visible captions, subtitles, logos, watermarks, UI text, or explanatory text. If the story mentions a plaque, sign, phone, indicator, or similar object, show the object naturally without relying on perfectly readable generated text.
9. Do not add a global art style. A downstream prompt template handles style separately.
10. Keep each prompt concise but specific enough to generate a distinct shot. Never return two identical visual prompts.
11. Treat repeated narration text across consecutive shots as timing overlap metadata, not as an instruction to duplicate the same image.

## Video subject
{subject_block}

## Locked Media Shot Plan
{json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}

Return ONLY valid JSON in this exact shape:
{{"shots":[{{"shot_index":1,"visual_prompt":"..."}}]}}
""".strip()


def _validate_llm_shots(data: dict[str, Any], expected_count: int) -> dict[int, str]:
    raw_shots = data.get("shots")
    if not isinstance(raw_shots, list):
        raise ValueError("visual shot planner response must contain a shots array")
    if len(raw_shots) != expected_count:
        raise ValueError(
            f"visual shot planner returned {len(raw_shots)} shots; expected {expected_count}"
        )

    prompts: dict[int, str] = {}
    normalized_prompts: dict[str, int] = {}
    for raw in raw_shots:
        if not isinstance(raw, dict):
            raise ValueError("every visual shot response item must be an object")
        try:
            shot_index = int(raw.get("shot_index"))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("visual shot response contains an invalid shot_index") from exc
        if shot_index < 1 or shot_index > expected_count or shot_index in prompts:
            raise ValueError("visual shot response contains duplicate or out-of-range shot_index")
        prompt = str(raw.get("visual_prompt", "") or "").strip()
        if not prompt:
            raise ValueError(f"visual prompt for shot {shot_index} is empty")
        if len(prompt) > MAX_VISUAL_PROMPT_CHARS:
            raise ValueError(
                f"visual prompt for shot {shot_index} exceeds {MAX_VISUAL_PROMPT_CHARS} characters"
            )
        normalized = _normalize_prompt_for_duplicate_check(prompt)
        if not normalized:
            raise ValueError(f"visual prompt for shot {shot_index} contains no useful text")
        if normalized in normalized_prompts:
            previous = normalized_prompts[normalized]
            raise ValueError(
                f"visual prompts for shots {previous} and {shot_index} are identical"
            )
        normalized_prompts[normalized] = shot_index
        prompts[shot_index] = prompt

    if set(prompts) != set(range(1, expected_count + 1)):
        raise ValueError("visual shot response does not cover every expected shot index")
    return prompts


def generate_visual_shot_plan(
    base_plan: Any,
    *,
    response_generator: Callable[[str], str],
    video_subject: str = "",
) -> VisualShotPlan:
    payload = build_visual_shot_payload(base_plan)

    prompts: dict[int, str] = {}
    batch_count = math.ceil(len(payload) / VISUAL_SHOT_BATCH_SIZE)

    for batch_number, batch_start in enumerate(
        range(0, len(payload), VISUAL_SHOT_BATCH_SIZE),
        start=1,
    ):
        source_batch = payload[
            batch_start : batch_start + VISUAL_SHOT_BATCH_SIZE
        ]

        # The LLM sees compact local indices 1..N inside each batch.
        # Results are mapped back to the authoritative global shot indices
        # immediately after validation.
        batch_payload: list[dict[str, Any]] = []

        for local_index, item in enumerate(source_batch, start=1):
            local_item = dict(item)
            local_item["shot_index"] = local_index
            batch_payload.append(local_item)

        global_first = int(source_batch[0]["shot_index"])
        global_last = int(source_batch[-1]["shot_index"])

        logger.info(
            "visual shot planner batch "
            f"{batch_number}/{batch_count}: "
            f"shots {global_first}-{global_last}"
        )

        prompt = _build_prompt(
            batch_payload,
            video_subject=video_subject,
        )

        batch_prompts = None

        for attempt in range(1, MAX_VISUAL_SHOT_ATTEMPTS + 1):
            try:
                response = response_generator(prompt)

                if str(response or "").startswith("Error:"):
                    detail = str(response).removeprefix("Error:").strip()
                    raise ValueError(
                        "visual shot planner LLM request failed: "
                        + (detail or "unknown LLM error")
                    )

                batch_prompts = _validate_llm_shots(
                    _parse_json_object(response),
                    len(batch_payload),
                )
                break

            except (json.JSONDecodeError, ValueError) as exc:
                if attempt >= MAX_VISUAL_SHOT_ATTEMPTS:
                    raise

                logger.warning(
                    "visual shot planner batch "
                    f"{batch_number}/{batch_count} rejected; "
                    f"retrying {attempt}/"
                    f"{MAX_VISUAL_SHOT_ATTEMPTS - 1}: {exc}"
                )

        if batch_prompts is None:
            raise ValueError(
                f"visual shot planner batch "
                f"{batch_number}/{batch_count} "
                "did not produce a valid response"
            )

        for local_index, item in enumerate(source_batch, start=1):
            global_index = int(item["shot_index"])
            prompts[global_index] = batch_prompts[local_index]

    # Revalidate the combined result so completeness, global ordering,
    # empty prompts and exact duplicates are still enforced across batches.
    prompts = _validate_llm_shots(
        {
            "shots": [
                {
                    "shot_index": index,
                    "visual_prompt": prompts[index],
                }
                for index in range(1, len(payload) + 1)
            ]
        },
        len(payload),
    )

    shots: list[VisualShot] = []
    shared_count = 0

    for item in payload:
        shared = bool(item["shared_narration_with_previous"])
        if shared:
            shared_count += 1

        shots.append(
            VisualShot(
                index=int(item["shot_index"]),
                start=float(item["start"]),
                end=float(item["end"]),
                duration=float(item["duration"]),
                scene_indices=tuple(
                    int(value) for value in item["scene_indices"]
                ),
                narration_indices=tuple(
                    int(value) for value in item["narration_indices"]
                ),
                narration_text=str(item["narration_text"]),
                shared_narration_with_previous=shared,
                continues_same_scene=bool(
                    item["continues_same_scene"]
                ),
                visual_prompt=prompts[int(item["shot_index"])],
            )
        )

    return VisualShotPlan(
        source_shot_count=len(payload),
        shared_narration_continuation_count=shared_count,
        shots=tuple(shots),
    )
