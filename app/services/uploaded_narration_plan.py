from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.services import media_shot_plan, narration_alignment, visual_timeline


@dataclass(frozen=True)
class UploadedNarrationPlan:
    alignment: narration_alignment.ExternalNarrationAlignment
    semantic_timeline: visual_timeline.VisualTimeline
    active_shot_plan: media_shot_plan.MediaShotPlan

    def to_dict(self) -> dict[str, Any]:
        return {
            "alignment": self.alignment.to_dict(),
            "narration_timeline": self.alignment.timeline.to_dict(),
            "semantic_timeline": self.semantic_timeline.to_dict(),
            "active_shot_plan": self.active_shot_plan.to_dict(),
        }


def build_uploaded_narration_plan(
    *,
    audio_file: str,
    script: str,
    max_clip_duration: float,
    audio_duration: float | None = None,
    recognized_words: list[narration_alignment.RecognizedWord] | None = None,
) -> UploadedNarrationPlan:
    """Build the existing audio-first planning chain for an uploaded voiceover.

    This is intentionally a thin adapter. Once strict audio↔script alignment has
    produced the provider-neutral NarrationTimeline, the exact same semantic and
    media-shot planners used by internal TTS take over.
    """
    alignment = narration_alignment.align_external_narration(
        audio_file=audio_file,
        script=script,
        audio_duration=audio_duration,
        recognized_words=recognized_words,
    )
    semantic_timeline = visual_timeline.build_visual_timeline(alignment.timeline)
    active_shot_plan = media_shot_plan.build_media_shot_plan(
        semantic_timeline,
        max_clip_duration=float(max_clip_duration),
    )
    return UploadedNarrationPlan(
        alignment=alignment,
        semantic_timeline=semantic_timeline,
        active_shot_plan=active_shot_plan,
    )
