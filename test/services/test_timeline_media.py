import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import timeline_media


class TimelineMediaTests(unittest.TestCase):
    def _timeline(self):
        return [
            {"index": 1, "start": 0.0, "end": 4.15, "duration": 999},
            {"index": 2, "start": 4.15, "end": 8.30},
            {"index": 3, "start": 8.30, "end": 11.24},
        ]

    def test_valid_timeline_preserves_locked_windows_and_derived_duration(self):
        shots = timeline_media.normalize_locked_shot_timeline(
            self._timeline(),
            audio_duration=11.24,
            max_clip_duration=5.0,
            expected_count=3,
        )
        self.assertEqual(len(shots), 3)
        self.assertAlmostEqual(shots[0]["duration"], 4.15)
        self.assertAlmostEqual(shots[-1]["end"], 11.24)

    def test_rejects_prompt_count_mismatch(self):
        with self.assertRaises(timeline_media.TimelineMediaError):
            timeline_media.normalize_locked_shot_timeline(
                self._timeline(),
                audio_duration=11.24,
                max_clip_duration=5.0,
                expected_count=2,
            )

    def test_rejects_gap_or_overlap(self):
        broken = self._timeline()
        broken[1] = {"index": 2, "start": 4.20, "end": 8.30}
        with self.assertRaises(timeline_media.TimelineMediaError):
            timeline_media.normalize_locked_shot_timeline(
                broken,
                audio_duration=11.24,
                max_clip_duration=5.0,
            )

    def test_rejects_shot_over_hard_maximum(self):
        with self.assertRaises(timeline_media.TimelineMediaError):
            timeline_media.normalize_locked_shot_timeline(
                [{"index": 1, "start": 0.0, "end": 5.01}],
                audio_duration=5.01,
                max_clip_duration=5.0,
            )

    def test_rejects_stale_timeline_for_different_audio(self):
        with self.assertRaises(timeline_media.TimelineMediaError):
            timeline_media.normalize_locked_shot_timeline(
                self._timeline(),
                audio_duration=12.0,
                max_clip_duration=5.0,
            )

    def test_pairs_one_prompt_per_locked_shot(self):
        entries = timeline_media.pair_prompts_with_timeline(
            ["shot one", "shot two", "shot three"],
            self._timeline(),
            audio_duration=11.24,
            max_clip_duration=5.0,
        )
        self.assertEqual([entry["prompt"] for entry in entries], ["shot one", "shot two", "shot three"])
        self.assertEqual([round(entry["duration"], 2) for entry in entries], [4.15, 4.15, 2.94])

    def test_clip_speed_compensation_preserves_target_window(self):
        self.assertAlmostEqual(
            timeline_media.source_duration_for_timeline_shot(4.15, 2.0),
            8.30,
        )
        self.assertAlmostEqual(
            timeline_media.source_duration_for_timeline_shot(4.15, 0.5),
            2.075,
        )

    def test_timeline_aware_generated_sources_include_comfyui_video(self):
        self.assertTrue(timeline_media.is_timeline_aware_image_source("comfyui_t2i"))
        self.assertTrue(timeline_media.is_timeline_aware_video_source("comfyui_video"))
        self.assertTrue(timeline_media.is_timeline_aware_generated_source("comfyui_video"))
        self.assertFalse(timeline_media.is_timeline_aware_video_source("pexels"))

    def test_t2v_generation_seconds_round_up_after_clip_speed_compensation(self):
        self.assertEqual(
            timeline_media.generation_seconds_for_timeline_video_shot(4.15, 1.0),
            5,
        )
        self.assertEqual(
            timeline_media.generation_seconds_for_timeline_video_shot(4.15, 2.0),
            9,
        )
        self.assertEqual(
            timeline_media.generation_seconds_for_timeline_video_shot(4.15, 0.5),
            3,
        )

    def test_t2v_source_length_validation_happens_after_clip_speed(self):
        self.assertTrue(
            timeline_media.timeline_video_source_is_long_enough(5.0, 4.15, 1.0)
        )
        self.assertFalse(
            timeline_media.timeline_video_source_is_long_enough(4.0, 4.15, 1.0)
        )
        self.assertTrue(
            timeline_media.timeline_video_source_is_long_enough(8.30, 4.15, 2.0)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
