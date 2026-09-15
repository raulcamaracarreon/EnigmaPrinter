import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import media_shot_plan, visual_timeline


class TestMediaShotPlan(unittest.TestCase):
    def _timeline(self):
        narration = {
            "audio_duration": 18.0,
            "timing_source": "native",
            "segments": [
                {"index": 1, "start": 0.0, "end": 1.0, "text": "Cada noche,"},
                {"index": 2, "start": 1.0, "end": 3.0, "text": "poco después de las once,"},
                {"index": 3, "start": 3.0, "end": 8.3, "text": "una mujer salía de su oficina."},
                {"index": 4, "start": 8.3, "end": 11.0, "text": "El vestíbulo estaba vacío."},
                {"index": 5, "start": 11.0, "end": 18.0, "text": "El ascensor zumbaba constantemente."},
            ],
        }
        return visual_timeline.build_visual_timeline(narration)

    def test_clip_duration_is_hard_maximum_not_target(self):
        timeline = self._timeline()
        plan = media_shot_plan.build_media_shot_plan(timeline, max_clip_duration=5.0)
        self.assertTrue(plan.shots)
        self.assertTrue(all(shot.duration <= 5.0 + 1e-6 for shot in plan.shots))
        self.assertTrue(any(shot.duration < 5.0 - 0.25 for shot in plan.shots))

    def test_ten_second_limit_keeps_semantic_scenes_but_uses_fewer_shots(self):
        timeline = self._timeline()
        five = media_shot_plan.build_media_shot_plan(timeline, max_clip_duration=5.0)
        ten = media_shot_plan.build_media_shot_plan(timeline, max_clip_duration=10.0)
        self.assertEqual(len(timeline.segments), 3)
        self.assertLess(len(ten.shots), len(five.shots))
        self.assertTrue(all(shot.duration <= 10.0 + 1e-6 for shot in ten.shots))

    def test_prefers_real_narration_boundary_when_feasible(self):
        narration = {
            "audio_duration": 9.0,
            "timing_source": "native",
            "segments": [
                {"index": 1, "start": 0.0, "end": 4.0, "text": "La puerta se abrió,"},
                {"index": 2, "start": 4.0, "end": 9.0, "text": "y la figura apareció."},
            ],
        }
        timeline = visual_timeline.build_visual_timeline(narration)
        plan = media_shot_plan.build_media_shot_plan(timeline, max_clip_duration=5.0)
        self.assertEqual(len(plan.shots), 2)
        self.assertAlmostEqual(plan.shots[0].end, 4.0)
        self.assertEqual(plan.shots[0].end_boundary_source, media_shot_plan.BOUNDARY_NARRATION)

    def test_uses_fine_alignment_before_balanced_fallback(self):
        narration = {
            "audio_duration": 6.6,
            "timing_source": "native",
            "segments": [
                {"index": 1, "start": 0.0, "end": 6.6, "text": "Sólo quedaban encendidas unas lámparas amarillentas y el zumbido constante del ascensor."},
            ],
            "alignment_units": [
                {"index": 1, "start": 0.0, "end": 0.7, "text": "Sólo"},
                {"index": 2, "start": 0.7, "end": 1.4, "text": "quedaban"},
                {"index": 3, "start": 1.4, "end": 2.2, "text": "encendidas"},
                {"index": 4, "start": 2.2, "end": 2.7, "text": "unas"},
                {"index": 5, "start": 2.7, "end": 3.5, "text": "lámparas amarillentas"},
                {"index": 6, "start": 3.5, "end": 3.8, "text": "y"},
                {"index": 7, "start": 3.8, "end": 4.6, "text": "el zumbido"},
                {"index": 8, "start": 4.6, "end": 5.4, "text": "constante"},
                {"index": 9, "start": 5.4, "end": 6.6, "text": "del ascensor."},
            ],
        }
        timeline = visual_timeline.build_visual_timeline(narration)
        plan = media_shot_plan.build_media_shot_plan(timeline, max_clip_duration=5.0)
        self.assertEqual(len(plan.shots), 2)
        self.assertEqual(plan.shots[0].end_boundary_source, media_shot_plan.BOUNDARY_ALIGNMENT)
        self.assertFalse(any(shot.uses_internal_estimate for shot in plan.shots))
        self.assertNotEqual(plan.shots[0].narration_text, plan.shots[1].narration_text)
        self.assertIn("lámparas amarillentas", plan.shots[0].narration_text)
        self.assertIn("zumbido", plan.shots[1].narration_text)

    def test_inserts_balanced_internal_cut_when_alignment_is_too_coarse(self):
        narration = {
            "audio_duration": 8.3,
            "timing_source": "native",
            "segments": [
                {"index": 1, "start": 0.0, "end": 8.3, "text": "Una sola unidad de narración demasiado larga para cinco segundos."}
            ],
        }
        timeline = visual_timeline.build_visual_timeline(narration)
        plan = media_shot_plan.build_media_shot_plan(timeline, max_clip_duration=5.0)
        self.assertEqual(len(plan.shots), 2)
        self.assertTrue(any(shot.uses_internal_estimate for shot in plan.shots))
        self.assertAlmostEqual(plan.shots[-1].end, 8.3)

    def test_preferred_cut_must_be_existing_validated_boundary(self):
        narration = {
            "audio_duration": 6.0,
            "timing_source": "native",
            "segments": [{"index": 1, "start": 0.0, "end": 6.0, "text": "uno dos tres cuatro cinco seis"}],
            "alignment_units": [
                {"index": i + 1, "start": float(i), "end": float(i + 1), "text": word}
                for i, word in enumerate("uno dos tres cuatro cinco seis".split())
            ],
        }
        timeline = visual_timeline.build_visual_timeline(narration)
        plan = media_shot_plan.build_media_shot_plan(
            timeline, max_clip_duration=4.0, preferred_scene_cuts={1: [2.0]}
        )
        self.assertAlmostEqual(plan.shots[0].end, 2.0)
        with self.assertRaises(ValueError):
            media_shot_plan.build_media_shot_plan(
                timeline, max_clip_duration=4.0, preferred_scene_cuts={1: [2.25]}
            )

    def test_shot_plan_is_contiguous_and_covers_full_audio(self):
        timeline = self._timeline()
        plan = media_shot_plan.build_media_shot_plan(timeline, max_clip_duration=3.0)
        self.assertAlmostEqual(plan.shots[0].start, 0.0)
        self.assertAlmostEqual(plan.shots[-1].end, 18.0)
        for previous, current in zip(plan.shots, plan.shots[1:]):
            self.assertAlmostEqual(previous.end, current.start)


    def test_preferred_minimum_merges_short_semantic_scene_when_safe(self):
        narration = {
            "audio_duration": 9.0,
            "timing_source": "native",
            "segments": [
                {"index": 1, "start": 0.0, "end": 4.0, "text": "Primera escena."},
                {"index": 2, "start": 4.0, "end": 5.0, "text": "Breve."},
                {"index": 3, "start": 5.0, "end": 9.0, "text": "Tercera escena."},
            ],
        }
        timeline = visual_timeline.build_visual_timeline(narration)
        plan = media_shot_plan.build_media_shot_plan(timeline, max_clip_duration=5.0)
        self.assertEqual(plan.min_clip_duration, 3.0)
        self.assertEqual(plan.undersized_shot_count, 0)
        self.assertEqual([round(shot.duration, 2) for shot in plan.shots], [5.0, 4.0])
        self.assertEqual(plan.shots[0].scene_indices, (1, 2))

    def test_fine_alignment_can_repartition_across_short_scenes(self):
        narration = {
            "audio_duration": 6.4,
            "timing_source": "native",
            "segments": [
                {"index": 1, "start": 0.0, "end": 2.4, "text": "Primera acción."},
                {"index": 2, "start": 2.4, "end": 4.1, "text": "Segunda acción."},
                {"index": 3, "start": 4.1, "end": 6.4, "text": "Tercera acción."},
            ],
            "alignment_units": [
                {"index": 1, "start": 0.0, "end": 1.0, "text": "Primera"},
                {"index": 2, "start": 1.0, "end": 2.4, "text": "acción."},
                {"index": 3, "start": 2.4, "end": 3.2, "text": "Segunda"},
                {"index": 4, "start": 3.2, "end": 4.1, "text": "acción."},
                {"index": 5, "start": 4.1, "end": 5.2, "text": "Tercera"},
                {"index": 6, "start": 5.2, "end": 6.4, "text": "acción."},
            ],
        }
        timeline = visual_timeline.build_visual_timeline(narration)
        plan = media_shot_plan.build_media_shot_plan(timeline, max_clip_duration=5.0)
        self.assertEqual(plan.undersized_shot_count, 0)
        self.assertEqual(len(plan.shots), 2)
        self.assertTrue(all(3.0 <= shot.duration <= 5.0 for shot in plan.shots))
        self.assertTrue(any(len(shot.scene_indices) > 1 for shot in plan.shots))

    def test_effective_minimum_never_exceeds_user_maximum(self):
        narration = {
            "audio_duration": 6.0,
            "timing_source": "native",
            "segments": [
                {"index": 1, "start": 0.0, "end": 6.0, "text": "uno dos tres cuatro cinco seis."}
            ],
            "alignment_units": [
                {"index": i + 1, "start": float(i), "end": float(i + 1), "text": word}
                for i, word in enumerate("uno dos tres cuatro cinco seis".split())
            ],
        }
        timeline = visual_timeline.build_visual_timeline(narration)
        plan = media_shot_plan.build_media_shot_plan(timeline, max_clip_duration=2.0)
        self.assertEqual(plan.min_clip_duration, 2.0)
        self.assertEqual(plan.undersized_shot_count, 0)
        self.assertTrue(all(abs(shot.duration - 2.0) < 1e-6 for shot in plan.shots))

    def test_preferred_minimum_is_soft_when_no_valid_repartition_exists(self):
        narration = {
            "audio_duration": 5.5,
            "timing_source": "estimated",
            "segments": [
                {"index": 1, "start": 0.0, "end": 5.5, "text": "Una unidad larga sin alineación fina."}
            ],
        }
        timeline = visual_timeline.build_visual_timeline(narration)
        plan = media_shot_plan.build_media_shot_plan(timeline, max_clip_duration=5.0)
        self.assertTrue(all(shot.duration <= 5.0 + 1e-6 for shot in plan.shots))
        self.assertGreater(plan.undersized_shot_count, 0)
        self.assertAlmostEqual(plan.shots[0].start, 0.0)
        self.assertAlmostEqual(plan.shots[-1].end, 5.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
