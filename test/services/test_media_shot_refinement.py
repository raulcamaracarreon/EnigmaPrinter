import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import media_shot_plan, media_shot_refinement, visual_timeline


class TestMediaShotRefinement(unittest.TestCase):
    def _timeline_and_plan(self):
        narration = {
            "audio_duration": 6.6,
            "timing_source": "native",
            "segments": [
                {"index": 1, "start": 0.0, "end": 6.6, "text": "Sólo quedaban encendidas unas lámparas amarillentas y el zumbido constante del ascensor."}
            ],
            "alignment_units": [
                {"index": 1, "start": 0.0, "end": 0.8, "text": "Sólo"},
                {"index": 2, "start": 0.8, "end": 1.6, "text": "quedaban"},
                {"index": 3, "start": 1.6, "end": 2.4, "text": "encendidas"},
                {"index": 4, "start": 2.4, "end": 3.0, "text": "unas"},
                {"index": 5, "start": 3.0, "end": 3.6, "text": "lámparas amarillentas"},
                {"index": 6, "start": 3.6, "end": 3.9, "text": "y"},
                {"index": 7, "start": 3.9, "end": 4.7, "text": "el zumbido"},
                {"index": 8, "start": 4.7, "end": 5.5, "text": "constante"},
                {"index": 9, "start": 5.5, "end": 6.6, "text": "del ascensor."},
            ],
        }
        timeline = visual_timeline.build_visual_timeline(narration)
        plan = media_shot_plan.build_media_shot_plan(timeline, max_clip_duration=5.0)
        return timeline, plan

    def test_candidates_are_existing_boundaries_only(self):
        timeline, plan = self._timeline_and_plan()
        candidates = media_shot_refinement.build_refinement_candidates(timeline, plan)
        self.assertEqual(len(candidates), 1)
        scene = timeline.to_dict()["segments"][0]
        valid = set(scene["alignment_boundaries"]) | set(scene["narration_boundaries"])
        for candidate in candidates[0]["candidates"]:
            self.assertIn(candidate["boundary"], valid)

    def test_llm_can_select_candidate_id_but_not_timestamp(self):
        timeline, plan = self._timeline_and_plan()
        candidates = media_shot_refinement.build_refinement_candidates(timeline, plan)
        chosen = candidates[0]["candidates"][-1]

        def responder(_prompt):
            return json.dumps({"choices": [{"scene_index": 1, "boundary_id": chosen["id"]}]})

        result = media_shot_refinement.refine_media_shot_plan(
            timeline, plan, response_generator=responder
        )
        self.assertEqual(result.refined_scene_count, 1)
        self.assertAlmostEqual(result.plan.shots[0].end, chosen["boundary"])
        self.assertTrue(all(shot.duration <= 5.0 + 1e-6 for shot in result.plan.shots))

    def test_unknown_choice_is_rejected_and_deterministic_plan_survives(self):
        timeline, plan = self._timeline_and_plan()

        def responder(_prompt):
            return '{"choices":[{"scene_index":1,"boundary_id":"INVENTED_TIME"}]}'

        result = media_shot_refinement.refine_media_shot_plan(
            timeline, plan, response_generator=responder
        )
        self.assertEqual(result.refined_scene_count, 0)
        self.assertEqual(result.rejected_choice_count, 1)
        self.assertEqual(result.plan.to_dict(), plan.to_dict())


    def test_refinement_candidates_respect_preferred_minimum(self):
        timeline, plan = self._timeline_and_plan()
        candidates = media_shot_refinement.build_refinement_candidates(timeline, plan)
        self.assertTrue(candidates)
        minimum = plan.min_clip_duration
        for scene in candidates:
            for candidate in scene["candidates"]:
                self.assertGreaterEqual(candidate["left_duration"] + 1e-6, minimum)
                self.assertGreaterEqual(candidate["right_duration"] + 1e-6, minimum)


if __name__ == "__main__":
    unittest.main(verbosity=2)
