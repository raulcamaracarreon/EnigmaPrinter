import json
import unittest

from app.services import visual_shot_planner


class VisualShotPlannerTests(unittest.TestCase):
    def _base_plan(self):
        return {
            "shots": [
                {
                    "index": 1,
                    "scene_index": 1,
                    "scene_indices": [1],
                    "start": 0.0,
                    "end": 4.15,
                    "narration_indices": [1, 2, 3],
                    "narration_text": "Cada noche, poco después de las once, una mujer salía de su oficina.",
                    "end_boundary_source": "balanced_internal_cut",
                },
                {
                    "index": 2,
                    "scene_index": 1,
                    "scene_indices": [1],
                    "start": 4.15,
                    "end": 8.30,
                    "narration_indices": [3],
                    "narration_text": "una mujer salía de su oficina.",
                    "end_boundary_source": "scene_boundary",
                },
                {
                    "index": 3,
                    "scene_index": 2,
                    "scene_indices": [2],
                    "start": 8.30,
                    "end": 11.24,
                    "narration_indices": [4],
                    "narration_text": "El vestíbulo estaba casi vacío a esa hora.",
                    "end_boundary_source": "scene_boundary",
                },
            ]
        }

    def test_payload_marks_shared_narration_and_scene_continuation(self):
        payload = visual_shot_planner.build_visual_shot_payload(self._base_plan())
        self.assertFalse(payload[0]["shared_narration_with_previous"])
        self.assertTrue(payload[1]["shared_narration_with_previous"])
        self.assertTrue(payload[1]["continues_same_scene"])
        self.assertFalse(payload[2]["shared_narration_with_previous"])
        self.assertFalse(payload[2]["continues_same_scene"])

    def test_generate_plan_preserves_locked_timing_and_order(self):
        response = json.dumps(
            {
                "shots": [
                    {"shot_index": 1, "visual_prompt": "Establishing view of the dim office corridor late at night."},
                    {"shot_index": 2, "visual_prompt": "The woman leaves the office and walks toward the vintage elevator."},
                    {"shot_index": 3, "visual_prompt": "Nearly empty lobby under weak yellow lamps, elevator doors in the distance."},
                ]
            }
        )
        plan = visual_shot_planner.generate_visual_shot_plan(
            self._base_plan(), response_generator=lambda prompt: response
        )
        data = plan.to_dict()
        self.assertEqual(data["source_shot_count"], 3)
        self.assertEqual(data["shared_narration_continuation_count"], 1)
        self.assertEqual(data["shots"][0]["start"], 0.0)
        self.assertEqual(data["shots"][1]["end"], 8.3)
        self.assertEqual(data["shots"][1]["narration_indices"], [3])
        self.assertEqual(len(data["visual_prompts"]), 3)

    def test_rejects_missing_shot(self):
        response = json.dumps(
            {"shots": [{"shot_index": 1, "visual_prompt": "One useful prompt."}]}
        )
        with self.assertRaisesRegex(ValueError, "expected 3"):
            visual_shot_planner.generate_visual_shot_plan(
                self._base_plan(), response_generator=lambda prompt: response
            )

    def test_rejects_duplicate_visual_prompts(self):
        response = json.dumps(
            {
                "shots": [
                    {"shot_index": 1, "visual_prompt": "The woman walks toward the elevator."},
                    {"shot_index": 2, "visual_prompt": "The woman walks toward the elevator."},
                    {"shot_index": 3, "visual_prompt": "The empty lobby is visible."},
                ]
            }
        )
        with self.assertRaisesRegex(ValueError, "identical"):
            visual_shot_planner.generate_visual_shot_plan(
                self._base_plan(), response_generator=lambda prompt: response
            )

    def test_response_cannot_change_timestamps(self):
        response = json.dumps(
            {
                "shots": [
                    {"shot_index": 1, "start": 99, "end": 100, "visual_prompt": "Wide view of a dark office floor."},
                    {"shot_index": 2, "start": 100, "end": 101, "visual_prompt": "The woman approaches the elevator doors."},
                    {"shot_index": 3, "start": 101, "end": 102, "visual_prompt": "The empty lobby waits below cold light."},
                ]
            }
        )
        plan = visual_shot_planner.generate_visual_shot_plan(
            self._base_plan(), response_generator=lambda prompt: response
        )
        self.assertEqual(plan.shots[0].start, 0.0)
        self.assertEqual(plan.shots[0].end, 4.15)
        self.assertEqual(plan.shots[2].end, 11.24)

    def test_prompt_contains_overlap_instruction(self):
        captured = {}

        def responder(prompt):
            captured["prompt"] = prompt
            return json.dumps(
                {
                    "shots": [
                        {"shot_index": 1, "visual_prompt": "Wide office corridor at night."},
                        {"shot_index": 2, "visual_prompt": "Closer view of the woman entering the elevator."},
                        {"shot_index": 3, "visual_prompt": "Empty lobby beneath weak lamps."},
                    ]
                }
            )

        visual_shot_planner.generate_visual_shot_plan(
            self._base_plan(), response_generator=responder
        )
        self.assertIn("shared_narration_with_previous", captured["prompt"])
        self.assertIn("Do not repeat the previous visual composition", captured["prompt"])


if __name__ == "__main__":
    unittest.main()
