import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import llm, material


class TestComfyUIVideoRegressions(unittest.TestCase):
    def test_ltx25_primitive_prompt_duration_and_conditioning_are_injected(self):
        """LTX-2.5 style API graphs keep prompt/duration in Primitive nodes."""
        workflow = {
            "conditioning": {
                "class_type": "LTXVConditioning",
                "inputs": {
                    "positive": ["positive_encoder", 0],
                    "negative": ["negative_encoder", 0],
                },
            },
            "positive_encoder": {
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "CLIP Text Encode (Prompt)"},
                "inputs": {"text": ["switch", 0]},
            },
            "negative_encoder": {
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "CLIP Text Encode (Prompt)"},
                "inputs": {"text": "old negative"},
            },
            "switch": {
                "class_type": "ComfySwitchNode",
                "_meta": {"title": "If/Else Switch"},
                "inputs": {
                    "switch": ["enhance_toggle", 0],
                    "on_false": ["prompt_primitive", 0],
                    "on_true": ["prompt_enhancer", 0],
                },
            },
            "prompt_primitive": {
                "class_type": "PrimitiveStringMultiline",
                "_meta": {"title": "Prompt"},
                "inputs": {"value": "Man running in a forest at dusk"},
            },
            "prompt_enhancer": {
                "class_type": "TextGenerateLTX2Prompt",
                "_meta": {"title": "Generate LTX2 Prompt"},
                "inputs": {"prompt": ["prompt_primitive", 0]},
            },
            "enhance_toggle": {
                "class_type": "PrimitiveBoolean",
                "inputs": {"value": False},
            },
            "duration": {
                "class_type": "PrimitiveInt",
                "_meta": {"title": "Duration"},
                "inputs": {"value": 3},
            },
            "noise_a": {
                "class_type": "RandomNoise",
                "inputs": {"noise_seed": 42},
            },
            "noise_b": {
                "class_type": "RandomNoise",
                "inputs": {"noise_seed": 123456},
            },
            "save": {
                "class_type": "SaveVideo",
                "_meta": {"title": "Save Video"},
                "inputs": {"filename_prefix": "video/LTX_2.5_t2v"},
            },
        }

        with (
            patch.object(
                material, "_load_comfyui_video_workflow", return_value=workflow
            ),
            patch.object(
                material,
                "_comfyui_video_prompt",
                return_value="A woman crossing a colonial plaza at night",
            ),
            patch.object(
                material,
                "_comfyui_video_negative_prompt",
                return_value="text, watermark",
            ),
            patch.object(
                material,
                "is_comfyui_video_resolution_override_enabled",
                return_value=False,
            ),
        ):
            prepared = material._prepare_comfyui_video_workflow(
                search_term="unused",
                width=1280,
                height=720,
                target_duration=5,
            )

        self.assertEqual(
            prepared["prompt_primitive"]["inputs"]["value"],
            "A woman crossing a colonial plaza at night",
        )
        self.assertEqual(
            prepared["negative_encoder"]["inputs"]["text"],
            "text, watermark",
        )
        self.assertEqual(prepared["duration"]["inputs"]["value"], 5)
        self.assertNotEqual(prepared["noise_a"]["inputs"]["noise_seed"], 42)
        self.assertNotEqual(
            prepared["noise_b"]["inputs"]["noise_seed"], 123456
        )
        self.assertTrue(
            prepared["save"]["inputs"]["filename_prefix"].startswith(
                "mpt_comfyui_video_"
            )
        )

    def test_video_output_extractor_supports_vhs_mp4_history(self):
        """VHS_VideoCombine commonly exposes MP4 files through the `gifs` key."""
        payload = {
            "42": {
                "gifs": [
                    {
                        "filename": "mpt_test_00001-audio.mp4",
                        "subfolder": "",
                        "type": "output",
                        "format": "video/h264-mp4",
                    }
                ]
            }
        }

        result = material._extract_first_comfyui_video(payload)

        self.assertIsNotNone(result)
        self.assertEqual(result["filename"], "mpt_test_00001-audio.mp4")

    def test_video_workflow_injects_common_prompt_size_seed_and_seconds(self):
        """Common T2V fields are updated without fixed ComfyUI node IDs."""
        workflow = {
            "1": {
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "Positive Prompt"},
                "inputs": {"text": "old positive"},
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "Negative Prompt"},
                "inputs": {"text": "old negative"},
            },
            "3": {
                "class_type": "EmptyVideoLatent",
                "inputs": {
                    "width": 512,
                    "height": 512,
                    "duration_seconds": 5,
                },
            },
            "4": {
                "class_type": "RandomNoise",
                "inputs": {"noise_seed": 123},
            },
            "5": {
                "class_type": "VHS_VideoCombine",
                "inputs": {"filename_prefix": "old_prefix"},
            },
        }

        with (
            patch.object(
                material, "_load_comfyui_video_workflow", return_value=workflow
            ),
            patch.object(
                material,
                "_comfyui_video_prompt",
                return_value="Man running in a forest at dusk",
            ),
            patch.object(
                material,
                "_comfyui_video_negative_prompt",
                return_value="text, watermark",
            ),
            patch.object(
                material,
                "is_comfyui_video_resolution_override_enabled",
                return_value=True,
            ),
        ):
            prepared = material._prepare_comfyui_video_workflow(
                search_term="unused",
                width=1280,
                height=720,
                target_duration=3,
            )

        self.assertEqual(
            prepared["1"]["inputs"]["text"],
            "Man running in a forest at dusk",
        )
        self.assertEqual(prepared["2"]["inputs"]["text"], "text, watermark")
        self.assertEqual(prepared["3"]["inputs"]["width"], 1280)
        self.assertEqual(prepared["3"]["inputs"]["height"], 720)
        self.assertEqual(prepared["3"]["inputs"]["duration_seconds"], 3)
        self.assertNotEqual(prepared["4"]["inputs"]["noise_seed"], 123)
        self.assertTrue(
            prepared["5"]["inputs"]["filename_prefix"].startswith(
                "mpt_comfyui_video_"
            )
        )

    def test_video_workflow_does_not_guess_model_specific_frame_count(self):
        """Frame-count-only workflows keep their model-specific length untouched."""
        workflow = {
            "1": {
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "Positive Prompt"},
                "inputs": {"text": "old prompt"},
            },
            "2": {
                "class_type": "EmptyVideoLatent",
                "inputs": {
                    "width": 768,
                    "height": 432,
                    "num_frames": 81,
                    "fps": 16,
                },
            },
        }

        with (
            patch.object(
                material, "_load_comfyui_video_workflow", return_value=workflow
            ),
            patch.object(
                material,
                "_comfyui_video_prompt",
                return_value="test prompt",
            ),
            patch.object(
                material,
                "_comfyui_video_negative_prompt",
                return_value="",
            ),
        ):
            prepared = material._prepare_comfyui_video_workflow(
                search_term="unused",
                width=1280,
                height=720,
                target_duration=3,
            )

        self.assertEqual(prepared["2"]["inputs"]["num_frames"], 81)
        self.assertEqual(prepared["2"]["inputs"]["fps"], 16)
        self.assertEqual(prepared["2"]["inputs"]["width"], 768)
        self.assertEqual(prepared["2"]["inputs"]["height"], 432)

    def test_video_generation_stops_after_actual_clips_cover_narration(self):
        """Local T2V stops early once real generated durations cover the audio."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            clip_paths = []
            for index in range(3):
                clip_path = Path(tmp_dir) / f"clip-{index}.mp4"
                clip_path.write_bytes(b"fake")
                clip_paths.append(str(clip_path))

            generated_items = []
            for index, clip_path in enumerate(clip_paths):
                item = material.MaterialInfo(
                    provider="comfyui_video",
                    url=clip_path,
                    duration=3,
                    source_info={
                        "provider": "comfyui_video",
                        "search_term": f"term {index}",
                        "actual_duration": 3.0,
                        "rendition": {"width": 1280, "height": 720},
                    },
                )
                generated_items.append([item])

            with (
                patch.object(
                    material,
                    "generate_videos_comfyui_video",
                    side_effect=generated_items,
                ) as generate,
                patch.object(material, "_persist_material_sources"),
            ):
                result = material._download_videos_comfyui_video_on_demand(
                    task_id="test-task",
                    search_terms=["one", "two", "three"],
                    video_aspect=material.VideoAspect.landscape,
                    audio_duration=5.0,
                    max_clip_duration=3,
                    material_directory=tmp_dir,
                )

        self.assertEqual(result, clip_paths[:2])
        self.assertEqual(generate.call_count, 2)

    def test_timeline_video_generation_runs_every_locked_shot_in_order(self):
        """Timeline T2V never stops early after aggregate coverage is reached."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            clip_paths = []
            generated_items = []
            for index, actual_duration in enumerate((5.0, 5.0, 3.0), start=1):
                clip_path = Path(tmp_dir) / f"timeline-{index}.mp4"
                clip_path.write_bytes(b"fake")
                clip_paths.append(str(clip_path))
                item = material.MaterialInfo(
                    provider="comfyui_video",
                    url=str(clip_path),
                    duration=int(actual_duration),
                    source_info={
                        "provider": "comfyui_video",
                        "search_term": f"shot {index}",
                        "actual_duration": actual_duration,
                        "rendition": {"width": 1280, "height": 720},
                    },
                )
                generated_items.append([item])

            timeline = [
                {"index": 1, "start": 0.0, "end": 4.15},
                {"index": 2, "start": 4.15, "end": 8.30},
                {"index": 3, "start": 8.30, "end": 11.24},
            ]

            with (
                patch.object(
                    material,
                    "generate_videos_comfyui_video",
                    side_effect=generated_items,
                ) as generate,
                patch.object(material, "_persist_material_sources"),
            ):
                result = material._download_comfyui_video_with_timeline(
                    task_id="test-task",
                    search_terms=["one", "two", "three"],
                    video_aspect=material.VideoAspect.landscape,
                    audio_duration=11.24,
                    max_clip_duration=5,
                    clip_speed=1.0,
                    shot_timeline=timeline,
                    material_directory=tmp_dir,
                )

        self.assertEqual(result, clip_paths)
        self.assertEqual(generate.call_count, 3)
        self.assertEqual(
            [call.kwargs["minimum_duration"] for call in generate.call_args_list],
            [5, 5, 3],
        )

    def test_timeline_video_generation_rejects_source_too_short_for_locked_shot(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            clip_path = Path(tmp_dir) / "short.mp4"
            clip_path.write_bytes(b"fake")
            item = material.MaterialInfo(
                provider="comfyui_video",
                url=str(clip_path),
                duration=4,
                source_info={
                    "provider": "comfyui_video",
                    "search_term": "shot",
                    "actual_duration": 4.0,
                    "rendition": {"width": 1280, "height": 720},
                },
            )
            with (
                patch.object(
                    material,
                    "generate_videos_comfyui_video",
                    return_value=[item],
                ),
                patch.object(material, "_persist_material_sources"),
            ):
                with self.assertRaisesRegex(
                    material.timeline_media.TimelineMediaError,
                    r"too short.*shot=1",
                ):
                    material._download_comfyui_video_with_timeline(
                        task_id="test-task",
                        search_terms=["one"],
                        video_aspect=material.VideoAspect.landscape,
                        audio_duration=4.15,
                        max_clip_duration=5,
                        clip_speed=1.0,
                        shot_timeline=[{"index": 1, "start": 0.0, "end": 4.15}],
                        material_directory=tmp_dir,
                    )


    def test_subject_anchors_expand_multiple_visual_entity_types(self):
        """Scene Prompts expand only the reusable subjects referenced by each scene."""
        anchors = """
[CHAR_1]
A lean man in his early thirties with short dark hair and a dark olive jacket

[LOC_1]
An abandoned wooden cabin in a misty forest clearing

[OBJ_1]
An old brass lantern with weak flickering light

[VEH_1]
A weathered red pickup truck with a cracked windshield

[CREATURE_1]
A large black wolf with a pale scar over its left eye
"""
        scenes = """
[CHAR_1] runs toward [LOC_1]
[CHAR_1] raises [OBJ_1] beside [VEH_1]
[CREATURE_1] watches [CHAR_1] from the tree line
"""

        expanded = material.expand_comfyui_video_scene_prompts(
            scene_prompts=scenes,
            subject_anchors=anchors,
        )

        self.assertEqual(len(expanded), 3)
        self.assertIn("A lean man in his early thirties", expanded[0])
        self.assertIn("An abandoned wooden cabin", expanded[0])
        self.assertNotIn("[CHAR_1]", expanded[0])
        self.assertIn("An old brass lantern", expanded[1])
        self.assertIn("weathered red pickup truck", expanded[1])
        self.assertIn("large black wolf", expanded[2])

    def test_scene_prompt_rejects_undefined_subject_anchor(self):
        """Unknown anchor placeholders fail before an expensive T2V inference."""
        with self.assertRaisesRegex(
            ValueError,
            r"undefined Subject Anchor.*\[CHAR_2\]",
        ):
            material.expand_comfyui_video_scene_prompts(
                scene_prompts="[CHAR_1] sees [CHAR_2] near the cabin",
                subject_anchors="""
[CHAR_1]
A man in a dark olive jacket
""",
            )

    def test_duplicate_subject_anchor_is_rejected(self):
        """Ambiguous duplicate anchor definitions are rejected explicitly."""
        with self.assertRaisesRegex(
            ValueError,
            r"\[CHAR_1\].*defined more than once",
        ):
            material.parse_comfyui_video_subject_anchors(
                """
[CHAR_1]
First description
[CHAR_1]
Second description
"""
            )

    def test_video_generation_rejects_insufficient_scene_prompts_before_inference(self):
        """T2V preflight prevents an unavoidable final-loop before inference starts."""
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            patch.object(
                material,
                "generate_videos_comfyui_video",
                return_value=[],
            ) as generate,
        ):
            with self.assertRaisesRegex(
                material.ComfyUIVideoError,
                r"at least 6 Scene Prompts.*only 3 were provided",
            ):
                material._download_videos_comfyui_video_on_demand(
                    task_id="test-task",
                    search_terms=["one", "two", "three"],
                    video_aspect=material.VideoAspect.landscape,
                    audio_duration=16.08,
                    max_clip_duration=3,
                    material_directory=tmp_dir,
                )

        self.assertEqual(generate.call_count, 0)



class TestComfyUIVisualContinuityLLM(unittest.TestCase):
    def test_anchor_plan_preserves_scene_count_order_and_formats_blocks(self):
        response = {
            "subject_anchors": [
                {
                    "tag": "CHAR_1",
                    "description": "A young radio technician with short dark hair and a gray work shirt",
                },
                {
                    "tag": "LOC_1",
                    "description": "A small isolated radio station with vintage analog equipment",
                },
                {
                    "tag": "OBJ_1",
                    "description": "An old analog radio receiver with brass dials and an amber needle",
                },
            ],
            "scene_prompts": [
                {"index": 1, "prompt": "[CHAR_1] works alone inside [LOC_1] at midnight"},
                {"index": 2, "prompt": "[CHAR_1] tunes [OBJ_1] inside [LOC_1]"},
                {"index": 3, "prompt": "[OBJ_1] glows inside the darkened [LOC_1]"},
            ],
        }
        with patch.object(llm, "_generate_response", return_value=json.dumps(response)):
            plan = llm.generate_subject_anchors_and_scene_prompts(
                visual_prompts=["scene one", "scene two", "scene three"],
                video_subject="A mysterious radio station",
            )

        self.assertEqual(plan["scene_count"], 3)
        self.assertEqual(plan["anchor_count"], 3)
        self.assertEqual(len(plan["scene_prompts"]), 3)
        self.assertTrue(plan["scene_prompts"][0].startswith("[CHAR_1]"))
        self.assertIn("[CHAR_1]\nA young radio technician", plan["subject_anchors"])
        self.assertIn("[LOC_1]\nA small isolated radio station", plan["subject_anchors"])
        self.assertIn("[OBJ_1]\nAn old analog radio receiver", plan["subject_anchors"])

    def test_anchor_plan_retries_when_scene_count_changes(self):
        invalid = {
            "subject_anchors": [],
            "scene_prompts": [{"index": 1, "prompt": "only one scene"}],
        }
        valid = {
            "subject_anchors": [],
            "scene_prompts": [
                {"index": 1, "prompt": "scene one"},
                {"index": 2, "prompt": "scene two"},
            ],
        }
        with patch.object(
            llm,
            "_generate_response",
            side_effect=[json.dumps(invalid), json.dumps(valid)],
        ) as generate:
            plan = llm.generate_subject_anchors_and_scene_prompts(
                visual_prompts="scene one\nscene two"
            )

        self.assertEqual(plan["scene_count"], 2)
        self.assertEqual(generate.call_count, 2)

    def test_anchor_plan_retries_on_undefined_reference(self):
        invalid = {
            "subject_anchors": [
                {"tag": "CHAR_1", "description": "A technician in a gray work shirt"}
            ],
            "scene_prompts": [
                {"index": 1, "prompt": "[CHAR_2] enters the station"}
            ],
        }
        valid = {
            "subject_anchors": [
                {"tag": "CHAR_1", "description": "A technician in a gray work shirt"}
            ],
            "scene_prompts": [
                {"index": 1, "prompt": "[CHAR_1] enters the station"}
            ],
        }
        with patch.object(
            llm,
            "_generate_response",
            side_effect=[json.dumps(invalid), json.dumps(valid)],
        ) as generate:
            plan = llm.generate_subject_anchors_and_scene_prompts(
                visual_prompts=["A technician enters the station"]
            )

        self.assertEqual(plan["scene_count"], 1)
        self.assertEqual(generate.call_count, 2)
        self.assertIn("[CHAR_1]", plan["scene_prompts"][0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
