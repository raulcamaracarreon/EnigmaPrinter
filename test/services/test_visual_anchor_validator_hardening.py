import json
import unittest
from unittest.mock import patch

from app.services import llm, material


class TestVisualAnchorValidatorHardening(unittest.TestCase):
    def setUp(self):
        # These tests exercise Stages 1-3 in isolation.
        # Stage 4 has its own tests below.
        self.expand_patcher = patch.object(
            llm,
            "_expand_visual_anchor_scene_prompts",
            side_effect=lambda scene_prompts, anchors: list(
                scene_prompts
            ),
        )
        self.refine_patcher = patch.object(
            llm,
            "refine_expanded_scene_prompts",
            side_effect=lambda expanded_prompts, **kwargs: list(
                expanded_prompts
            ),
        )
        self.expand_patcher.start()
        self.refine_patcher.start()

    def tearDown(self):
        self.refine_patcher.stop()
        self.expand_patcher.stop()

    def test_possessive_anchor_reference_is_allowed_and_expands_cleanly(self):
        for possessive in (
            "[CHAR_1]'s hand grips a phone.",
            "[CHAR_1]\u2019s hand grips a phone.",
        ):
            with self.subTest(possessive=possessive):
                data = {
                    "subject_anchors": [
                        {
                            "tag": "CHAR_1",
                            "description": "a dark-haired man",
                        }
                    ],
                    "scene_prompts": [
                        {
                            "index": 1,
                            "prompt": possessive,
                        }
                    ],
                }

                plan = llm._normalize_visual_anchor_plan(
                    data,
                    expected_scene_count=1,
                )

                self.assertEqual(
                    plan["scene_prompts"],
                    [possessive],
                )

        expanded = material.expand_comfyui_video_scene_prompts(
            scene_prompts="[CHAR_1]'s hand grips a phone.",
            subject_anchors=(
                "[CHAR_1]\n"
                "a dark-haired man"
            ),
        )

        self.assertEqual(
            expanded,
            [
                "a dark-haired man's hand grips a phone."
            ],
        )

    def test_valid_nonpossessive_anchor_reference_passes(self):
        data = {
            "subject_anchors": [
                {
                    "tag": "CHAR_1",
                    "description": "a dark-haired man",
                },
                {
                    "tag": "OBJ_1",
                    "description": "an antique red telephone",
                },
            ],
            "scene_prompts": [
                {
                    "index": 1,
                    "prompt": (
                        "[CHAR_1] grips [OBJ_1] with one hand."
                    ),
                }
            ],
        }

        plan = llm._normalize_visual_anchor_plan(
            data,
            expected_scene_count=1,
        )

        self.assertEqual(plan["scene_count"], 1)
        self.assertEqual(plan["anchor_count"], 2)

    def test_invalid_global_anchor_is_retried(self):
        responses = [
            json.dumps(
                {
                    "subject_anchors": [
                        {
                            "tag": "BAD_1",
                            "description": "an invalid anchor",
                        }
                    ]
                }
            ),
            json.dumps(
                {
                    "subject_anchors": [
                        {
                            "tag": "CHAR_1",
                            "description": "a dark-haired man",
                        }
                    ]
                }
            ),
            json.dumps(
                {
                    "scene_prompts": [
                        {
                            "index": 1,
                            "prompt": "[CHAR_1] stands beside a window.",
                        }
                    ]
                }
            ),
        ]

        with patch.object(
            llm,
            "_generate_response",
            side_effect=responses,
        ) as generate:
            plan = llm.generate_subject_anchors_and_scene_prompts(
                ["A dark-haired man stands beside a window."]
            )

        self.assertEqual(generate.call_count, 3)
        self.assertEqual(plan["anchor_count"], 1)
        self.assertEqual(plan["scene_count"], 1)

    def test_scene_rewrite_batches_share_one_global_anchor_library(self):
        visual_prompts = [
            f"A dark-haired man appears in visual scene {index}."
            for index in range(1, 11)
        ]

        anchor_response = json.dumps(
            {
                "subject_anchors": [
                    {
                        "tag": "CHAR_1",
                        "description": "a dark-haired man",
                    }
                ]
            }
        )

        first_batch_response = json.dumps(
            {
                "scene_prompts": [
                    {
                        "index": index,
                        "prompt": (
                            f"[CHAR_1] appears in rewritten scene {index}."
                        ),
                    }
                    for index in range(1, 9)
                ]
            }
        )

        second_batch_response = json.dumps(
            {
                "scene_prompts": [
                    {
                        "index": index,
                        "prompt": (
                            f"[CHAR_1] appears in rewritten scene {index}."
                        ),
                    }
                    for index in range(9, 11)
                ]
            }
        )

        with patch.object(
            llm,
            "_generate_response",
            side_effect=[
                anchor_response,
                first_batch_response,
                second_batch_response,
            ],
        ) as generate:
            plan = llm.generate_subject_anchors_and_scene_prompts(
                visual_prompts
            )

        self.assertEqual(generate.call_count, 3)
        self.assertEqual(plan["anchor_count"], 1)
        self.assertEqual(plan["scene_count"], 10)

        self.assertEqual(
            plan["scene_prompts"][0],
            "[CHAR_1] appears in rewritten scene 1.",
        )
        self.assertEqual(
            plan["scene_prompts"][8],
            "[CHAR_1] appears in rewritten scene 9.",
        )

        prompts_sent = [
            call.args[0]
            for call in generate.call_args_list
        ]

        self.assertIn('"tag": "CHAR_1"', prompts_sent[1])
        self.assertIn('"tag": "CHAR_1"', prompts_sent[2])
        self.assertIn('"index": 9', prompts_sent[2])
        self.assertNotIn('"index": 1,', prompts_sent[2])

    def test_only_failed_scene_batch_is_repaired(self):
        visual_prompts = [
            f"A dark-haired man appears in visual scene {index}."
            for index in range(1, 11)
        ]

        anchor_response = json.dumps(
            {
                "subject_anchors": [
                    {
                        "tag": "CHAR_1",
                        "description": "a dark-haired man",
                    },
                    {
                        "tag": "OBJ_1",
                        "description": "an antique red telephone",
                    },
                ]
            }
        )

        first_batch_response = json.dumps(
            {
                "scene_prompts": [
                    {
                        "index": index,
                        "prompt": (
                            f"[CHAR_1] appears in rewritten "
                            f"scene {index}."
                        ),
                    }
                    for index in range(1, 9)
                ]
            }
        )

        second_batch_invalid = json.dumps(
            {
                "scene_prompts": [
                    {
                        "index": 9,
                        "prompt": (
                            "[CHAR_1] touches [OBJ_9]."
                        ),
                    },
                    {
                        "index": 10,
                        "prompt": (
                            "[CHAR_1] turns toward the doorway."
                        ),
                    },
                ]
            }
        )

        second_batch_repaired = json.dumps(
            {
                "scene_prompts": [
                    {
                        "index": 9,
                        "prompt": (
                            "[CHAR_1] touches [OBJ_1]."
                        ),
                    },
                    {
                        "index": 10,
                        "prompt": (
                            "[CHAR_1] turns toward the doorway."
                        ),
                    },
                ]
            }
        )

        with patch.object(
            llm,
            "_generate_response",
            side_effect=[
                anchor_response,
                first_batch_response,
                second_batch_invalid,
                second_batch_repaired,
            ],
        ) as generate:
            plan = llm.generate_subject_anchors_and_scene_prompts(
                visual_prompts
            )

        self.assertEqual(generate.call_count, 4)
        self.assertEqual(plan["scene_count"], 10)

        self.assertEqual(
            plan["scene_prompts"][8],
            "[CHAR_1] touches [OBJ_1].",
        )

        first_batch_prompt = (
            generate.call_args_list[1].args[0]
        )
        second_batch_prompt = (
            generate.call_args_list[2].args[0]
        )
        repair_prompt = (
            generate.call_args_list[3].args[0]
        )

        self.assertIn(
            '"index": 1',
            first_batch_prompt,
        )
        self.assertIn(
            '"index": 9',
            second_batch_prompt,
        )
        self.assertNotIn(
            "Scene Prompt Repair Editor",
            second_batch_prompt,
        )
        self.assertIn(
            "Scene Prompt Repair Editor",
            repair_prompt,
        )
        self.assertIn(
            "[OBJ_9]",
            repair_prompt,
        )
        self.assertNotIn(
            '"index": 1,',
            repair_prompt,
        )

    def test_bad_batch_is_retried_and_corrected(self):
        responses = [
            json.dumps(
                {
                    "subject_anchors": [
                        {
                            "tag": "CHAR_1",
                            "description": "a dark-haired man",
                        },
                        {
                            "tag": "OBJ_1",
                            "description": "an antique red telephone",
                        },
                    ]
                }
            ),
            json.dumps(
                {
                    "scene_prompts": [
                        {
                            "index": 1,
                            "prompt": (
                                "[CHAR_1] grips [OBJ_9] by hand."
                            ),
                        }
                    ]
                }
            ),
            json.dumps(
                {
                    "scene_prompts": [
                        {
                            "index": 1,
                            "prompt": (
                                "[CHAR_1] grips [OBJ_1] by hand."
                            ),
                        }
                    ]
                }
            ),
        ]

        with patch.object(
            llm,
            "_generate_response",
            side_effect=responses,
        ) as generate:
            plan = llm.generate_subject_anchors_and_scene_prompts(
                [
                    "A dark-haired man holds "
                    "an antique red telephone."
                ]
            )

        self.assertEqual(generate.call_count, 3)
        self.assertEqual(plan["scene_count"], 1)
        self.assertEqual(plan["anchor_count"], 2)
        self.assertEqual(
            plan["scene_prompts"],
            ["[CHAR_1] grips [OBJ_1] by hand."],
        )

        initial_scene_prompt = (
            generate.call_args_list[1].args[0]
        )
        repair_prompt = (
            generate.call_args_list[2].args[0]
        )

        self.assertNotEqual(
            initial_scene_prompt,
            repair_prompt,
        )
        self.assertIn(
            "Scene Prompt Repair Editor",
            repair_prompt,
        )
        self.assertIn(
            "Validation Error",
            repair_prompt,
        )
        self.assertIn(
            "[OBJ_9]",
            repair_prompt,
        )
        self.assertIn(
            "undefined",
            repair_prompt,
        )


class TestExpandedScenePromptRefinement(unittest.TestCase):

    def test_generation_preserves_tagged_scene_prompts_for_human_editing(self):
        anchor_response = json.dumps(
            {
                "subject_anchors": [
                    {
                        "tag": "CHAR_1",
                        "description": (
                            "a man named Gabriel "
                            "with short dark hair"
                        ),
                    },
                    {
                        "tag": "OBJ_1",
                        "description": (
                            "an old red rotary telephone"
                        ),
                    },
                ]
            }
        )

        scene_response = json.dumps(
            {
                "scene_prompts": [
                    {
                        "index": 1,
                        "prompt": (
                            "[CHAR_1]'s hand reaches "
                            "for [OBJ_1]."
                        ),
                    }
                ]
            }
        )

        with patch.object(
            llm,
            "_generate_response",
            side_effect=[
                anchor_response,
                scene_response,
            ],
        ) as generate:
            plan = (
                llm.generate_subject_anchors_and_scene_prompts(
                    [
                        "Gabriel reaches for an old red "
                        "rotary telephone."
                    ]
                )
            )

        self.assertEqual(
            generate.call_count,
            2,
        )

        self.assertEqual(
            plan["scene_prompts"],
            [
                "[CHAR_1]'s hand reaches for [OBJ_1]."
            ],
        )

        self.assertIn(
            "[CHAR_1]",
            plan["subject_anchors"],
        )

        self.assertIn(
            "[OBJ_1]",
            plan["subject_anchors"],
        )

    def test_refinement_falls_back_to_expanded_prompts(self):
        original = [
            (
                "a man named Gabriel with short dark hair, "
                "black jacket and blue jeans's hand reaches "
                "for an old red rotary telephone."
            )
        ]

        with patch.object(
            llm,
            "_generate_response",
            return_value=(
                "Error: local model unavailable"
            ),
        ) as generate:
            result = (
                llm.refine_expanded_scene_prompts(
                    original
                )
            )

        self.assertEqual(
            generate.call_count,
            2,
        )

        self.assertEqual(
            result,
            original,
        )

    def test_refinement_batches_more_than_eight_prompts(self):
        prompts = [
            f"Expanded visual prompt {index}."
            for index in range(1, 11)
        ]

        first_response = json.dumps(
            {
                "scene_prompts": [
                    {
                        "index": index,
                        "prompt": (
                            f"Refined visual prompt "
                            f"{index}."
                        ),
                    }
                    for index in range(1, 9)
                ]
            }
        )

        second_response = json.dumps(
            {
                "scene_prompts": [
                    {
                        "index": index,
                        "prompt": (
                            f"Refined visual prompt "
                            f"{index}."
                        ),
                    }
                    for index in range(9, 11)
                ]
            }
        )

        with patch.object(
            llm,
            "_generate_response",
            side_effect=[
                first_response,
                second_response,
            ],
        ) as generate:
            result = (
                llm.refine_expanded_scene_prompts(
                    prompts
                )
            )

        self.assertEqual(
            generate.call_count,
            2,
        )

        self.assertEqual(
            len(result),
            10,
        )

        self.assertEqual(
            result[0],
            "Refined visual prompt 1.",
        )

        self.assertEqual(
            result[8],
            "Refined visual prompt 9.",
        )


    def test_refinement_repairs_long_anchor_possessive(self):
        anchors = [
            {
                "tag": "CHAR_1",
                "description": (
                    "an elderly man with short hair "
                    "wearing a sleep shirt"
                ),
            }
        ]

        expanded = [
            (
                "Point-of-view shot from behind "
                "an elderly man with short hair "
                "wearing a sleep shirt's shoulder "
                "looking into the dark room."
            )
        ]

        first_response = json.dumps(
            {
                "scene_prompts": [
                    {
                        "index": 1,
                        "prompt": expanded[0],
                    }
                ]
            }
        )

        repaired_response = json.dumps(
            {
                "scene_prompts": [
                    {
                        "index": 1,
                        "prompt": (
                            "Point-of-view shot from behind "
                            "the shoulder of an elderly man "
                            "with short hair wearing a sleep "
                            "shirt, looking into the dark room."
                        ),
                    }
                ]
            }
        )

        with patch.object(
            llm,
            "_generate_response",
            side_effect=[
                first_response,
                repaired_response,
            ],
        ) as generate:
            result = (
                llm.refine_expanded_scene_prompts(
                    expanded,
                    source_scene_prompts=[
                        (
                            "Point-of-view shot from behind "
                            "[CHAR_1]'s shoulder looking "
                            "into the dark room."
                        )
                    ],
                    anchors=anchors,
                )
            )

        self.assertEqual(
            generate.call_count,
            2,
        )

        self.assertEqual(
            result,
            [
                (
                    "Point-of-view shot from behind "
                    "the shoulder of an elderly man "
                    "with short hair wearing a sleep "
                    "shirt, looking into the dark room."
                )
            ],
        )

        repair_prompt = (
            generate.call_args_list[1].args[0]
        )

        self.assertIn(
            "Expansion Artifact Repair Editor",
            repair_prompt,
        )

        self.assertIn(
            "full expanded anchor used as possessive",
            repair_prompt,
        )

        self.assertIn(
            (
                "an elderly man with short hair "
                "wearing a sleep shirt's"
            ),
            repair_prompt,
        )

    def test_refinement_repairs_malformed_determiner(self):
        expanded = [
            (
                "Gabriel reaches into the open "
                "a wooden drawer containing stored items."
            )
        ]

        first_response = json.dumps(
            {
                "scene_prompts": [
                    {
                        "index": 1,
                        "prompt": expanded[0],
                    }
                ]
            }
        )

        repaired_response = json.dumps(
            {
                "scene_prompts": [
                    {
                        "index": 1,
                        "prompt": (
                            "Gabriel reaches into the open "
                            "wooden drawer containing "
                            "stored items."
                        ),
                    }
                ]
            }
        )

        with patch.object(
            llm,
            "_generate_response",
            side_effect=[
                first_response,
                repaired_response,
            ],
        ) as generate:
            result = (
                llm.refine_expanded_scene_prompts(
                    expanded
                )
            )

        self.assertEqual(
            generate.call_count,
            2,
        )

        self.assertEqual(
            result,
            [
                (
                    "Gabriel reaches into the open "
                    "wooden drawer containing "
                    "stored items."
                )
            ],
        )

        self.assertIn(
            "malformed duplicated determiner",
            generate.call_args_list[1].args[0],
        )


if __name__ == "__main__":
    unittest.main()
