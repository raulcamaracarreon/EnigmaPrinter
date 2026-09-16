import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import narration_alignment, uploaded_narration_plan


class TestUploadedNarrationPlan(unittest.TestCase):
    def test_uploaded_audio_reaches_semantic_timeline_and_active_shot_plan(self):
        words = [
            narration_alignment.RecognizedWord("Una", 0.2, 0.4),
            narration_alignment.RecognizedWord("mujer", 0.4, 0.8),
            narration_alignment.RecognizedWord("esperaba", 0.8, 1.3),
            narration_alignment.RecognizedWord("sola", 1.3, 1.6),
            narration_alignment.RecognizedWord("en", 1.6, 1.75),
            narration_alignment.RecognizedWord("el", 1.75, 1.85),
            narration_alignment.RecognizedWord("andén", 1.85, 2.2),
            narration_alignment.RecognizedWord("Un", 4.2, 4.4),
            narration_alignment.RecognizedWord("maletín", 4.4, 4.9),
            narration_alignment.RecognizedWord("apareció", 4.9, 5.4),
            narration_alignment.RecognizedWord("junto", 5.4, 5.7),
            narration_alignment.RecognizedWord("a", 5.7, 5.8),
            narration_alignment.RecognizedWord("ella", 5.8, 6.1),
        ]

        result = uploaded_narration_plan.build_uploaded_narration_plan(
            audio_file="unused.wav",
            script="Una mujer esperaba sola en el andén. Un maletín apareció junto a ella.",
            audio_duration=7.0,
            max_clip_duration=3.0,
            recognized_words=words,
        )
        data = result.to_dict()

        self.assertEqual(data["narration_timeline"]["timing_source"], "derived")
        self.assertEqual(len(data["narration_timeline"]["segments"]), 2)
        self.assertEqual(len(data["semantic_timeline"]["segments"]), 2)
        self.assertGreaterEqual(len(data["active_shot_plan"]["shots"]), 2)
        self.assertAlmostEqual(data["active_shot_plan"]["audio_duration"], 7.0)
        self.assertTrue(
            all(
                shot["duration"] <= 3.0 + 1e-6
                for shot in data["active_shot_plan"]["shots"]
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
