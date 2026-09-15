import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import visual_timeline


class TestVisualTimeline(unittest.TestCase):
    def test_merges_punctuation_only_units_without_creating_scene(self):
        narration = {
            "audio_duration": 6.0,
            "timing_source": "native",
            "segments": [
                {"index": 1, "start": 0.0, "end": 2.0, "text": "Dijo: “Contesta."},
                {"index": 2, "start": 2.0, "end": 2.1, "text": "”"},
                {"index": 3, "start": 2.1, "end": 6.0, "text": "Las puertas se cerraron."},
            ],
        }
        timeline = visual_timeline.build_visual_timeline(narration)

        self.assertEqual(len(timeline.segments), 2)
        self.assertIn("Contesta.”", timeline.segments[0].narration_text)
        self.assertNotIn("Contesta. ”", timeline.segments[0].narration_text)

    def test_weak_comma_clauses_form_one_semantic_scene(self):
        narration = {
            "audio_duration": 8.3,
            "timing_source": "native",
            "segments": [
                {"index": 1, "start": 0.0, "end": 0.78, "text": "Cada noche,"},
                {"index": 2, "start": 0.78, "end": 2.51, "text": "poco después de las once,"},
                {"index": 3, "start": 2.51, "end": 8.3, "text": "una mujer salía de su oficina."},
            ],
        }
        timeline = visual_timeline.build_visual_timeline(narration)

        self.assertEqual(len(timeline.segments), 1)
        self.assertEqual(timeline.segments[0].narration_indices, (1, 2, 3))
        self.assertEqual(
            timeline.segments[0].narration_text,
            "Cada noche, poco después de las once, una mujer salía de su oficina.",
        )

    def test_semantic_scene_boundaries_do_not_depend_on_clip_duration(self):
        narration = {
            "audio_duration": 12.0,
            "timing_source": "native",
            "segments": [
                {"index": 1, "start": 0.0, "end": 1.0, "text": "Una noche,"},
                {"index": 2, "start": 1.0, "end": 5.5, "text": "el ascensor se detuvo."},
                {"index": 3, "start": 5.5, "end": 8.0, "text": "Las puertas se abrieron."},
                {"index": 4, "start": 8.0, "end": 12.0, "text": "El pasillo estaba vacío."},
            ],
        }
        first = visual_timeline.build_visual_timeline(narration)
        second = visual_timeline.build_visual_timeline(narration)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertNotIn("target_scene_duration", first.to_dict())

    def test_visual_windows_are_contiguous_and_cover_audio(self):
        narration = {
            "audio_duration": 12.0,
            "timing_source": "derived",
            "segments": [
                {"index": 1, "start": 0.2, "end": 2.0, "text": "Primera frase."},
                {"index": 2, "start": 2.8, "end": 5.5, "text": "Segunda frase."},
                {"index": 3, "start": 6.0, "end": 11.2, "text": "Última frase."},
            ],
        }
        timeline = visual_timeline.build_visual_timeline(narration)

        self.assertAlmostEqual(timeline.segments[0].start, 0.0)
        self.assertAlmostEqual(timeline.segments[-1].end, 12.0)
        for previous, current in zip(timeline.segments, timeline.segments[1:]):
            self.assertAlmostEqual(previous.end, current.start)
        self.assertEqual(timeline.timing_source, "derived")

    def test_preserves_fine_alignment_boundaries_without_changing_semantic_scene_count(self):
        narration = {
            "audio_duration": 6.0,
            "timing_source": "native",
            "segments": [
                {"index": 1, "start": 0.0, "end": 6.0, "text": "Las lámparas amarillas zumbaban en silencio."},
            ],
            "alignment_units": [
                {"index": 1, "start": 0.0, "end": 1.0, "text": "Las"},
                {"index": 2, "start": 1.0, "end": 2.0, "text": "lámparas"},
                {"index": 3, "start": 2.0, "end": 3.0, "text": "amarillas"},
                {"index": 4, "start": 3.0, "end": 4.0, "text": "zumbaban"},
                {"index": 5, "start": 4.0, "end": 5.0, "text": "en"},
                {"index": 6, "start": 5.0, "end": 6.0, "text": "silencio."},
            ],
        }
        timeline = visual_timeline.build_visual_timeline(narration)

        self.assertEqual(len(timeline.segments), 1)
        self.assertEqual(len(timeline.segments[0].alignment_spans), 6)
        self.assertEqual(timeline.segments[0].alignment_boundaries, (1.0, 2.0, 3.0, 4.0, 5.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
