import sys
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import narration_timeline


class _Cue:
    def __init__(self, content: str, start: float, end: float):
        self.content = content
        self.start = timedelta(seconds=start)
        self.end = timedelta(seconds=end)


class TestNarrationTimeline(unittest.TestCase):
    def test_native_edge_cues_preserve_sentence_times(self):
        sub_maker = SimpleNamespace(
            cues=[
                _Cue("Las puertas", 0.10, 0.80),
                _Cue(" se abrieron.", 0.80, 1.50),
                _Cue("Ella esperó.", 1.70, 2.60),
            ]
        )
        timeline = narration_timeline.build_narration_timeline(
            script="Las puertas se abrieron. Ella esperó.",
            audio_duration=3.0,
            sub_maker=sub_maker,
            provider_timing_source=narration_timeline.TIMING_NATIVE,
        )

        self.assertEqual(timeline.timing_source, "native")
        self.assertEqual(len(timeline.segments), 2)
        self.assertAlmostEqual(timeline.segments[0].start, 0.10)
        self.assertAlmostEqual(timeline.segments[0].end, 1.50)
        self.assertAlmostEqual(timeline.segments[1].start, 1.70)
        self.assertAlmostEqual(timeline.segments[1].end, 2.60)
        self.assertEqual(len(timeline.alignment_units), 3)
        self.assertEqual(timeline.alignment_units[0].text, "Las puertas")


    def test_explicit_native_word_buffer_survives_coarse_submaker_cues(self):
        # Regression: subtitle-oriented SubMaker cues can be coarser than the
        # original provider WordBoundary stream. The audio-first planner must
        # prefer the explicitly preserved native word timing.
        sub_maker = SimpleNamespace(
            cues=[
                _Cue(
                    "Sólo quedaban encendidas unas lámparas amarillentas y el zumbido constante del ascensor.",
                    0.0,
                    6.6,
                )
            ],
            _mpt_alignment_units=[
                {"start": 0.0, "end": 0.7, "text": "Sólo"},
                {"start": 0.7, "end": 1.4, "text": "quedaban"},
                {"start": 1.4, "end": 2.2, "text": "encendidas"},
                {"start": 2.2, "end": 2.7, "text": "unas"},
                {"start": 2.7, "end": 3.2, "text": "lámparas"},
                {"start": 3.2, "end": 3.8, "text": "amarillentas"},
                {"start": 3.8, "end": 4.1, "text": "y"},
                {"start": 4.1, "end": 4.6, "text": "el"},
                {"start": 4.6, "end": 5.2, "text": "zumbido"},
                {"start": 5.2, "end": 5.7, "text": "constante"},
                {"start": 5.7, "end": 5.9, "text": "del"},
                {"start": 5.9, "end": 6.6, "text": "ascensor."},
            ],
        )
        timeline = narration_timeline.build_narration_timeline(
            script=(
                "Sólo quedaban encendidas unas lámparas amarillentas "
                "y el zumbido constante del ascensor."
            ),
            audio_duration=6.6,
            sub_maker=sub_maker,
            provider_timing_source=narration_timeline.TIMING_NATIVE,
        )

        self.assertEqual(len(timeline.segments), 1)
        self.assertEqual(len(timeline.alignment_units), 12)
        self.assertEqual(timeline.alignment_units[5].text, "amarillentas")
        self.assertAlmostEqual(timeline.alignment_units[5].start, 3.2)

    def test_native_legacy_offsets_support_azure_v2_style_boundaries(self):
        sub_maker = SimpleNamespace(
            cues=[],
            subs=["Las ", "puertas ", "se abrieron.", "Ella esperó."],
            offset=[
                (1_000_000, 4_000_000),
                (4_000_000, 8_000_000),
                (8_000_000, 15_000_000),
                (17_000_000, 26_000_000),
            ],
        )
        timeline = narration_timeline.build_narration_timeline(
            script="Las puertas se abrieron. Ella esperó.",
            audio_duration=3.0,
            sub_maker=sub_maker,
            provider_timing_source=narration_timeline.TIMING_NATIVE,
        )

        self.assertEqual(timeline.timing_source, "native")
        self.assertEqual(len(timeline.segments), 2)
        self.assertAlmostEqual(timeline.segments[0].start, 0.10)
        self.assertAlmostEqual(timeline.segments[0].end, 1.50)
        self.assertEqual(len(timeline.alignment_units), 4)

    def test_derived_provider_offsets_remain_marked_derived(self):
        sub_maker = SimpleNamespace(
            cues=[],
            subs=["Primera frase.", "Segunda frase."],
            offset=[(0, 20_000_000), (20_000_000, 50_000_000)],
        )
        timeline = narration_timeline.build_narration_timeline(
            script="Primera frase. Segunda frase.",
            audio_duration=5.0,
            sub_maker=sub_maker,
            provider_timing_source=narration_timeline.TIMING_DERIVED,
        )

        self.assertEqual(timeline.timing_source, "derived")
        self.assertEqual([s.timing_source for s in timeline.segments], ["derived", "derived"])
        self.assertEqual(len(timeline.alignment_units), 2)

    def test_missing_alignment_falls_back_to_estimated_full_coverage(self):
        timeline = narration_timeline.build_narration_timeline(
            script="Una frase corta. Otra frase bastante más larga para narrar.",
            audio_duration=10.0,
            sub_maker=None,
        )

        self.assertEqual(timeline.timing_source, "estimated")
        self.assertEqual(len(timeline.segments), 2)
        self.assertAlmostEqual(timeline.segments[0].start, 0.0)
        self.assertAlmostEqual(timeline.segments[-1].end, 10.0)
        self.assertGreater(timeline.segments[1].duration, timeline.segments[0].duration)
        self.assertEqual(timeline.alignment_units, ())

    def test_provider_quality_is_capability_based(self):
        self.assertEqual(narration_timeline.provider_timing_source("azure-tts-v1"), "native")
        self.assertEqual(narration_timeline.provider_timing_source("azure-tts-v2"), "native")
        self.assertEqual(narration_timeline.provider_timing_source("elevenlabs"), "derived")
        self.assertEqual(narration_timeline.provider_timing_source("kokoro"), "derived")


if __name__ == "__main__":
    unittest.main(verbosity=2)
