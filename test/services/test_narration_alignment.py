import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import narration_alignment


class TestNarrationAlignment(unittest.TestCase):
    def test_exact_spanish_script_builds_source_text_timeline(self):
        words = [
            narration_alignment.RecognizedWord("Una", 0.20, 0.45),
            narration_alignment.RecognizedWord("mujer", 0.46, 0.85),
            narration_alignment.RecognizedWord("esperaba", 0.86, 1.35),
            narration_alignment.RecognizedWord("sola", 1.36, 1.70),
            narration_alignment.RecognizedWord("Miró", 2.10, 2.38),
            narration_alignment.RecognizedWord("el", 2.39, 2.50),
            narration_alignment.RecognizedWord("maletín", 2.51, 3.05),
        ]

        result = narration_alignment.align_external_narration(
            audio_file="unused.wav",
            script="Una mujer esperaba sola. Miró el maletín.",
            audio_duration=4.0,
            recognized_words=words,
        )

        self.assertEqual(result.timeline.timing_source, "derived")
        self.assertEqual(len(result.timeline.segments), 2)
        self.assertEqual(result.timeline.segments[0].text, "Una mujer esperaba sola.")
        self.assertEqual(result.timeline.segments[1].text, "Miró el maletín.")
        self.assertAlmostEqual(result.timeline.segments[0].start, 0.20)
        self.assertAlmostEqual(result.timeline.segments[0].end, 1.70)
        self.assertAlmostEqual(result.timeline.segments[1].start, 2.10)
        self.assertAlmostEqual(result.timeline.segments[1].end, 3.05)
        self.assertEqual(
            [unit.text for unit in result.timeline.alignment_units],
            ["Una", "mujer", "esperaba", "sola", "Miró", "el", "maletín"],
        )
        self.assertEqual(result.script_coverage, 1.0)
        self.assertEqual(result.recognized_coverage, 1.0)

    def test_alignment_tolerates_accents_and_minor_whisper_spelling(self):
        words = [
            narration_alignment.RecognizedWord("La", 0.0, 0.2),
            narration_alignment.RecognizedWord("fotografia", 0.2, 0.8),
            narration_alignment.RecognizedWord("estaba", 0.8, 1.1),
            narration_alignment.RecognizedWord("sobre", 1.1, 1.4),
            narration_alignment.RecognizedWord("el", 1.4, 1.5),
            narration_alignment.RecognizedWord("anden", 1.5, 1.9),
        ]

        result = narration_alignment.align_external_narration(
            audio_file="unused.wav",
            script="La fotografía estaba sobre el andén.",
            audio_duration=2.5,
            recognized_words=words,
        )

        self.assertEqual(result.matched_word_count, 6)
        self.assertEqual(result.timeline.segments[0].text, "La fotografía estaba sobre el andén.")
        self.assertEqual(result.timeline.alignment_units[1].text, "fotografía")
        self.assertEqual(result.timeline.alignment_units[-1].text, "andén")

    def test_materially_different_audio_is_rejected_without_estimated_fallback(self):
        words = [
            narration_alignment.RecognizedWord("mañana", 0.0, 0.3),
            narration_alignment.RecognizedWord("iremos", 0.3, 0.7),
            narration_alignment.RecognizedWord("al", 0.7, 0.8),
            narration_alignment.RecognizedWord("mercado", 0.8, 1.2),
        ]

        with self.assertRaisesRegex(
            narration_alignment.NarrationAlignmentError,
            "does not match the script closely enough",
        ):
            narration_alignment.align_external_narration(
                audio_file="unused.wav",
                script="Una mujer esperaba sola en la estación abandonada.",
                audio_duration=3.0,
                recognized_words=words,
            )

    def test_extra_spoken_material_is_rejected(self):
        words = [
            narration_alignment.RecognizedWord("La", 0.0, 0.1),
            narration_alignment.RecognizedWord("mujer", 0.1, 0.4),
            narration_alignment.RecognizedWord("esperó", 0.4, 0.8),
            narration_alignment.RecognizedWord("esto", 0.9, 1.1),
            narration_alignment.RecognizedWord("no", 1.1, 1.2),
            narration_alignment.RecognizedWord("está", 1.2, 1.4),
            narration_alignment.RecognizedWord("en", 1.4, 1.5),
            narration_alignment.RecognizedWord("el", 1.5, 1.6),
            narration_alignment.RecognizedWord("guion", 1.6, 2.0),
        ]

        with self.assertRaisesRegex(
            narration_alignment.NarrationAlignmentError,
            "too much speech outside the script",
        ):
            narration_alignment.align_external_narration(
                audio_file="unused.wav",
                script="La mujer esperó.",
                audio_duration=2.5,
                recognized_words=words,
            )

    def test_every_script_segment_needs_real_timing_evidence(self):
        words = [
            narration_alignment.RecognizedWord("Primera", 0.0, 0.4),
            narration_alignment.RecognizedWord("frase", 0.4, 0.8),
            narration_alignment.RecognizedWord("Tercera", 1.6, 2.0),
            narration_alignment.RecognizedWord("frase", 2.0, 2.4),
        ]

        with self.assertRaisesRegex(
            narration_alignment.NarrationAlignmentError,
            "script segment 2 could not be aligned",
        ):
            narration_alignment.align_external_narration(
                audio_file="unused.wav",
                script="Primera frase. Segunda frase. Tercera frase.",
                audio_duration=3.0,
                recognized_words=words,
                minimum_script_coverage=0.60,
                minimum_recognized_coverage=0.60,
            )

    def test_transcribe_alignment_words_reuses_subtitle_word_timestamps(self):
        fake_items = [
            (1, "00:00:00,100 --> 00:00:00,400", "Hola"),
            (2, "00:00:00,450 --> 00:00:00,900", "mundo"),
        ]
        test_audio = Path(__file__)

        def fake_create(*, audio_file, subtitle_file, word_level):
            self.assertEqual(audio_file, str(test_audio))
            self.assertTrue(word_level)
            Path(subtitle_file).write_text("placeholder", encoding="utf-8")
            return None

        with (
            patch.object(narration_alignment.subtitle, "create", side_effect=fake_create),
            patch.object(
                narration_alignment.subtitle,
                "file_to_subtitles",
                return_value=fake_items,
            ),
        ):
            words = narration_alignment.transcribe_alignment_words(str(test_audio))

        self.assertEqual([word.text for word in words], ["Hola", "mundo"])
        self.assertAlmostEqual(words[0].start, 0.1)
        self.assertAlmostEqual(words[1].end, 0.9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
