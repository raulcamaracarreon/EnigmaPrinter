import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

# Add the MoneyPrinterTurbo project root to sys.path when the test is run
# from test/services, matching the project's existing test files.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import material
from app.services import subtitle
from app.services import task as tm
from app.services import voice as vs
from app.utils import utils


class TestComfyUIT2IRegressions(unittest.TestCase):
    def test_multiline_visual_prompts_keep_internal_commas(self):
        """One prompt per line must preserve commas inside each visual prompt."""
        params = SimpleNamespace(
            video_terms=(
                "deserted colonial plaza, heavy fog, old stone buildings\n"
                "elderly watchman, frightened expression, lantern light\n"
                "shadowy figure, long coat, face hidden in darkness"
            ),
            match_materials_to_script=True,
        )

        terms = tm.generate_terms("regression-task", params, "unused script")

        self.assertEqual(
            terms,
            [
                "deserted colonial plaza, heavy fog, old stone buildings",
                "elderly watchman, frightened expression, lantern light",
                "shadowy figure, long coat, face hidden in darkness",
            ],
        )

    def test_legacy_single_line_comma_terms_still_work(self):
        """Existing comma-separated keyword input remains backward compatible."""
        params = SimpleNamespace(
            video_terms="moonlight, old plaza, fog",
            match_materials_to_script=True,
        )

        terms = tm.generate_terms("regression-task", params, "unused script")

        self.assertEqual(terms, ["moonlight", "old plaza", "fog"])

    def test_comfyui_t2i_timing_covers_audio_and_is_frame_aligned(self):
        """Automatic timing must cover narration + margin without looping clips."""
        cases = (
            (91.0, 19, 4.8),
            (24.0, 4, 181 / 30),  # 6.033333... seconds
        )

        for audio_duration, image_count, expected in cases:
            with self.subTest(audio_duration=audio_duration, image_count=image_count):
                duration = material.calculate_comfyui_t2i_clip_duration(
                    audio_duration,
                    image_count,
                )
                self.assertAlmostEqual(duration, expected, places=9)

                required = (
                    audio_duration
                    + material.COMFYUI_T2I_TIMING_SAFETY_SECONDS
                )
                self.assertGreaterEqual(duration * image_count, required)

                frames = duration * material.COMFYUI_T2I_RENDER_FPS
                self.assertAlmostEqual(frames, round(frames), places=9)

        self.assertEqual(material.calculate_comfyui_t2i_clip_duration(0, 4), 0.0)
        self.assertEqual(material.calculate_comfyui_t2i_clip_duration(10, 0), 0.0)

    def test_splitter_can_preserve_spanish_closing_punctuation(self):
        """Subtitle mode keeps punctuation while legacy mode still strips it."""
        text = "¿Quién camina? ¡Deténgase! ¿Lo volverían a ver?"

        self.assertEqual(
            utils.split_string_by_punctuations(text, keep_punctuation=True),
            ["¿Quién camina?", "¡Deténgase!", "¿Lo volverían a ver?"],
        )
        self.assertEqual(
            utils.split_string_by_punctuations(text),
            ["¿Quién camina", "¡Deténgase", "¿Lo volverían a ver"],
        )

    def test_voice_subtitle_generation_restores_script_punctuation(self):
        """Legacy TTS boundaries may omit marks; the final SRT keeps script punctuation."""
        script = "¿Quién camina? ¡Deténgase! ¿Lo volverían a ver?"
        sub_maker = SimpleNamespace(
            subs=["¿Quién camina", "¡Deténgase", "¿Lo volverían a ver"],
            offset=[
                (0, 10_000_000),
                (10_000_000, 20_000_000),
                (20_000_000, 30_000_000),
            ],
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            subtitle_file = Path(tmp_dir) / "subtitle.srt"
            vs.create_subtitle(sub_maker, script, str(subtitle_file))
            content = subtitle_file.read_text(encoding="utf-8")

        self.assertIn("¿Quién camina?", content)
        self.assertIn("¡Deténgase!", content)
        self.assertIn("¿Lo volverían a ver?", content)

    def test_subtitle_correct_restores_missing_closing_marks(self):
        """The correction path must not strip punctuation a second time."""
        script = "¿Teletransportación?\n¿Leyenda?\n¿O una historia que creció durante siglos?"
        initial_srt = """1
00:00:00,000 --> 00:00:01,000
¿Teletransportación

2
00:00:01,000 --> 00:00:02,000
¿Leyenda

3
00:00:02,000 --> 00:00:04,000
¿O una historia que creció durante siglos

"""

        with tempfile.TemporaryDirectory() as tmp_dir:
            subtitle_file = Path(tmp_dir) / "subtitle.srt"
            subtitle_file.write_text(initial_srt, encoding="utf-8")

            subtitle.correct(str(subtitle_file), script)
            content = subtitle_file.read_text(encoding="utf-8")

        self.assertIn("¿Teletransportación?", content)
        self.assertIn("¿Leyenda?", content)
        self.assertIn("¿O una historia que creció durante siglos?", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
