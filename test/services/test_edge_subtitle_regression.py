import sys
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import voice


class EdgeSubtitleRegressionTests(unittest.TestCase):
    def test_closing_quote_fragment_is_merged_into_previous_script_line(self):
        merged = voice._merge_punctuation_only_script_lines(
            [
                "Era su propia voz.",
                "“Contesta.",
                "”",
                "Las puertas comenzaron a cerrarse.",
            ]
        )
        self.assertEqual(
            merged,
            [
                "Era su propia voz.",
                "“Contesta.”",
                "Las puertas comenzaron a cerrarse.",
            ],
        )

    def test_edge_cue_aggregation_continues_after_standalone_closing_quote(self):
        raw_lines = [
            "Era su propia voz.",
            "“Contesta.",
            "”",
            "Las puertas comenzaron a cerrarse.",
        ]
        script_lines = voice._merge_punctuation_only_script_lines(raw_lines)
        words = [
            "Era", "su", "propia", "voz",
            "Contesta",
            "Las", "puertas", "comenzaron", "a", "cerrarse",
        ]
        cues = []
        cursor = 0.0
        for word in words:
            cues.append(
                SimpleNamespace(
                    content=word,
                    start=timedelta(seconds=cursor),
                    end=timedelta(seconds=cursor + 0.25),
                )
            )
            cursor += 0.25
        sub_maker = SimpleNamespace(cues=cues)

        items = voice._build_subtitle_items_from_edge_cues(sub_maker, script_lines)

        self.assertEqual(len(items), 3)
        self.assertIn("Era su propia voz.", items[0])
        self.assertIn("“Contesta.”", items[1])
        self.assertIn("Las puertas comenzaron a cerrarse.", items[2])


if __name__ == "__main__":
    unittest.main(verbosity=2)
