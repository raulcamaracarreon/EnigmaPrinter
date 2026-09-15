import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import llm


class TestVisualPromptGeneration(unittest.TestCase):
    def test_prompt_count_uses_midpoint_of_estimated_range(self):
        self.assertEqual(llm.calculate_visual_prompt_count(13.5, 18.2, 5), 4)
        self.assertEqual(llm.calculate_visual_prompt_count(13.5, 18.2, 3), 6)
        self.assertEqual(llm.calculate_visual_prompt_count(13.5, 18.2, 10), 2)

    def test_visual_prompt_mode_requests_complete_prompts(self):
        generated = [
            f"A detailed visual scene prompt number {index} showing a subject in a concrete environment at dusk"
            for index in range(1, 5)
        ]
        with patch.object(llm, '_generate_response', return_value=json.dumps(generated)) as mocked:
            result = llm.generate_terms(
                'forest mystery',
                'A man runs through a forest and discovers an abandoned cabin.',
                amount=4,
                match_script_order=True,
                visual_prompt_mode=True,
            )

        self.assertEqual(result, generated)
        sent_prompt = mocked.call_args.args[0]
        self.assertIn('exactly 4 chronological visual scene prompts', sent_prompt)
        self.assertIn('12-35 words', sent_prompt)
        self.assertIn('repeat stable visible descriptors', sent_prompt)

    def test_legacy_stock_search_mode_remains_short(self):
        generated = ['forest trail', 'abandoned cabin', 'night fog']
        with patch.object(llm, '_generate_response', return_value=json.dumps(generated)) as mocked:
            result = llm.generate_terms(
                'forest mystery',
                'A man runs through a forest.',
                amount=3,
                visual_prompt_mode=False,
            )

        self.assertEqual(result, generated)
        sent_prompt = mocked.call_args.args[0]
        self.assertIn('1-3 words', sent_prompt)
        self.assertIn('stock-footage search', sent_prompt)


if __name__ == '__main__':
    unittest.main(verbosity=2)
