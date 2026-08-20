import unittest

from vlm_solver import build_prompt, parse_answer, select_frames


class VlmSolverTests(unittest.TestCase):
    def manifest(self, count=10):
        return {"frames": [{"index": i, "timestamp_seconds": float(i), "path": f"f{i}.jpg"} for i in range(count)]}

    def test_select_frames_windows_to_question_range(self):
        picked = select_frames({"start_seconds": 3, "end_seconds": 6}, self.manifest(), max_frames=6)
        self.assertTrue(all(3.0 <= frame["timestamp_seconds"] <= 6.0 for frame in picked))
        self.assertGreater(len(picked), 0)

    def test_select_frames_samples_evenly_without_a_window(self):
        picked = select_frames({}, self.manifest(20), max_frames=4)
        self.assertEqual(len(picked), 4)
        self.assertEqual(picked[0]["index"], 0)
        self.assertEqual(picked[-1]["index"], 19)

    def test_select_frames_falls_back_to_full_manifest_when_window_is_empty(self):
        picked = select_frames({"start_seconds": 100, "end_seconds": 200}, self.manifest(5), max_frames=3)
        self.assertGreater(len(picked), 0)

    def test_select_frames_handles_missing_or_empty_manifest(self):
        self.assertEqual(select_frames({}, {}, max_frames=4), [])
        self.assertEqual(select_frames({}, {"frames": []}, max_frames=4), [])

    def test_build_prompt_renders_options_and_geometry_context(self):
        question = {"question": "Which way at the end?", "options": ["North", "South", "East", "West", "Northeast"]}
        prompt = build_prompt(question, "compass", route_signature_narrative="1 sustained turn(s) detected: right ~90°.")
        self.assertIn("A. North", prompt)
        self.assertIn("E. Northeast", prompt)
        self.assertIn("right ~90", prompt)
        self.assertIn("exactly one letter", prompt)

    def test_build_prompt_omits_absent_evidence_sections(self):
        prompt = build_prompt({"question": "Which map?", "options": ["a", "b", "c", "d", "e"]}, "map")
        self.assertNotIn("Geometry evidence", prompt)
        self.assertNotIn("OCR", prompt)

    def test_parse_answer_accepts_bare_letter_and_number(self):
        self.assertEqual(parse_answer("C"), 2)
        self.assertEqual(parse_answer(" (b) "), 1)
        self.assertEqual(parse_answer("3"), 2)

    def test_parse_answer_accepts_labeled_answer(self):
        self.assertEqual(parse_answer("The answer is D."), 3)
        self.assertEqual(parse_answer("Answer: A"), 0)

    def test_parse_answer_rejects_ambiguous_or_empty_text(self):
        self.assertIsNone(parse_answer(""))
        self.assertIsNone(parse_answer(None))
        self.assertIsNone(parse_answer("I think it shows a park near a fountain."))
        self.assertIsNone(parse_answer("Maybe A or C, hard to tell."))

    def test_parse_answer_rejects_out_of_range_index(self):
        self.assertIsNone(parse_answer("F"))
        self.assertIsNone(parse_answer("6"))


if __name__ == "__main__":
    unittest.main()
