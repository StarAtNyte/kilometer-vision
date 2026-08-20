import unittest

from specialists import choose, score_compass, score_distance, score_similarity


class SpecialistTests(unittest.TestCase):
    def test_missing_evidence_is_uniform_and_explicit(self):
        result = score_compass(["North", "South", "East", "West"], None)
        self.assertEqual(result["status"], "insufficient_heading_evidence")
        self.assertEqual(choose(result), 0)

    def test_distance_scores_nearest_numeric_option(self):
        result = score_distance(["10 meters", "50 meters", "100 meters"], 48, 5)
        self.assertEqual(choose(result), 1)

    def test_similarity_requires_exact_option_count(self):
        result = score_similarity(["A", "B"], [0.2, 0.8], "landmark_match")
        self.assertEqual(choose(result), 1)
        self.assertEqual(score_similarity(["A", "B"], [0.2], "x")["status"], "insufficient_visual_evidence")


if __name__ == "__main__":
    unittest.main()
