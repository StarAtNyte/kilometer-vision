import unittest

from question_planner import classify_question, classify_question_facets, compile_options


class QuestionPlannerTests(unittest.TestCase):
    def test_classifies_spatial_families(self):
        self.assertEqual(classify_question("Where is the camera pointing at the end?"), "compass")
        self.assertEqual(classify_question("What is the straight-line distance in meters?"), "distance")
        self.assertEqual(classify_question("Which map correctly represents the path?"), "map")
        self.assertEqual(classify_question({"question": "Does this image show a landmark?", "image_paths": ["ref.png"]}), "landmark")

    def test_specific_metric_cue_beats_reference_image_fallback(self):
        question = {
            "question": "Does the image show a location, and what is the straight-line distance?",
            "image_paths": ["reference.png"],
            "options": ["100 meters", "0.2 km", "The place is not in the video", "30 meters", "1 mile"],
        }
        self.assertEqual(classify_question(question), "distance")
        self.assertEqual(classify_question_facets(question), ("landmark_presence", "distance"))

    def test_compiles_directions_and_metric_values_without_answer_fields(self):
        question = {
            "question": "Given the route, what direction is the camera facing?",
            "options": ["North", "Southwest", "East", "Northwest", "South"],
            "answer_id": 3,
            "answer": "Northwest",
        }
        result = compile_options(question)
        self.assertEqual(result["category"], "compass")
        self.assertEqual(result["options"][1]["normalized_compass_directions"], ["southwest"])
        self.assertEqual(result["options"][3]["discriminating_tokens"], ["northwest"])
        self.assertNotIn("answer", result)
        self.assertNotIn("answer_id", result)

    def test_normalises_distance_units_to_metres_and_splits_atoms(self):
        question = {
            "question": "How far apart are the two places?",
            "options": ["Around 1.5 km", "100 meters and then 20 m", "1 mile", "3 ft", "Not present"],
        }
        options = compile_options(question)["options"]
        self.assertEqual(options[0]["numeric_distance_values_m"], [1500.0])
        self.assertEqual(options[1]["numeric_distance_values_m"], [100.0, 20.0])
        self.assertAlmostEqual(options[2]["numeric_distance_values_m"][0], 1609.344)
        self.assertEqual(options[1]["atomic_text"], ["100 meters", "20 m"])

    def test_accepts_mapping_question_and_stringifies_scalar_options(self):
        result = compile_options({"question": "Which route?", "options": ["A", 2, "C"]})
        self.assertEqual([item["text"] for item in result["options"]], ["A", "2", "C"])
        self.assertEqual(result["category"], "landmark")


if __name__ == "__main__":
    unittest.main()
