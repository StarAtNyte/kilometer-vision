import json
import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from submission import _geometry_distance_estimate, _landmark_presence_index, audit_predictions, generate_predictions, select_option


class SubmissionTests(unittest.TestCase):
    def payload(self):
        return {
            "vid": {
                "0": {"id": 0, "question": "Which map?", "options": ["A", "B", "C", "D", "E"], "answer": "secret", "answer_id": 4},
                "1": {"id": 1, "question": "Which direction at end?", "options": ["North", "South", "East", "West", "Northwest"], "answer": "secret", "answer_id": 2},
            }
        }

    def test_generation_ignores_labels_and_audits(self):
        payload = self.payload()
        predictions, evidence = generate_predictions(payload)
        self.assertTrue(audit_predictions(payload, predictions)["valid"])
        self.assertEqual(len(predictions["vid"]), 2)
        self.assertEqual(len(evidence["videos"]["vid"]), 2)
        self.assertTrue(all(set(row) == {"id", "answer", "answer_id"} for row in predictions["vid"]))

    def test_geometry_context_is_exposed_without_affecting_submission_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "geometry").mkdir()
            (root / "geometry" / "vid.json").write_text(json.dumps({"backend": "da3", "frames": [{"extrinsic": [1]}]}))
            _, evidence = generate_predictions(self.payload(), root=root, split="validation")
            evidence_rows = evidence["videos"]["vid"]
            self.assertEqual(evidence_rows[0]["evidence"][1]["source"], "da3")
            self.assertEqual(evidence_rows[0]["evidence"][1]["pose_frames"], 1)

    def test_geometry_changes_non_metric_fallback_and_distance_prior(self):
        question = {"question": "Which route turn is shown?", "options": ["turn left", "turn right", "straight", "visited before", "not visited"]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "geometry").mkdir()
            base = {"backend": "da3", "frames": [{"index": 0, "timestamp_seconds": 0, "extrinsic": [[1,0,0,0],[0,1,0,0],[0,0,1,0]], "confidence": 0.1}]}
            changed = {"backend": "da3", "frames": [{"index": 0, "timestamp_seconds": 0, "extrinsic": [[1,0,0,10],[0,1,0,0],[0,0,1,0]], "confidence": 0.9}]}
            (root / "geometry" / "vid.json").write_text(json.dumps(base))
            first = select_option("vid", "0", question, root=root, split="validation")
            (root / "geometry" / "vid.json").write_text(json.dumps(changed))
            second = select_option("vid", "0", question, root=root, split="validation")
            self.assertNotEqual(first, second)

            distance = {"question": "How far did the camera travel?", "options": ["50 meters", "90 meters", "100 meters", "150 meters", "200 meters"], "start_seconds": 0, "end_seconds": 80}
            (root / "geometry" / "vid.json").write_text(json.dumps(base))
            low_motion = select_option("vid", "distance", distance, root=root, split="validation")
            (root / "geometry" / "vid.json").write_text(json.dumps(changed))
            high_motion = select_option("vid", "distance", distance, root=root, split="validation")
            self.assertIn(low_motion, range(5)); self.assertIn(high_motion, range(5))
            self.assertNotEqual(_geometry_distance_estimate(100.0, {"available": True, "path_length_relative": 1.0, "closure_ratio": 1.0, "mean_confidence": 0.1}), _geometry_distance_estimate(100.0, {"available": True, "path_length_relative": 1.0, "closure_ratio": 0.0, "mean_confidence": 0.9}))

    def test_landmark_presence_prefers_semantic_yes_no_options(self):
        options = ["No, it does not appear", "Yes, it is shown", "Maybe", "Not visited", "Unknown"]
        self.assertEqual(_landmark_presence_index(options, True), 1)
        self.assertEqual(_landmark_presence_index(options, False), 0)

    def test_relative_pose_heading_is_used_only_with_initial_anchor(self):
        question = {"question": "Starting facing north, which direction are you facing at the end?", "options": ["North", "East", "South", "West", "Northeast"]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "geometry").mkdir()
            payload = {"backend": "da3", "frames": [
                {"timestamp_seconds": 0, "extrinsic": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], "confidence": 1.0},
                {"timestamp_seconds": 1, "extrinsic": [[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]], "confidence": 1.0},
            ]}
            (root / "geometry" / "vid.json").write_text(json.dumps(payload))
            self.assertEqual(select_option("vid", "heading", question, root=root, split="validation"), 1)

    def test_vlm_ok_record_is_preferred_over_specialist_path(self):
        question = {"question": "Which map matches the route?", "options": ["A", "B", "C", "D", "E"]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "vlm").mkdir()
            (root / "vlm" / "vid.json").write_text(json.dumps({"0": {"selected_index": 3, "raw_text": "D", "status": "ok", "checkpoint": "Qwen/Qwen2.5-VL-7B-Instruct"}}))
            self.assertEqual(select_option("vid", "0", question, root=root, split="validation"), 3)

    def test_vlm_record_is_ignored_for_distance_category(self):
        question = {"question": "How far did the camera travel?", "options": ["50 meters", "90 meters", "100 meters", "150 meters", "200 meters"], "start_seconds": 0, "end_seconds": 80}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "vlm").mkdir()
            (root / "vlm" / "vid.json").write_text(json.dumps({"0": {"selected_index": 4, "raw_text": "E", "status": "ok", "checkpoint": "test-checkpoint"}}))
            without_vlm = select_option("vid", "0", question, root=None, split=None)
            with_vlm = select_option("vid", "0", question, root=root, split="validation")
            self.assertEqual(without_vlm, with_vlm)

    def test_promoted_question_window_motion_precedes_vlm_and_can_be_disabled(self):
        question = {"question": "The camera is pointing towards the North at the start. Where is it pointing at the end?",
                    "options": ["East", "South", "West", "Northwest", "North"]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "vlm").mkdir()
            (root / "vlm" / "vid.json").write_text(json.dumps({"0": {"selected_index": 0, "status": "ok"}}))
            path = root / "motion" / "validation" / "compass"
            path.mkdir(parents=True)
            (path / "vid.json").write_text(json.dumps({"0": {"status": "ok", "summary": {"net_yaw_degrees": -148}}}))
            self.assertEqual(select_option("vid", "0", question, root=root, split="validation"), 1)
            with patch.dict(os.environ, {"KILOMETERVISION_ENABLE_MOTION": "0"}):
                self.assertEqual(select_option("vid", "0", question, root=root, split="validation"), 0)

    def test_vlm_unparsed_record_falls_through_to_specialist_path(self):
        question = {"question": "Which map matches the route?", "options": ["A", "B", "C", "D", "E"]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "vlm").mkdir()
            (root / "vlm" / "vid.json").write_text(json.dumps({"0": {"selected_index": None, "raw_text": "not sure", "status": "unparsed", "checkpoint": "Qwen/Qwen2.5-VL-7B-Instruct"}}))
            without_vlm = select_option("vid", "0", question, root=None, split=None)
            with_unparsed_vlm = select_option("vid", "0", question, root=root, split="validation")
            self.assertEqual(without_vlm, with_unparsed_vlm)

    def test_split_namespaced_vlm_record_precedes_legacy_record(self):
        question = {"question": "Which map matches the route?", "options": ["A", "B", "C", "D", "E"]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "vlm" / "validation").mkdir(parents=True)
            (root / "vlm" / "vid.json").write_text(json.dumps({"0": {"selected_index": 1, "status": "ok"}}))
            (root / "vlm" / "validation" / "vid.json").write_text(json.dumps({"0": {"selected_index": 3, "status": "ok"}}))
            self.assertEqual(select_option("vid", "0", question, root=root, split="validation"), 3)

    def test_vlm_evidence_trace_reports_status(self):
        payload = self.payload()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "vlm").mkdir()
            (root / "vlm" / "vid.json").write_text(json.dumps({"0": {"selected_index": 4, "raw_text": "E", "status": "ok", "checkpoint": "test-checkpoint"}}))
            _, evidence = generate_predictions(payload, root=root, split="validation")
            vlm_entry = evidence["videos"]["vid"][0]["evidence"][2]
            self.assertEqual(vlm_entry["status"], "ok")
            self.assertEqual(vlm_entry["checkpoint"], "test-checkpoint")
            no_record_entry = evidence["videos"]["vid"][1]["evidence"][2]
            self.assertEqual(no_record_entry["status"], "unavailable")

    def test_audit_rejects_answer_id_mismatch(self):
        payload = self.payload()
        predictions, _ = generate_predictions(payload)
        predictions["vid"][0]["answer"] = "wrong option"
        report = audit_predictions(payload, predictions)
        self.assertFalse(report["valid"])
        self.assertTrue(any("does not equal" in error for error in report["errors"]))

    def test_missing_question_fails_audit(self):
        payload = self.payload()
        predictions, _ = generate_predictions(payload)
        predictions["vid"].pop()
        report = audit_predictions(payload, predictions)
        self.assertFalse(report["valid"])
        self.assertEqual(report["expected"], 2)


if __name__ == "__main__":
    unittest.main()
