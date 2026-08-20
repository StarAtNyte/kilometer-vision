import json
import tempfile
import unittest
from pathlib import Path

from geometry_contract import build_route_signature, load_da3_geometry


class GeometryContractTests(unittest.TestCase):
    def test_missing_geometry_falls_back_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "geometry").mkdir()
            geometry = load_da3_geometry("vid", root)
            self.assertEqual(geometry.status, "geometry_unavailable")
            self.assertFalse(geometry.route_signature.available)

    def test_load_da3_geometry_parses_extrinsics_into_real_poses(self):
        payload = {
            "backend": "da3",
            "frames": [
                {"timestamp_seconds": 0, "extrinsic": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], "confidence": 1.0},
                {"timestamp_seconds": 1, "extrinsic": [[0, 0, 1, 1], [0, 1, 0, 0], [-1, 0, 0, 0]], "confidence": 1.0},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "geometry").mkdir()
            (root / "geometry" / "vid.json").write_text(json.dumps(payload))
            geometry = load_da3_geometry("vid", root)
            self.assertEqual(geometry.backend, "da3")
            self.assertTrue(geometry.route_signature.available)
            # 90-degree synthetic rotation, matching the compass-lane fixture already
            # used for submission.py's relative-yaw math.
            self.assertEqual(geometry.route_signature.topology["turn_count"], 1)
            self.assertAlmostEqual(geometry.route_signature.turn_angles_degrees[0], 90.0, places=3)

    def test_build_route_signature_writes_narrative_and_artifact(self):
        payload = {
            "backend": "da3",
            "frames": [
                {"timestamp_seconds": 0, "extrinsic": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], "confidence": 1.0},
                {"timestamp_seconds": 1, "extrinsic": [[0, 0, 1, 1], [0, 1, 0, 0], [-1, 0, 0, 0]], "confidence": 1.0},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "geometry").mkdir()
            (root / "geometry" / "vid.json").write_text(json.dumps(payload))
            result = build_route_signature("vid", root)
            output = root / "geometry" / "vid.route_signature.json"
            self.assertTrue(output.exists())
            self.assertEqual(json.loads(output.read_text()), result)
            self.assertTrue(result["available"])
            self.assertEqual(result["topology"]["turn_count"], 1)
            self.assertIn("sustained turn", result["narrative"])
            self.assertIn("right ~90", result["narrative"])

    def test_build_route_signature_handles_missing_geometry_without_raising(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "geometry").mkdir()
            result = build_route_signature("vid", root)
            self.assertFalse(result["available"])
            self.assertIn("relative-scale evidence only", result["narrative"])


if __name__ == "__main__":
    unittest.main()
