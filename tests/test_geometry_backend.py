import json
import unittest

from geometry_backend import CameraPose, align_trajectories, apply_alignment, discover_backends, extract_route_signature, fallback_geometry


class GeometryBackendTests(unittest.TestCase):
    def test_fallback_is_deterministic_and_explicitly_unavailable(self):
        result = fallback_geometry("vid", [0.0, 1.5, 3.0])
        self.assertEqual(result.status, "geometry_unavailable")
        self.assertEqual([frame.timestamp_seconds for frame in result.frames], [0.0, 1.5, 3.0])
        self.assertTrue(all(not frame.pose.available for frame in result.frames))
        self.assertEqual(result.to_dict()["route_signature"]["uncertainty"], 1.0)
        json.dumps(result.to_dict())

    def test_optional_backend_discovery_is_lazy(self):
        discovered = discover_backends()
        self.assertEqual(set(discovered), {"da3", "vggt", "mapanything"})
        self.assertTrue(all("available" in item for item in discovered.values()))

    def test_sim3_alignment_recovers_scale_rotation_and_translation(self):
        source = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 1, 0)]
        target = [(3, -2, 4), (3, 0, 4), (1, -2, 4), (3, -2, 6), (1, 2, 4)]
        alignment = align_trajectories(source, target)
        transformed = apply_alignment(source, alignment)
        self.assertAlmostEqual(alignment.scale, 2.0, places=4)
        for got, want in zip(transformed, target):
            for left, right in zip(got, want):
                self.assertAlmostEqual(left, right, places=4)
        self.assertEqual(alignment.inlier_count, 5)

    def test_robust_alignment_rejects_one_outlier(self):
        source = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 1, 0), (9, 9, 9)]
        target = [(x + 4, y - 1, z + 2) for x, y, z in source]
        target[-1] = (100, -100, 50)
        result = align_trajectories(source, target, robust=True)
        self.assertLess(result.inlier_count, len(source))
        self.assertGreaterEqual(result.confidence, 0.4)

    def test_route_signature_counts_turns_and_uncertainty(self):
        poses = [
            CameraPose(0, translation=(0, 0, 0), confidence=1, available=True),
            CameraPose(1, translation=(0, 0, 1), confidence=1, available=True),
            CameraPose(2, rotation=(0, 0, 1, 0, 1, 0, -1, 0, 0), translation=(1, 0, 1), confidence=1, available=True),
        ]
        signature = extract_route_signature(poses)
        self.assertTrue(signature.available)
        self.assertEqual(signature.topology["turn_count"], 1)
        self.assertEqual(signature.uncertainty, 0.0)


if __name__ == "__main__":
    unittest.main()
