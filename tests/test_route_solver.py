import json
import tempfile
import unittest
from pathlib import Path

from kilometer_vision.route_solver import _distance, _image_vector, estimate_route


class RouteSolverTests(unittest.TestCase):
    def _manifest(self, root: Path) -> Path:
        frames = root / "frames"
        frames.mkdir()
        try:
            from PIL import Image
        except ImportError:  # pragma: no cover - CI normally has Pillow
            Image = None
        for index, color in enumerate(((0, 0, 0), (255, 255, 255), (0, 0, 0), (255, 0, 0))):
            path = frames / f"frame-{index}.png"
            if Image is not None:
                image = Image.new("RGB", (24, 24), color)
                # Add a deterministic spatial pattern so normalized image
                # vectors retain visual novelty rather than all being constant.
                for x in range(0, 24, 2 + index):
                    for y in range(0, 24, 2 + index):
                        image.putpixel((x, y), (255 - color[0], 255 - color[1], 255 - color[2]))
                image.save(path)
            else:
                path.write_bytes(bytes([index + 1]) * 32)
        manifest = root / "video.json"
        manifest.write_text(json.dumps({
            "video_id": "synthetic",
            "frames": [
                {"index": i, "timestamp_seconds": float(i), "path": f"frames/frame-{i}.png"}
                for i in range(4)
            ],
        }), encoding="utf-8")
        return manifest

    def test_estimate_route_is_atomic_and_heading_is_explicitly_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._manifest(root)
            output = root / "trajectories" / "synthetic.json"
            result = estimate_route(manifest, output, smoothing_radius=0)
            self.assertTrue(output.exists())
            self.assertFalse(output.with_name(output.name + ".partial").exists())
            self.assertEqual(result["video_id"], "synthetic")
            self.assertEqual(len(result["observations"]), 4)
            self.assertTrue(all(item["heading_delta_degrees"] is None for item in result["observations"]))
            self.assertTrue(all(item["uncertainty"] >= 0.8 for item in result["observations"]))
            self.assertGreaterEqual(len(result["turn_candidates"]), 1)

    def test_color_descriptor_separates_same_luminance_images(self):
        try:
            from PIL import Image
        except ImportError:  # pragma: no cover
            self.skipTest("Pillow unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            red = root / "red.png"; blue = root / "blue.png"
            Image.new("RGB", (20, 20), (220, 20, 20)).save(red)
            Image.new("RGB", (20, 20), (20, 20, 220)).save(blue)
            self.assertIsNotNone(_image_vector(red))
            self.assertGreater(_distance(_image_vector(red), _image_vector(blue)) or 0.0, 0.05)

    def test_rejects_frame_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "video.json"
            manifest.write_text(json.dumps({
                "frames": [{"index": 0, "timestamp_seconds": 0, "path": "../outside.png"}],
            }), encoding="utf-8")
            with self.assertRaises(ValueError):
                estimate_route(manifest, root / "out.json")

    def test_smoothing_radius_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._manifest(root)
            with self.assertRaises(ValueError):
                estimate_route(manifest, root / "out.json", smoothing_radius=-1)


if __name__ == "__main__":
    unittest.main()
