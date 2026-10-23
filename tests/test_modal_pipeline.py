import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import kilometer_vision.jobs.modal_pipeline as pipeline


class MetadataAuditTests(unittest.TestCase):
    def record(self, **overrides):
        record = {
            "id": 0,
            "question": "synthetic question",
            "options": ["A", "B", "C", "D", "E"],
            "video_url": "https://www.youtube.com/watch?v=video_1",
            "start_frame": 25,
            "end_frame": 75,
            "image/original_frame_rate": 25,
            "start_seconds": 1,
            "end_seconds": 3,
        }
        record.update(overrides)
        return record

    def test_validates_five_options_and_frame_seconds(self):
        payload = {"video_1": {"0": self.record()}}
        audit = pipeline.validate_metadata(payload, expected_counts=(1, 1))
        self.assertTrue(audit.valid, audit.errors)
        self.assertEqual(audit.video_count, 1)
        self.assertEqual(audit.question_count, 1)

        bad = {"video_1": {"0": self.record(options=["A"], end_seconds=9)}}
        audit = pipeline.validate_metadata(bad)
        self.assertFalse(audit.valid)
        self.assertTrue(any("five options" in error for error in audit.errors))
        self.assertTrue(any("frame/time mismatch" in error for error in audit.errors))

    def test_referenced_image_must_be_in_matching_archive(self):
        payload = {
            "video_1": {
                "0": self.record(image_paths=["candidate.png"]),
            }
        }
        missing = pipeline.validate_metadata(payload, image_members=["other.png"])
        self.assertFalse(missing.valid)
        self.assertTrue(any("missing referenced image" in error for error in missing.errors))

        present = pipeline.validate_metadata(payload, image_members=["archive-prefix/candidate.png"])
        self.assertTrue(present.valid, present.errors)


class FrameManifestTests(unittest.TestCase):
    def test_build_is_bounded_atomic_and_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            shared = base / "shared" / "walking-tours" / "synthetic_video"
            source = shared / "tour.mkv"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"synthetic source; it is never copied")
            root = base / "volume"
            calls = []

            def fake_extract(source_path, destination, timestamp, **kwargs):
                calls.append((source_path, timestamp))
                Path(destination).parent.mkdir(parents=True, exist_ok=True)
                Path(destination).write_bytes(f"frame at {timestamp}".encode())

            metadata = {"duration_seconds": 2.4, "frame_rate": 10.0, "frame_count": 24, "width": 1920, "height": 1080, "probe": "synthetic"}
            with patch.object(pipeline, "probe_video", return_value=metadata), patch.object(pipeline, "extract_frame", side_effect=fake_extract):
                first = pipeline.build_frame_manifests(source_root=shared.parent, root=root, sample_fps=2, max_frames=3, frame_width=64)
                second = pipeline.build_frame_manifests(source_root=shared.parent, root=root, sample_fps=2, max_frames=3, frame_width=64)

            self.assertEqual(first["videos"][0]["status"], "built")
            self.assertEqual(first["videos"][0]["frames"], 3)
            self.assertEqual(second["videos"][0]["status"], "skipped")
            self.assertEqual(len(calls), 3)
            manifest = root / "frames" / "synthetic_video.json"
            features = root / "frames" / "synthetic_video.features.json"
            self.assertTrue(manifest.exists())
            self.assertTrue(features.exists())
            self.assertFalse((root / "frames" / "synthetic_video.json.partial").exists())
            self.assertTrue((root / "logs" / "frame_build.json").exists())
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual([frame["timestamp_seconds"] for frame in payload["frames"]], [0.0, 0.5, 1.0])
            self.assertFalse((root / "synthetic_video.mkv").exists())

    def test_sampling_rejects_invalid_timing_and_is_capped(self):
        with self.assertRaises(ValueError):
            pipeline._frame_timestamps({"duration_seconds": 0, "frame_rate": 10}, 1, 10)
        samples = pipeline._frame_timestamps({"duration_seconds": 100, "frame_rate": 25, "frame_count": 2500}, 10, 4)
        self.assertEqual(len(samples), 4)
        self.assertEqual(samples[-1][0], 3)


class CompilerAndImageIndexTests(unittest.TestCase):
    @staticmethod
    def png_bytes(width=2, height=3):
        # Header-only bytes are enough to verify dependency-light dimensions;
        # Pillow, when installed, may report the descriptor as undecodable.
        return b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + width.to_bytes(4, "big") + height.to_bytes(4, "big") + b"\x08\x02\x00\x00\x00"

    def test_compiler_classifies_families_and_excludes_ground_truth(self):
        records = [
            ("compass", "Which direction are you facing after turning left?"),
            ("distance", "How far is the landmark in meters?"),
            ("map", "Which map route matches the walk?"),
            ("landmark", "Did this reference image appear?"),
        ]
        for expected, text in records:
            compiled = pipeline.compile_question("video", "q", {"question": text, "options": ["north", "south", "east", "west", "left"], "answer": "SECRET", "answer_id": 4})
            self.assertEqual(compiled["family"], expected)
            self.assertNotIn("answer", json.dumps(compiled))
            self.assertEqual(compiled["option_differences"]["varying_directions"], ["east", "left", "north", "south", "west"])

    def test_image_index_is_atomic_and_only_indexes_references(self):
        payload = {"video_1": {"0": {
            "question": "Which map route matches?",
            "options": ["A", "B", "C", "D", "E"],
            "image_paths": ["candidate.png"],
            "answer": "SECRET", "answer_id": 2,
        }}}
        image = self.png_bytes()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "volume"
            pipeline.bootstrap_volume(root)
            manifest = root / "manifests" / "validation.json"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            archive_path = root / "manifests" / "validation_images.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("prefix/candidate.png", image)
            original_digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
            result = pipeline.index_split_images("validation", root=root)
            indexed = json.loads(Path(result["images"]).read_text(encoding="utf-8"))
            descriptor = indexed["images"][0]["descriptor"]
            self.assertEqual(descriptor["sha256"], hashlib.sha256(image).hexdigest())
            self.assertEqual((descriptor["width"], descriptor["height"]), (2, 3))
            self.assertEqual(indexed["indexed_count"], 1)
            self.assertEqual(hashlib.sha256(archive_path.read_bytes()).hexdigest(), original_digest)
            self.assertFalse((root / "route-features" / "validation_images.json.partial").exists())
            compiled = json.loads((root / "route-features" / "validation_questions.json").read_text(encoding="utf-8"))
            self.assertNotIn("answer", json.dumps(compiled))
            audit = json.loads((root / "logs" / "validation_compiler_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["classified_count"], 1)
            self.assertEqual(audit["ground_truth_fields_read"], [])


class VolumeActionTests(unittest.TestCase):
    def test_bootstrap_and_atomic_ingest_are_idempotent(self):
        record = {
            "id": 0,
            "question": "synthetic",
            "options": ["A", "B", "C", "D", "E"],
            "start_frame": 0,
            "end_frame": 10,
            "image/original_frame_rate": 10,
            "start_seconds": 0,
            "end_seconds": 1,
            "image_paths": ["candidate.png"],
        }
        payload = {"synthetic_video": {"0": record}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "volume" / "2707"
            result = pipeline.bootstrap_volume(root)
            self.assertEqual(len(result["directories"]), len(pipeline.CANONICAL_DIRS))

            with tempfile.TemporaryDirectory() as source_dir:
                source = Path(source_dir)
                metadata = source / "metadata.json"
                metadata.write_text(json.dumps(payload), encoding="utf-8")
                archive = source / "images.zip"
                with zipfile.ZipFile(archive, "w") as output:
                    output.writestr("prefix/candidate.png", b"tiny synthetic image")
                sources = {"metadata": metadata.as_uri(), "images": archive.as_uri()}
                with patch.object(pipeline, "EXPECTED_COUNTS", {"validation": (1, 1)}):
                    first = pipeline.ingest_split("validation", root=root, sources=sources)
                    second = pipeline.ingest_split("validation", root=root, sources=sources)

            self.assertTrue(first["audit"]["valid"])
            self.assertEqual(first["manifest"], second["manifest"])
            self.assertFalse((root / "manifests" / "validation.json.partial").exists())
            self.assertFalse((root / "manifests" / "validation_images.zip.partial").exists())
            self.assertTrue((root / "logs" / "validation_audit.json").exists())


if __name__ == "__main__":
    unittest.main()
