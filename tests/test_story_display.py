import copy
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from story_writer.display import build_display_groups, refresh_panel_timeline


class StoryDisplayTests(unittest.TestCase):
    def fixture(self, root, sizes):
        segments, timeline = [], []
        for index, size in enumerate(sizes):
            image = root / f"panel_{index}.png"
            Image.new("RGB", size, "white").save(image)
            segments.append({"primary_panel_id": f"p{index}", "panel_image": str(image),
                             "chapter": "chapter_001", "page": "00001.jpg"})
            timeline.append({"panel_id": f"p{index}", "panel_image": str(image),
                             "chapter": "chapter_001", "start_word": index * 25,
                             "end_word": (index + 1) * 25,
                             "estimated_start_seconds": index * 10,
                             "estimated_end_seconds": (index + 1) * 10})
        return segments, timeline

    def test_tall_panels_share_frame_in_reading_order_and_keep_timing(self):
        with tempfile.TemporaryDirectory() as temporary:
            segments, timeline = self.fixture(Path(temporary), [(300, 900)] * 3)
            original = copy.deepcopy(timeline)
            groups = build_display_groups(segments, timeline)
            self.assertEqual([g["panel_ids"] for g in groups], [["p0", "p1"], ["p2"]])
            first = groups[0]
            self.assertEqual(first["layout"], "side_by_side_rtl")
            self.assertGreater(first["placements"][0]["x"], first["placements"][1]["x"])
            self.assertEqual((first["start_word"], first["end_word"]), (0, 50))
            self.assertEqual((first["estimated_start_seconds"], first["estimated_end_seconds"]), (0, 20))
            self.assertEqual(timeline, original)
            for group in groups:
                for placement in group["placements"]:
                    self.assertGreaterEqual(placement["x"], 0)
                    self.assertGreaterEqual(placement["y"], 0)
                    self.assertLessEqual(placement["x"] + placement["width"], 1)
                    self.assertLessEqual(placement["y"] + placement["height"], 1)
                    self.assertAlmostEqual(placement["width"] * 1920 / (placement["height"] * 1080), 1 / 3)

    def test_wide_panels_stack_but_well_fitting_panels_stay_single(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            segments, timeline = self.fixture(root, [(1200, 200)] * 2)
            self.assertEqual(build_display_groups(segments, timeline)[0]["layout"], "stacked")
            segments, timeline = self.fixture(root, [(1600, 900)] * 2)
            self.assertEqual([g["layout"] for g in build_display_groups(segments, timeline)], ["single", "single"])

    def test_no_pair_across_page_or_chapter_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            segments, timeline = self.fixture(Path(temporary), [(300, 900)] * 2)
            segments[1]["page"] = "00002.jpg"
            self.assertEqual(len(build_display_groups(segments, timeline)), 2)
            segments[1]["page"] = segments[0]["page"]
            segments[1]["chapter"] = timeline[1]["chapter"] = "chapter_002"
            self.assertEqual(len(build_display_groups(segments, timeline)), 2)

    def test_mismatched_order_or_sources_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            segments, timeline = self.fixture(Path(temporary), [(300, 900)] * 2)
            with self.assertRaises(ValueError):
                build_display_groups(segments, timeline[::-1])
            timeline[0]["panel_image"] = "different.png"
            with self.assertRaises(ValueError):
                build_display_groups(segments, timeline)

    def test_invalid_duration_or_missing_crop_does_not_drop_panel(self):
        with tempfile.TemporaryDirectory() as temporary:
            segments, timeline = self.fixture(Path(temporary), [(300, 900)])
            timeline[0]["estimated_end_seconds"] = -1
            with self.assertRaises(ValueError):
                build_display_groups(segments, timeline)
            timeline[0]["estimated_end_seconds"] = 10
            Path(timeline[0]["panel_image"]).unlink()
            with self.assertRaises(FileNotFoundError):
                build_display_groups(segments, timeline)

    def test_refresh_preserves_story_and_timing_and_backs_up_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            segments, rows = self.fixture(root, [(300, 900)] * 2)
            metadata = {"input_fingerprint": "current", "run_id": "run1"}
            story = {"manga": "Example", "segments": segments, "metadata": metadata}
            timeline = {"manga": "Example", "timeline": rows, "metadata": metadata}
            story_path, timeline_path = root / "story.json", root / "timeline.json"
            story_path.write_text(json.dumps(story), encoding="utf-8")
            timeline_path.write_text(json.dumps(timeline), encoding="utf-8")
            original_story, original_timeline = story_path.read_bytes(), timeline_path.read_bytes()
            updated = refresh_panel_timeline(story_path, timeline_path)
            self.assertEqual(updated["timeline"], rows)
            self.assertEqual(story_path.read_bytes(), original_story)
            backups = list(root.glob("timeline.before_layout_*.json"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), original_timeline)
            refresh_panel_timeline(story_path, timeline_path)
            self.assertEqual(len(list(root.glob("timeline.before_layout_*.json"))), 1)
            story["metadata"]["input_fingerprint"] = "stale"
            story_path.write_text(json.dumps(story), encoding="utf-8")
            with self.assertRaises(ValueError):
                refresh_panel_timeline(story_path, timeline_path)
