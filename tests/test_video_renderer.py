import tempfile
import json
from pathlib import Path
import unittest
import wave
from unittest.mock import patch

from PIL import Image

from test_video import align_groups, panel_backdrop, combine_speech, load_inputs, render_frame, speech_files


class VideoTests(unittest.TestCase):
    def test_upload_uses_exact_new_render_and_requires_opt_in(self):
        import test_video
        with tempfile.TemporaryDirectory() as temporary:
            manga = Path(temporary) / "Example"
            exact_video = manga / "video_runs" / "specific_run" / "recap.mp4"
            with patch.object(test_video, "find_manga_dir", return_value=manga), \
                 patch.object(test_video, "create_video", return_value=exact_video), \
                 patch("youtube_uploader.upload_video") as upload:
                self.assertEqual(test_video.main(["Example"]), exact_video)
                upload.assert_not_called()
                test_video.main(["Example", "--upload", "--upload-settings", "custom.json"])
                upload.assert_called_once_with(exact_video, "Example", Path("custom.json"))

    def test_render_failure_cannot_trigger_upload(self):
        import test_video
        with patch.object(test_video, "find_manga_dir", return_value=Path("Example")), \
             patch.object(test_video, "create_video", side_effect=RuntimeError("render failed")), \
             patch("youtube_uploader.upload_video") as upload:
            with self.assertRaisesRegex(RuntimeError, "render failed"):
                test_video.main(["Example", "--upload"])
            upload.assert_not_called()

    def test_backdrop_softens_ink_and_desaturates_color(self):
        panel = Image.new('RGB', (160, 100), 'black')
        panel.paste('yellow', (80, 0, 160, 100))
        row = list(panel_backdrop(panel, panel.size).crop((0, 50, 160, 51)).getdata())
        self.assertTrue(all(r == g == b for r, g, b in row))
        self.assertGreater(row[0][0], 0)
        self.assertLess(row[0], row[80])
        self.assertLess(row[80], row[-1])

    def test_render_keeps_foreground_white_and_border_visible(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, frame = root / "panel.png", root / "frame.png"
            Image.new("RGB", (100, 200), "white").save(source)
            render_frame({"placements": [{"panel_image": str(source), "x": .25,
                                          "y": 0, "width": .5, "height": 1}]}, frame, (320, 180))
            with Image.open(frame) as result:
                self.assertEqual(result.mode, "RGB")
                self.assertEqual(result.getpixel((160, 90)), (255, 255, 255))
                self.assertIsNotNone(result.getcolors(maxcolors=320 * 180))
                self.assertIn((40, 40, 40), [color for _, color in result.getcolors(maxcolors=320 * 180)])

    def test_load_rejects_stale_story_and_preserves_valid_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "panel.png"
            Image.new("RGB", (100, 200), "white").save(image)
            segment = {"primary_panel_id": "a", "text": "A boy watches.",
                       "panel_image": str(image), "chapter": "chapter_001", "page": "1.jpg"}
            row = {"panel_id": "a", "panel_image": str(image), "chapter": "chapter_001",
                   "start_word": 0, "end_word": 3, "estimated_start_seconds": 0,
                   "estimated_end_seconds": 1}
            metadata = {"input_fingerprint": "hash", "run_id": "run"}
            story = {"manga": "Example", "metadata": metadata, "segments": [segment]}
            timeline = {"manga": "Example", "metadata": dict(metadata), "timeline": [row]}
            story_path, timeline_path = root / "story.json", root / "timeline.json"
            story_path.write_text(json.dumps(story), encoding="utf-8")
            timeline_path.write_text(json.dumps(timeline), encoding="utf-8")
            loaded, groups = load_inputs(story_path, timeline_path)
            self.assertEqual(loaded, story)
            self.assertEqual(groups[0]["panel_ids"], ["a"])
            timeline["metadata"]["run_id"] = "other"
            timeline_path.write_text(json.dumps(timeline), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_inputs(story_path, timeline_path)

    def test_tts_reuses_only_matching_text_and_voice(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, cache = root / "run", root / "cache"
            run.mkdir()
            segments = [{"text": "A boy watches."}]
            with patch("test_video.subprocess.run") as speak:
                paths = speech_files(segments, "voice A", cache, run)
                self.assertEqual(speak.call_count, 1)
                paths[0].write_bytes(b"cached")
                self.assertEqual(speech_files(segments, "voice A", cache, run), paths)
                self.assertEqual(speak.call_count, 1)
                other = speech_files(segments, "voice B", cache, run)
                self.assertNotEqual(other, paths)
                self.assertEqual(speak.call_count, 2)

    def test_real_audio_not_estimated_timing_controls_groups(self):
        segments = [{"primary_panel_id": p} for p in ("a", "b", "c")]
        groups = [{"panel_ids": ["a", "b"]}, {"panel_ids": ["c"]}]
        aligned = align_groups(groups, segments, [1.25, 2.5, 4.0])
        self.assertEqual(aligned[0]["audio_end_seconds"], 3.75)
        self.assertEqual(aligned[1]["audio_start_seconds"], 3.75)
        self.assertEqual(aligned[1]["audio_end_seconds"], 7.75)
        with self.assertRaises(ValueError):
            align_groups(groups[::-1], segments, [1, 2, 3])
        with self.assertRaises(ValueError):
            align_groups(groups, segments, [0, 2, 3])

    def test_combined_wav_preserves_samples_and_rejects_incompatible_audio(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files = [root / "a.wav", root / "b.wav"]
            for path, frames in zip(files, [8000, 4000]):
                with wave.open(str(path), "wb") as output:
                    output.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
                    output.writeframes(b"\0\0" * frames)
            durations = combine_speech(files, root / "all.wav")
            self.assertEqual(durations, [1.0, 0.5])
            with wave.open(str(root / "all.wav"), "rb") as audio:
                self.assertEqual(audio.getnframes(), 12000)
            with wave.open(str(files[1]), "wb") as output:
                output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                output.writeframes(b"\0\0" * 100)
            with self.assertRaises(ValueError):
                combine_speech(files, root / "invalid.wav")
