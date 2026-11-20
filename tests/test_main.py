"""Orchestration checks; no model, network, speech, or rendering runs."""
from pathlib import Path
import subprocess
import tempfile
import sys
import unittest
from unittest.mock import patch

import main as pipeline


class PipelineTests(unittest.TestCase):
    def run_pipeline(self, argv, codes):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "input" / "Example").mkdir(parents=True)
            codes = iter(codes)

            def stage(command, log, name, timeout):
                log.touch()
                return next(codes)

            with patch.object(pipeline.config, "INPUT_DIR", root / "input"), \
                 patch.object(pipeline.config, "OUTPUT_DIR", root / "output"), \
                 patch.object(pipeline, "_run_stage", side_effect=stage) as run, \
                 patch.object(pipeline.subprocess, "run", side_effect=lambda *a, **kw: subprocess.CompletedProcess([], next(codes))) as download:
                code = pipeline.main(argv)
                calls = download.call_args_list + run.call_args_list
                logs = [p.name for p in (root / "output").rglob("*.log")]
            return code, calls, logs

    def test_default_runs_all_stages_with_gemini(self):
        code, calls, logs = self.run_pipeline(["Example"], [0] * 5)
        self.assertEqual(code, 0)
        self.assertEqual([Path(c.args[0][-2]).name for c in calls], [
            "test_panels.py", "test_description_google.py", "test_character_reid.py", "test_write_storry.py", "test_video.py"])
        self.assertTrue(all(c.args[0][-1] == "Example" for c in calls))
        self.assertEqual(len(logs), 5)

    def test_failure_stops_later_stages_and_preserves_exit_code(self):
        code, calls, logs = self.run_pipeline(["Example"], [0, 7])
        self.assertEqual(code, 7)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(logs), 2)

    def test_upload_is_explicit_and_only_attached_to_video(self):
        code, calls, _ = self.run_pipeline(["Example", "--upload", "--upload-settings", "youtube_upload.json"], [0] * 5)
        self.assertEqual(code, 0)
        self.assertTrue(all("--upload" not in c.args[0] for c in calls[:-1]))
        self.assertIn("--upload", calls[-1].args[0])
        self.assertEqual(Path(calls[-1].args[0][-1]), Path("youtube_upload.json").resolve())

    def test_upload_failure_propagates_from_video_stage(self):
        code, calls, _ = self.run_pipeline(["Example", "--start-at", "video", "--upload"], [9])
        self.assertEqual(code, 9)
        self.assertEqual(len(calls), 1)

    def test_resume_and_explicit_local_backend(self):
        code, calls, _ = self.run_pipeline(["Example", "--start-at", "description", "--description-backend", "qwen"], [0] * 4)
        self.assertEqual(code, 0)
        self.assertEqual(Path(calls[0].args[0][-2]).name, "test_description.py")
        self.assertEqual(len(calls), 4)

    def test_download_failure_stops_before_processing(self):
        code, calls, logs = self.run_pipeline(["Example", "--download"], [4])
        self.assertEqual(code, 4)
        self.assertEqual(len(calls), 1)
        self.assertEqual(Path(calls[0].args[0][3]).name, "test_download.py")
        self.assertEqual(logs, [])

    def test_stage_streams_progress_and_retains_full_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "stage.log"
            with patch("builtins.print") as output:
                code = pipeline._run_stage([sys.executable, "-u", "-c",
                    "print('raw diagnostics'); print('[progress] Story [##] 1/1')"], log, "story", 10)
            self.assertEqual(code, 0)
            self.assertIn("raw diagnostics", log.read_text(encoding="utf-8"))
            self.assertTrue(any("[progress]" in str(c) for c in output.call_args_list))

    def test_watchdog_stops_retries_even_when_output_keeps_arriving(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "stage.log"
            code = pipeline._run_stage([sys.executable, "-u", "-c",
                "import time\nwhile True:\n print('retrying', flush=True)\n time.sleep(.02)"], log, "story", .3)
            self.assertEqual(code, 124)

    def test_invalid_watchdog_timeout_is_rejected(self):
        for value in ("0", "-1", "nan", "inf"):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                pipeline.main(["Example", "--stall-timeout", value])
