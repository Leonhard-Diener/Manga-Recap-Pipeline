"""Run the manga recap pipeline, keeping standalone stage runners available."""
from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime
import subprocess
import sys
import os
import signal
import time
from collections import deque

import config
from common.utils import find_manga_dir


STAGES = ("panels", "description", "reid", "story", "video")


def _stop_stage(process):
    """Stop only this stage's process tree, including its owned model server."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if process.poll() is None:
        process.kill()
    process.wait(timeout=15)


def _run_stage(command, log, stage, stall_timeout):
    last_activity = heartbeat = time.monotonic()
    last_progress = None
    pending = ""
    with log.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(command, cwd=config.BASE_DIR, stdout=output, stderr=subprocess.STDOUT,
                                   start_new_session=os.name != "nt")
        try:
            with log.open(encoding="utf-8", errors="replace") as reader:
                while True:
                    # Only real completion updates reset the model-stage watchdog, not retry messages.
                    data = reader.read(65536)
                    now = time.monotonic()
                    if data:
                        pending += data
                        lines = pending.split("\n")
                        pending = lines.pop()
                        for line in lines:
                            if line.startswith("[progress]"):
                                print(line, flush=True)
                                if line != last_progress:
                                    last_activity = now
                                    last_progress = line
                            elif any(word in line.lower() for word in ("retry", "waiting", "failed", "unresolved")):
                                print(line, flush=True)
                        if stage not in {"description", "story"}:
                            last_activity = now
                    code = process.poll()
                    if code is not None and not data:
                        if pending:
                            print(pending, flush=True)
                        if code:
                            with log.open(encoding="utf-8", errors="replace") as failed:
                                print("".join(deque(failed, maxlen=8)), file=sys.stderr)
                        return code
                    if now - last_activity >= stall_timeout and code is None:
                        print(f"Stopped {stage}: no progress for {stall_timeout:g}s. Saved checkpoints retained; see {log}",
                              file=sys.stderr, flush=True)
                        _stop_stage(process)
                        return 124
                    if now - heartbeat >= 30:
                        print(f"{stage}: still running; {now - last_activity:.0f}s since last progress. Log: {log}", flush=True)
                        heartbeat = now
                    time.sleep(0.1)
        except BaseException:
            if process.poll() is None:
                _stop_stage(process)
            raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("title", nargs="?", default=config.MANGA_TITLE or None)
    parser.add_argument("--download", action="store_true", help="Download from MangaDex first (interactive selection).")
    parser.add_argument("--upload", action="store_true", help="Upload the newly rendered video to YouTube (opt-in).")
    parser.add_argument("--upload-settings", default=str(config.BASE_DIR / "youtube_upload.json"))
    parser.add_argument("--start-at", choices=STAGES, default="panels", help="Resume from this stage; earlier outputs must exist.")
    parser.add_argument("--description-backend", choices=("gemini", "qwen"), default="gemini",
                        help="Description provider; no automatic fallback (default: gemini).")
    parser.add_argument("--stall-timeout", type=float, default=1800,
                        help="Stop a stage after this many seconds without progress (default: 1800).")
    args = parser.parse_args(argv)
    if not 0 < args.stall_timeout < float("inf"):
        parser.error("--stall-timeout must be finite and greater than zero")
    if args.download:
        if not args.title:
            parser.error("--download requires a manga title (argument or config.MANGA_TITLE)")
        result = subprocess.run([sys.executable, "-X", "utf8", str(config.BASE_DIR / "test_download.py"), args.title],
                                cwd=config.BASE_DIR)
        if result.returncode:
            return result.returncode

    manga = find_manga_dir(config.INPUT_DIR, args.title)
    scripts = {
        "panels": "test_panels.py",
        "description": "test_description_google.py" if args.description_backend == "gemini" else "test_description.py",
        "reid": "test_character_reid.py",
        "story": "test_write_storry.py",
        "video": "test_video.py",
    }
    run_dir = config.OUTPUT_DIR / manga.name / "pipeline_runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"Manga: {manga.name}\nPipeline logs: {run_dir}", flush=True)
    for stage in STAGES[STAGES.index(args.start_at):]:
        log = run_dir / f"{stage}.log"
        print(f"Running {stage}… ({log})", flush=True)
        command = [sys.executable, "-u", "-X", "utf8", str(config.BASE_DIR / scripts[stage]), manga.name]
        if stage == "video" and args.upload:
            command += ["--upload", "--upload-settings", str(Path(args.upload_settings).resolve())]
        code = _run_stage(
            command,
            log, stage, args.stall_timeout,
        )
        if code:
            print(f"Stopped: {stage} exited with {code}. See {log}", file=sys.stderr)
            return code
        print(f"Completed {stage}.", flush=True)
    print(f"Pipeline complete. Video path is recorded in {run_dir / 'video.log'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
