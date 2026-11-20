"""Render saved story narration as a 1080p recap with offline Windows TTS.

Usage: python test_video.py ["Manga title"] [--voice "Microsoft David Desktop"]
Requires FFmpeg on PATH and Windows System.Speech. Never runs the story model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import wave

from PIL import Image, ImageDraw, ImageFilter, ImageOps

import config
from common.json_io import write_json
from common.utils import find_manga_dir
from story_writer.display import DISPLAY_POLICY, VISUAL_HOLD_SECONDS, build_display_groups, is_visual_hold, narration_blocks


SPEECH_SCRIPT = r"""
param([string]$Manifest)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$jobs = Get-Content -LiteralPath $Manifest -Raw -Encoding UTF8 | ConvertFrom-Json
$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    $speaker.SelectVoice($jobs.voice)
    foreach ($job in $jobs.items) {
        try {
            $speaker.SetOutputToWaveFile($job.temporary)
            $speaker.Speak($job.text)
            $speaker.SetOutputToNull()
            Move-Item -LiteralPath $job.temporary -Destination $job.path -Force
        } finally {
            $speaker.SetOutputToNull()
        }
    }
} finally { $speaker.Dispose() }
"""


def panel_backdrop(image, size):
    small = tuple(max(1, n // 4) for n in size)
    blur = ImageOps.fit(image.convert("L"), small).filter(ImageFilter.GaussianBlur(max(1, small[1] * .023)))
    return ImageOps.colorize(blur, (54, 54, 54), (249, 249, 249)).resize(size, Image.Resampling.BICUBIC)


def render_frame(group, destination, size=(1920, 1080)):
    images = []
    for placement in group["placements"]:
        with Image.open(placement["panel_image"]) as source:
            images.append(source.convert("RGB"))
    canvas = panel_backdrop(images[0], size)
    margin, border = max(1, round(min(size) * .05)), max(1, round(min(size) / 720))
    inner_w, inner_h = size[0] - 2 * margin, size[1] - 2 * margin
    for placement, image in zip(group["placements"], images):
        box_w = max(1, round(placement["width"] * inner_w) - border * 2)
        box_h = max(1, round(placement["height"] * inner_h) - border * 2)
        fitted = ImageOps.contain(image, (box_w, box_h), Image.Resampling.LANCZOS)
        x = margin + round(placement["x"] * inner_w) + (round(placement["width"] * inner_w) - fitted.width) // 2
        y = margin + round(placement["y"] * inner_h) + (round(placement["height"] * inner_h) - fitted.height) // 2
        box = (x - border, y - border, x + fitted.width + border - 1, y + fitted.height + border - 1)
        shadow = Image.new('L', size)
        offset = max(1, margin // 5)
        ImageDraw.Draw(shadow).rectangle((box[0], box[1]+offset, box[2], box[3]+offset), fill=145)
        canvas.paste((0, 0, 0), mask=shadow.filter(ImageFilter.GaussianBlur(max(1, margin // 3))))
        ImageDraw.Draw(canvas).rectangle(box, fill=(40, 40, 40))
        canvas.paste(fitted, (x, y))
    canvas.save(destination)


def load_inputs(story_path, timeline_path):
    story = json.loads(story_path.read_text(encoding="utf-8"))
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    if not isinstance(story.get("manga"), str) or Path(story["manga"]).name != story["manga"] or story["manga"] in ("", ".", ".."):
        raise ValueError("Invalid manga name in saved story.")
    metadata = story.get("metadata", {})
    other = timeline.get("metadata", {})
    if (not metadata.get("input_fingerprint")
            or metadata["input_fingerprint"] != other.get("input_fingerprint")
            or metadata.get("run_id") != other.get("run_id")
            or story.get("manga") != timeline.get("manga")):
        raise ValueError("Saved story and timeline provenance do not match.")
    segments = story["segments"]
    blocks = narration_blocks(segments)
    scene_visuals = {index for block in blocks for index in block[:-1]}
    if not segments or any(not isinstance(s.get("text"), str) or
                           (not s["text"].strip() and not is_visual_hold(s) and i not in scene_visuals)
                           for i, s in enumerate(segments)):
        raise ValueError("Every story segment needs narration or an explicit visual hold.")
    ids = [s["primary_panel_id"] for s in segments]
    if len(set(ids)) != len(ids):
        raise ValueError("Story has duplicate primary panel IDs.")
    groups = build_display_groups(segments, timeline["timeline"])
    return story, groups


def speech_files(segments, voice, cache, run):
    cache.mkdir(parents=True, exist_ok=True)
    paths, jobs = [], []
    for segment in segments:
        if is_visual_hold(segment):
            paths.append(None)
            continue
        digest = hashlib.sha256(json.dumps(
            ["windows_system_speech_v1", voice, segment["text"]], ensure_ascii=False
        ).encode("utf-8")).hexdigest()
        path = cache / f"{digest}.wav"
        paths.append(path)
        if not path.exists():
            jobs.append({"text": segment["text"], "path": str(path.resolve()),
                         "temporary": str((run / f"{digest}.partial.wav").resolve())})
    if jobs:
        manifest = run / "tts_jobs.json"
        write_json(manifest, {"voice": voice, "items": jobs})
        script = run / "speak.ps1"
        script.write_text(SPEECH_SCRIPT, encoding="utf-8-sig")
        with (run / "tts.log").open("w", encoding="utf-8") as log:
            subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-File",
                            str(script.resolve()), str(manifest.resolve())],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    return paths


def combine_speech(paths, destination):
    durations = []
    first_speech = next((path for path in paths if path is not None), None)
    if first_speech is None:
        raise ValueError("Cannot render a story with no narration at all.")
    with wave.open(str(first_speech), "rb") as source:
        parameters = source.getparams()
    with wave.open(str(destination), "wb") as output:
        output.setparams(parameters)
        audio_format = (parameters.nchannels, parameters.sampwidth, parameters.framerate, parameters.comptype)
        for path in paths:
            if path is None:
                frames = round(parameters.framerate * VISUAL_HOLD_SECONDS)
                silence = b"\x80" if parameters.sampwidth == 1 else b"\x00"
                output.writeframes(silence * frames * parameters.nchannels * parameters.sampwidth)
                durations.append(frames / parameters.framerate)
                continue
            with wave.open(str(path), "rb") as source:
                fmt = (source.getnchannels(), source.getsampwidth(), source.getframerate(), source.getcomptype())
                if audio_format is None:
                    audio_format = fmt
                    output.setparams(source.getparams())
                if fmt != audio_format or source.getnframes() <= 0:
                    raise ValueError(f"Invalid or incompatible speech WAV: {path}")
                frames = source.readframes(source.getnframes())
                if len(frames) != source.getnframes() * fmt[0] * fmt[1]:
                    raise ValueError(f"Truncated speech WAV: {path}")
                output.writeframes(frames)
                durations.append(source.getnframes() / source.getframerate())
    return durations


def align_groups(groups, segments, durations):
    if len(durations) != len(segments) or any(d <= 0 for d in durations):
        raise ValueError("Speech durations do not match story segments.")
    if [p for g in groups for p in g["panel_ids"]] != [s["primary_panel_id"] for s in segments]:
        raise ValueError("Display groups do not cover story panels in order.")
    by_id = dict(zip((s["primary_panel_id"] for s in segments), durations))
    cursor, aligned = 0.0, []
    for group in groups:
        duration = sum(by_id[panel] for panel in group["panel_ids"])
        aligned.append({**group, "audio_start_seconds": cursor,
                        "audio_end_seconds": cursor + duration, "audio_duration_seconds": duration})
        cursor += duration
    return aligned


def create_video(story_path, timeline_path, output_root, voice="Microsoft David Desktop"):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not shutil.which("powershell.exe"):
        raise RuntimeError("Requires FFmpeg on PATH and Windows PowerShell/System.Speech.")
    story, groups = load_inputs(story_path, timeline_path)
    run = output_root / story["manga"] / "video_runs" / str(time.time_ns())
    run.mkdir(parents=True)
    print(f"Run files and logs: {run.resolve()}", flush=True)
    print(f"Generating speech for {len(story['segments'])} panels with {voice}...", flush=True)
    blocks = narration_blocks(story["segments"])
    speech_segments = [story["segments"][block[-1]] for block in blocks]
    paths = speech_files(speech_segments, voice, output_root / story["manga"] / "video_tts", run)
    audio = run / "narration.wav"
    block_durations = combine_speech(paths, audio)
    durations, panel_paths = [], []
    for block, duration, path in zip(blocks, block_durations, paths):
        durations.extend([duration / len(block)] * len(block))
        panel_paths.extend([path] * len(block))
    aligned = align_groups(groups, story["segments"], durations)
    write_json(run / "audio_timeline.json", {
        "manga": story["manga"], "voice": voice, "display_policy": DISPLAY_POLICY,
        "source_metadata": story["metadata"],
        "segments": [{"panel_id": s["primary_panel_id"], "duration_seconds": duration,
                       "speech_file": str(path.resolve()) if path is not None else None}
                      for s, duration, path in zip(story["segments"], durations, panel_paths)],
        "display_groups": aligned,
    })
    lines = []
    for index, group in enumerate(aligned):
        filename = f"frame_{index:04d}.png"
        render_frame(group, run / filename)
        lines.extend([f"file '{filename}'", "option framerate 30",
                      f"duration {group['audio_duration_seconds']:.9f}"])
    lines.extend([f"file 'frame_{len(aligned) - 1:04d}.png'", "option framerate 30"])
    playlist = run / "frames.ffconcat"
    playlist.write_text("ffconcat version 1.0\n" + "\n".join(lines) + "\n", encoding="utf-8")
    temporary, final = run / "recap.partial.mp4", run / "recap.mp4"
    print(f"Rendering {len(aligned)} display groups ({sum(durations):.1f}s)...", flush=True)
    with (run / "ffmpeg.log").open("w", encoding="utf-8") as log:
        subprocess.run([ffmpeg, "-nostdin", "-y", "-f", "concat", "-safe", "0",
                        "-i", str(playlist.resolve()), "-i", str(audio.resolve()),
                        "-map", "0:v:0", "-map", "1:a:0", "-vf", "fps=30",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                        "-t", f"{sum(durations):.9f}", "-movflags", "+faststart", str(temporary.resolve())],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    temporary.replace(final)
    print(f"Video: {final.resolve()}", flush=True)
    return final


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("title", nargs="?", default=config.MANGA_TITLE or None)
    parser.add_argument("--voice", default="Microsoft David Desktop")
    parser.add_argument("--upload", action="store_true", help="Upload this completed render using youtube_upload.json.")
    parser.add_argument("--upload-settings", type=Path, default=config.BASE_DIR / "youtube_upload.json")
    args = parser.parse_args(argv)
    manga = find_manga_dir(config.INPUT_DIR, args.title)
    video = create_video(config.OUTPUT_DIR / f"{manga.name}_story.json",
                         config.OUTPUT_DIR / f"{manga.name}_panel_timeline.json",
                         config.OUTPUT_DIR, args.voice)
    if args.upload:
        from youtube_uploader import upload_video
        upload_video(video, manga.name, args.upload_settings)
    return video


if __name__ == "__main__":
    main()
