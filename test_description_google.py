"""Alternate description runner using the native Google Gemini API.

Requires ``GEMINI_API_KEY`` and panel artifacts from ``test_panels.py``.
Put the key and optional model settings in the repository's ``.env`` file.

Example:
    python -X utf8 test_description_google.py "Manga title"
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import config
from common.json_io import write_json
from common.utils import find_chapter_dirs, find_manga_dir
from description.describer import describe_manga
from description.google import GoogleDescriptionClient, normalize_google_model_id


def _load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if value[:1] == value[-1:] and value[:1] in {'"', "'"}:
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


def _parser() -> argparse.ArgumentParser:
    def model_id(value):
        try:
            return normalize_google_model_id(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError(str(error)) from error

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "title", nargs="?", default=config.MANGA_TITLE or None,
        help="Manga folder under input/ (auto-detected when omitted).",
    )
    parser.add_argument(
        "--model", type=model_id,
        default=os.getenv("GOOGLE_MODEL", "gemini-3.5-flash-lite"),
        help="Google model ID (normally set GOOGLE_MODEL in .env).",
    )
    parser.add_argument(
        "--media-resolution", choices=("low", "medium", "high"),
        default=os.getenv("GOOGLE_MEDIA_RESOLUTION", "high").lower(),
        help="Image token budget; high is recommended for manga text.",
    )
    parser.add_argument(
        "--output", type=Path,
        help="Output JSON path (default: output/<manga>_descriptions.json).",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=None,
                        help="Transient-error retries (default: keep waiting; 0 disables retries).")
    return parser


def main(argv: list[str] | None = None) -> dict:
    _load_env_file(config.BASE_DIR / ".env")
    args = _parser().parse_args(argv)
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key.startswith("YOUR_"):
        raise SystemExit("GEMINI_API_KEY is missing. Add it to the .env file.")

    manga_dir = find_manga_dir(config.INPUT_DIR, args.title)
    chapter_dirs = find_chapter_dirs(manga_dir)
    manga_output_dir = config.OUTPUT_DIR / manga_dir.name
    output_file = args.output or config.OUTPUT_DIR / f"{manga_dir.name}_descriptions.json"
    if not output_file.is_absolute():
        output_file = config.BASE_DIR / output_file

    client = GoogleDescriptionClient(
        api_key=api_key,
        model=args.model,
        media_resolution=args.media_resolution,
        timeout=args.timeout,
        max_retries=args.max_retries,
        response_dir=manga_output_dir / "google_responses",
    )
    print(
        f"Manga: {manga_dir.name}  |  Chapters: {len(chapter_dirs)}  |  "
        f"Model: {args.model}  |  Resolution: {args.media_resolution}\n",
        flush=True,
    )
    client.prepare()
    print(f"Thinking: HIGH (Gemini); API response archives: {manga_output_dir / 'google_responses'}", flush=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result = describe_manga(
        manga_output_dir,
        chapter_dirs,
        filter_non_story=True,
        backend=client,
        progress_callback=lambda partial: write_json(output_file, partial),
    )
    write_json(output_file, result)
    if result.get("pending_pages"):
        print(f"\nRun incomplete. Unresolved pages: {result['pending_pages']}")
        print("Completed pages are checkpointed. See chapter description_attempts/ for diagnostics.")
        return result
    pages = [
        page
        for chapter in result["page_filter"]["chapters"].values()
        for page in chapter.values()
    ]
    included = sum(page["included_in_story"] for page in pages)
    print(f"\nStory filter: {included} included, {len(pages) - included} excluded")
    print(f"\nSaved descriptions and character events to {output_file}")
    return result


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit("Stopped by user. Rerun the same command to resume saved progress.") from None
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(f"Stopped safely: {error}\n"
                         "Completed pages remain checkpointed; rerun after fixing the error.") from None
