"""Alternate description runner using an OpenRouter vision-model endpoint.

Requires ``OPENROUTER_API_KEY`` and the panel artifacts from ``test_panels.py``.
Change only ``OPENROUTER_MODEL`` in .env; model IDs and OpenRouter links work.
Use ``--model`` for a one-run override. Response formatting is automatic.

Examples
--------
    python test_description_openrouter.py "Manga title" --model google/gemma-4-26b-a4b-it:free
    $env:OPENROUTER_MODEL="provider/vision-model-id"
    python test_description_openrouter.py "Manga title"
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import config
from common.json_io import write_json
from common.utils import find_chapter_dirs, find_manga_dir
from description.describer import describe_manga
from description.openrouter import OpenRouterDescriptionClient, normalize_model_id


def _load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE entries without overriding existing variables."""
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
            return normalize_model_id(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError(str(error)) from error

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "title", nargs="?", default=config.MANGA_TITLE or None,
        help="Manga folder under input/ (auto-detected when omitted).",
    )
    parser.add_argument(
        "--model",
        type=model_id,
        default=os.getenv("OPENROUTER_MODEL", config.OPENROUTER_MODEL),
        help="OpenRouter model ID or model-page URL (normally set OPENROUTER_MODEL in .env).",
    )
    parser.add_argument(
        "--response-format", choices=("auto", "json_schema", "json_object", "text"),
        default=os.getenv("OPENROUTER_RESPONSE_FORMAT", "auto"),
        help="Automatically detect provider capabilities; override only for custom endpoints.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        help="OpenRouter-compatible API base URL.",
    )
    parser.add_argument(
        "--output", type=Path,
        help="Output JSON path (default: output/<manga>_descriptions.json).",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> dict:
    _load_env_file(config.BASE_DIR / ".env")
    args = _parser().parse_args(argv)
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key or api_key.startswith("HIER_"):
        raise SystemExit(
            "OPENROUTER_API_KEY fehlt. Trage den Key in der Datei .env ein."
        )

    manga_dir = find_manga_dir(config.INPUT_DIR, args.title)
    chapter_dirs = find_chapter_dirs(manga_dir)
    manga_output_dir = config.OUTPUT_DIR / manga_dir.name
    output_file = args.output or config.OUTPUT_DIR / f"{manga_dir.name}_descriptions.json"
    if not output_file.is_absolute():
        output_file = config.BASE_DIR / output_file

    client = OpenRouterDescriptionClient(
        api_key=api_key,
        model=args.model,
        base_url=args.base_url,
        timeout=args.timeout,
        max_retries=args.max_retries,
        app_url=os.getenv("OPENROUTER_APP_URL") or None,
        response_format=args.response_format,
    )
    print(
        f"Manga: {manga_dir.name}  |  Chapters: {len(chapter_dirs)}  |  "
        f"Model: {args.model}\n",
        flush=True,
    )
    client.prepare()
    print(f"Response format: {client.response_format} (locally validated)\n", flush=True)
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
    main()
