"""Download, clean edge pages, and extract manga panels in one pipeline."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import config
from downloader.download_chapter import download_chapter
from downloader.get_chapter_ids import get_chapter_ids
from downloader.get_manga_id import get_manga_id
from manga_recognizer import MangaLMMRecognizer
from panel_extractor import extract_panels


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class StoryPageClassifier(Protocol):
    """The small interface needed to keep or remove edge pages."""

    def is_story_page(self, image_path: Path) -> bool:
        """Return whether the complete page is narrative manga content."""


@dataclass(frozen=True, slots=True)
class EdgeCleanupResult:
    """Information about pages removed before panel extraction."""

    kept_pages: tuple[Path, ...]
    removed_pages: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Summary returned by :func:`run_pipeline`."""

    downloaded_pages: int
    cleanup: EdgeCleanupResult
    extracted_panels: int


def plan_edge_cleanup(
    page_directory: Path,
    classifier: StoryPageClassifier,
    *,
    pages: Sequence[Path] | None = None,
) -> EdgeCleanupResult:
    """Identify non-story images directly before and after a chapter.

    Interior pages are never evaluated. If the recognizer cannot find a story
    page at either edge, an error is raised before a caller can delete a file.
    """
    chapter_pages = list(pages) if pages is not None else _image_pages(page_directory)
    if not chapter_pages:
        raise FileNotFoundError(f"No image pages found in {page_directory}")

    first_story_index = _find_story_page(chapter_pages, classifier)
    last_story_index = _find_story_page(reversed(chapter_pages), classifier)

    if first_story_index is None or last_story_index is None:
        raise RuntimeError(
            "MangaLMM could not find a story page at both chapter edges; "
            "no images were deleted."
        )

    last_story_index = len(chapter_pages) - 1 - last_story_index
    if first_story_index > last_story_index:
        raise RuntimeError("Invalid story-page boundaries; no images were deleted.")

    removed = tuple(
        chapter_pages[:first_story_index] + chapter_pages[last_story_index + 1 :]
    )
    return EdgeCleanupResult(
        kept_pages=tuple(chapter_pages[first_story_index : last_story_index + 1]),
        removed_pages=removed,
    )


def edge_cleanup(
    page_directory: Path,
    classifier: StoryPageClassifier,
    *,
    pages: Sequence[Path] | None = None,
) -> EdgeCleanupResult:
    """Delete only confirmed non-story images at a chapter's two edges."""
    cleanup = plan_edge_cleanup(page_directory, classifier, pages=pages)
    for image_path in cleanup.removed_pages:
        image_path.unlink()
    return cleanup


def _find_story_page(
    pages: Iterable[Path],
    classifier: StoryPageClassifier,
) -> int | None:
    """Return the position of the first story page in an iterable."""
    for index, image_path in enumerate(pages):
        if classifier.is_story_page(image_path):
            return index
    return None


def _image_pages(page_directory: Path) -> list[Path]:
    if not page_directory.is_dir():
        raise NotADirectoryError(f"Downloaded page directory not found: {page_directory}")
    return sorted(
        (path for path in page_directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: (not path.stem.isdecimal(), int(path.stem) if path.stem.isdecimal() else path.name),
    )


def _chapter_pages(
    page_directory: Path,
    *,
    start_page: int,
    page_count: int,
) -> list[Path]:
    """Return the files downloaded for one chapter, in their page order."""
    end_page = start_page + page_count
    pages = [
        path
        for path in _image_pages(page_directory)
        if path.stem.isdecimal() and start_page <= int(path.stem) < end_page
    ]
    if not pages:
        raise FileNotFoundError(
            f"No downloaded pages found for the chapter starting at {start_page}."
        )
    return pages


async def run_pipeline(
    manga_title: str,
    *,
    language: str = "en",
    download_all_chapters: bool = config.DOWNLOAD_ALL_CHAPTERS,
    recognizer: StoryPageClassifier | None = None,
    manga_lookup: Callable[[str], str] = get_manga_id,
    chapter_lookup: Callable[[str, str], list[str]] = get_chapter_ids,
    downloader: Callable[[str, str, int], Awaitable[int]] = download_chapter,
    extractor: Callable[..., list[object]] = extract_panels,
) -> PipelineResult:
    """Run downloader -> VLM edge recognition -> existing panel extractor."""
    manga_id = manga_lookup(manga_title)
    chapter_ids = chapter_lookup(manga_id, language)
    selected_chapters = chapter_ids if download_all_chapters else chapter_ids[:1]
    if not selected_chapters:
        raise ValueError("No chapters were selected for download.")

    next_page_number = 1
    page_directory = config.INPUT_DIR / manga_title
    page_recognizer = recognizer
    removed_pages: list[Path] = []

    for chapter_index, chapter_id in enumerate(selected_chapters, start=1):
        print(f"Downloading chapter {chapter_index}/{len(selected_chapters)}...")
        chapter_start = next_page_number
        page_count = await downloader(chapter_id, manga_title, next_page_number)
        next_page_number += page_count

        if page_recognizer is None:
            page_recognizer = MangaLMMRecognizer(
                model_id=config.MANGA_LMM_MODEL_ID,
                device_map=config.MANGA_LMM_DEVICE_MAP,
                cache_dir=config.MANGA_LMM_CACHE_DIR,
            )
        chapter_pages = _chapter_pages(
            page_directory,
            start_page=chapter_start,
            page_count=page_count,
        )
        cleanup = edge_cleanup(page_directory, page_recognizer, pages=chapter_pages)
        removed_pages.extend(cleanup.removed_pages)

    cleanup = EdgeCleanupResult(
        kept_pages=tuple(_image_pages(page_directory)),
        removed_pages=tuple(removed_pages),
    )
    print(f"Removed {len(removed_pages)} non-story edge page(s).")

    panels = extractor(
        input_dir=page_directory,
        output_dir=config.OUTPUT_DIR,
        model_path=config.MODEL_PATH,
        image_size=config.IMAGE_SIZE,
        panel_confidence=config.PANEL_CONFIDENCE,
        text_confidence=config.TEXT_CONFIDENCE,
        body_confidence=config.BODY_CONFIDENCE,
        enable_tiled_fallback=config.ENABLE_TILED_FALLBACK,
        fallback_uncovered_ratio=config.FALLBACK_UNCOVERED_RATIO,
        fallback_grid=config.FALLBACK_GRID,
        fallback_overlap=config.FALLBACK_OVERLAP,
        minimum_panel_ratio=config.MINIMUM_PANEL_RATIO,
        duplicate_iou=config.DUPLICATE_IOU,
        duplicate_containment=config.DUPLICATE_CONTAINMENT,
        text_padding_x=config.TEXT_PADDING_X,
        text_padding_y=config.TEXT_PADDING_Y,
        text_safety_pixels=config.TEXT_SAFETY_PIXELS,
        body_padding_x=config.BODY_PADDING_X,
        body_padding_y=config.BODY_PADDING_Y,
        body_safety_pixels=config.BODY_SAFETY_PIXELS,
        max_content_expansion_ratio=config.MAX_CONTENT_EXPANSION_RATIO,
        jpeg_quality=config.JPEG_QUALITY,
        save_debug=config.SAVE_DEBUG,
    )
    return PipelineResult(
        downloaded_pages=next_page_number - 1,
        cleanup=cleanup,
        extracted_panels=len(panels),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("title", help="MangaDex title to download")
    parser.add_argument("--language", default="en", help="Chapter language (default: en)")
    parser.add_argument(
        "--all-chapters",
        action=argparse.BooleanOptionalAction,
        default=config.DOWNLOAD_ALL_CHAPTERS,
        help="Download every chapter instead of only the first one",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = asyncio.run(
        run_pipeline(
            args.title,
            language=args.language,
            download_all_chapters=args.all_chapters,
        )
    )
    print(
        f"Done: {result.downloaded_pages} downloaded, "
        f"{len(result.cleanup.removed_pages)} edge pages removed, "
        f"{result.extracted_panels} panels extracted."
    )


if __name__ == "__main__":
    main()
