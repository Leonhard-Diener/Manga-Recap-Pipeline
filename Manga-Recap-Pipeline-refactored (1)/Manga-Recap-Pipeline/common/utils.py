"""Small helpers shared by the downloader, cleanup, panel and description
steps, so each step doesn't reimplement natural sorting, image discovery,
chapter discovery, or moving model inputs onto the GPU.
"""

from __future__ import annotations

import re
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def natural_sort_key(path: Path) -> list[object]:
    """Sort key so 'page2' comes before 'page10'."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def list_images(directory: Path) -> list[Path]:
    """Return all supported image files in *directory*, naturally sorted."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    files = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    return sorted(files, key=natural_sort_key)


def find_manga_dir(input_root: Path, manga_title: str | None = None) -> Path:
    """Return the manga directory inside *input_root*.

    If *manga_title* is given, that exact subdirectory is used. Otherwise
    the single existing subdirectory of *input_root* is used, and an error
    is raised if there is none or more than one (to avoid silently picking
    the wrong manga).
    """
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_root}")

    if manga_title:
        manga_dir = input_root / manga_title
        if not manga_dir.is_dir():
            raise FileNotFoundError(f"Manga directory not found: {manga_dir}")
        return manga_dir

    subdirs = [path for path in sorted(input_root.iterdir()) if path.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No manga folder found in {input_root}")
    if len(subdirs) > 1:
        print(f"Multiple manga folders found, using first: {subdirs[0].name}")
    return subdirs[0]


def find_chapter_dirs(manga_dir: Path) -> list[Path]:
    """Return all chapter subdirectories of *manga_dir* that contain images,
    naturally sorted (chapter_2 before chapter_10).
    """
    chapters = [
        path
        for path in manga_dir.iterdir()
        if path.is_dir()
        and any(f.suffix.lower() in IMAGE_SUFFIXES for f in path.iterdir())
    ]
    if not chapters:
        raise FileNotFoundError(f"No chapter directories with images found in {manga_dir}")
    return sorted(chapters, key=natural_sort_key)


def move_to_device(inputs: dict, device, dtype=None) -> dict:
    """Move a processor's tensor dict onto *device*, casting floats to *dtype*.

    Only imports torch when actually called, so modules that don't need the
    Qwen models (e.g. the downloader or panel extractor) don't require torch
    just to use the other helpers in this file.
    """
    import torch

    if dtype is None:
        dtype = torch.float16

    for key, value in inputs.items():
        if not torch.is_tensor(value):
            continue
        if value.is_floating_point():
            inputs[key] = value.to(device=device, dtype=dtype)
        else:
            inputs[key] = value.to(device=device)
    return inputs
