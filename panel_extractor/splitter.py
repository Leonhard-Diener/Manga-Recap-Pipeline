"""Split merged panel detections back into their constituent panels.

Motivation
----------
The RT-DETR frame detector occasionally returns a single box for two panels
that are drawn without a visible gutter between them.  Detector confidence is
high in these cases, so no confidence threshold recovers the split.

This module tries two strategies, in order:

1. **Frame-line split** (cheap, no inference).  Manga panels are defined by a
   drawn black border stroke.  That stroke is thin, straight and spans the
   whole panel, whereas artwork is usually toned/curved and rarely spans the
   full box.  A long consecutive dark run inside a candidate box is therefore
   strong evidence of the frame line shared by two touching panels.
2. **AI fallback** (RT-DETR re-inference on a zoomed crop).  When no frame
   line can be found -- e.g. a character breaks out of the panel and covers
   the shared border -- the detector is asked again on the isolated, upscaled
   box.  Seeing the region at a larger effective resolution (cross-scale
   disagreement) it often separates the two panels it merged on the full page.

Content that overflows a panel is *not* this module's job: splitting is a
structural problem, overflow is handled later by the content-protection step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from .detector import predict
from .geometry import box_area, clip_box
from .postprocess import deduplicate
from .types import Detection

if TYPE_CHECKING:  # pragma: no cover - typing only
    import onnxruntime as ort


@dataclass
class SplitSettings:
    """Tunables for :func:`split_merged_panels`.

    All thresholds are deliberately conservative so the splitter only acts on
    clear evidence of a merge and never invents panels inside a single frame.
    """

    # Master switches.
    enable_frame_split: bool = True
    enable_ai_fallback: bool = True

    # Frame-line evidence.
    dark_threshold: int = 100        # grayscale value below which ink is "dark"
    min_line_ratio: float = 0.70     # fraction of the box width/height a line must span
    margin: float = 0.05             # ignore this fraction at each box edge
    max_line_thickness: int = 8      # px; thicker dark bands are art, not a frame line
    edge_gap_ratio: float = 0.02     # permitted uninked gap at each parent-box edge
    min_area_ratio: float = 0.04     # each resulting sub-box must be >= this page-area share
    max_depth: int = 3               # recursive splits per original box

    # AI fallback.
    ai_min_area_ratio: float = 0.15  # only ask the detector for boxes at least this big
    ai_min_line_ratio: float = 0.30  # partial frame line that flags a possible merge
    ai_min_frames: int = 2           # frames required to accept the AI split
    ai_target_size: int = 1024       # zoom the crop so its long side reaches this
    ai_max_upscale: float = 3.0      # cap on the zoom factor
    ai_min_frame_area_ratio: float = 0.08   # each accepted frame >= this share of the crop
    ai_max_frames: int = 4           # never return more frames than this


@dataclass(slots=True)
class _SplitContext:
    """Everything the recursive splitter needs to run."""

    ink: np.ndarray
    image: Image.Image
    session: "ort.InferenceSession"
    image_size: int
    panel_confidence: float
    text_confidence: float
    body_confidence: float
    settings: SplitSettings
    page_area: int
    width: int
    height: int


# ---------------------------------------------------------------------------
# Frame-line evidence
# ---------------------------------------------------------------------------


def _longest_run(mask: np.ndarray, axis: int) -> np.ndarray:
    """Return the longest consecutive ``True`` run along *axis* for every line.

    Fully vectorised: ``idx - last_false`` gives the current run length at
    every position, and its maximum is the longest run on that line.
    """
    if mask.size == 0:
        return np.zeros(mask.shape[1 - axis], dtype=np.int32)

    if axis == 1:
        index = np.arange(mask.shape[1], dtype=np.int32)[None, :]
        last_false = np.where(~mask, index, -1)
        np.maximum.accumulate(last_false, axis=1, out=last_false)
        return (index - last_false).max(axis=1)

    index = np.arange(mask.shape[0], dtype=np.int32)[:, None]
    last_false = np.where(~mask, index, -1)
    np.maximum.accumulate(last_false, axis=0, out=last_false)
    return (index - last_false).max(axis=0)


def _group_consecutive(indices: np.ndarray) -> list[tuple[int, int]]:
    """Group a sorted array of integers into inclusive ``(start, end)`` bands."""
    if indices.size == 0:
        return []

    bands: list[tuple[int, int]] = []
    start = previous = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value == previous + 1:
            previous = value
        else:
            bands.append((start, previous))
            start = previous = value
    bands.append((start, previous))
    return bands


def _candidate_splits(
    inner: np.ndarray,
    *,
    axis: int,
    min_ratio: float,
    max_thickness: int,
) -> list[tuple[int, float]]:
    """Return ``(position, score)`` candidates for frame lines inside *inner*.

    ``axis=1`` looks for horizontal lines (rows that span the inner width) and
    yields the row to split at.  ``axis=0`` looks for vertical lines.
    """
    if inner.size == 0:
        return []

    span = inner.shape[axis]
    if span < 4:
        return []

    runs = _longest_run(inner, axis)
    coverage = runs / float(span)

    # ``runs`` holds one value per candidate line (rows for axis=1, columns
    # for axis=0), so the edge test must use the number of lines -- not the
    # run length (``span``), which belongs to the other dimension.
    line_count = runs.shape[0]

    candidates = np.where(coverage >= min_ratio)[0]
    results: list[tuple[int, float]] = []

    for start, end in _group_consecutive(candidates):
        thickness = end - start + 1
        if thickness > max_thickness:
            continue
        # A frame line never sits exactly on the sampled edge; such a band is
        # almost always a leftover of the box's own border.
        if start <= 0 or end >= line_count - 1:
            continue
        score = float(coverage[start : end + 1].max())
        results.append(((start + end) // 2, score))

    return results


def _inner_region(
    ink: np.ndarray,
    box: np.ndarray,
    settings: SplitSettings,
) -> tuple[np.ndarray, int, int] | None:
    """Return the margin-trimmed interior of *box* as ``(inner, x1, y1)``.

    The margin only has to be large enough to drop the box's own border stroke.
    Keeping it small matters for panels that sit against a page edge: a thin
    strip panel merged with a taller neighbour puts their shared separator
    close to the merged box's edge, and a wide trim would hide it.
    """
    x1, y1, x2, y2 = (float(value) for value in box)
    width = x2 - x1
    height = y2 - y1
    if width < 8 or height < 8:
        return None

    inner_x1 = int(round(x1 + width * settings.margin))
    inner_y1 = int(round(y1 + height * settings.margin))
    inner_x2 = int(round(x2 - width * settings.margin))
    inner_y2 = int(round(y2 - height * settings.margin))
    if inner_x2 - inner_x1 < 4 or inner_y2 - inner_y1 < 4:
        return None

    inner = ink[inner_y1:inner_y2, inner_x1:inner_x2]
    if inner.size == 0:
        return None

    return inner, inner_x1, inner_y1


def _max_internal_line_ratio(
    ink: np.ndarray,
    box: np.ndarray,
    settings: SplitSettings,
) -> float:
    """Best coverage of any thin internal line in *box* (0.0 when none).

    A value between ``ai_min_line_ratio`` and ``min_line_ratio`` is the
    signature of a frame line that is partially hidden -- typically by artwork
    breaking out of a panel -- and is used to trigger the AI fallback.
    """
    region = _inner_region(ink, box, settings)
    if region is None:
        return 0.0

    inner, _, _ = region
    # A floor below the trigger threshold lets the row bands form around the
    # partially hidden line; the returned score is the true peak coverage.
    floor = max(0.05, settings.ai_min_line_ratio * 0.5)
    best = 0.0
    for axis in (1, 0):
        for _, score in _candidate_splits(
            inner,
            axis=axis,
            min_ratio=floor,
            max_thickness=settings.max_line_thickness,
        ):
            best = max(best, score)
    return best


def _connects_parent_edges(
    ink: np.ndarray,
    box: np.ndarray,
    *,
    axis: int,
    position: float,
    max_thickness: int,
    edge_gap_ratio: float,
) -> bool:
    """Return whether a candidate separator reaches both parent-box edges.

    A long horizontal or vertical side of a speech balloon can satisfy the
    line-coverage test inside the margin-trimmed box.  A real separator also
    joins the two *outside* frame edges of the parent panel.  Checking that
    connection on the original ink mask removes balloon/text-box false
    positives without requiring another detector pass.
    """
    x1, y1, x2, y2 = (int(round(float(value))) for value in box)
    x1 = max(0, min(x1, ink.shape[1] - 1))
    x2 = max(x1 + 1, min(x2, ink.shape[1]))
    y1 = max(0, min(y1, ink.shape[0] - 1))
    y2 = max(y1 + 1, min(y2, ink.shape[0]))

    # A candidate band may be a few pixels thick.  Inspect its whole width so
    # a rounded/anti-aliased frame stroke is not rejected merely because its
    # centre landed between two dark rows or columns.
    half_band = max(1, max_thickness // 2)

    if axis == 1:
        coordinate = int(round(position))
        start = max(y1, coordinate - half_band)
        end = min(y2, coordinate + half_band + 1)
        lines = ink[start:end, x1:x2]
        span = x2 - x1
    else:
        coordinate = int(round(position))
        start = max(x1, coordinate - half_band)
        end = min(x2, coordinate + half_band + 1)
        lines = ink[y1:y2, start:end].T
        span = y2 - y1

    allowed_gap = max(1, int(round(span * edge_gap_ratio)))
    for line in lines:
        dark = np.flatnonzero(line)
        if dark.size == 0:
            continue

        # Do not just compare the first and last dark pixel: the two vertical
        # sides of a speech box would then look like a continuous separator.
        # A single run must itself reach both parent edges.
        for run_start, run_end in _group_consecutive(dark):
            if run_start <= allowed_gap and run_end >= span - 1 - allowed_gap:
                return True

    return False


def _frame_line_split(
    ink: np.ndarray,
    box: np.ndarray,
    settings: SplitSettings,
    page_area: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Try to split *box* at an internal frame line, or return ``None``."""
    x1, y1, x2, y2 = (float(value) for value in box)
    region = _inner_region(ink, box, settings)
    if region is None:
        return None

    inner, inner_x1, inner_y1 = region
    min_sub_area = page_area * settings.min_area_ratio

    best: tuple[float, tuple[np.ndarray, np.ndarray]] | None = None

    def consider(
        first: np.ndarray,
        second: np.ndarray,
        score: float,
    ) -> None:
        nonlocal best
        first_area = box_area(first)
        second_area = box_area(second)
        if first_area < min_sub_area or second_area < min_sub_area:
            return
        # Prefer a balanced split when two candidates score equally well.
        balance = 1.0 - abs(first_area - second_area) / (first_area + second_area)
        rank = score + balance * 0.01
        if best is None or rank > best[0]:
            best = (rank, (first, second))

    # Horizontal shared border -> split into top / bottom.
    for position, score in _candidate_splits(
        inner,
        axis=1,
        min_ratio=settings.min_line_ratio,
        max_thickness=settings.max_line_thickness,
    ):
        split_y = float(inner_y1 + position)
        if not _connects_parent_edges(
            ink,
            box,
            axis=1,
            position=split_y,
            max_thickness=settings.max_line_thickness,
            edge_gap_ratio=settings.edge_gap_ratio,
        ):
            continue
        top = np.array([x1, y1, x2, split_y], dtype=np.float32)
        bottom = np.array([x1, split_y, x2, y2], dtype=np.float32)
        consider(top, bottom, score)

    # Vertical shared border -> split into left / right.
    for position, score in _candidate_splits(
        inner,
        axis=0,
        min_ratio=settings.min_line_ratio,
        max_thickness=settings.max_line_thickness,
    ):
        split_x = float(inner_x1 + position)
        if not _connects_parent_edges(
            ink,
            box,
            axis=0,
            position=split_x,
            max_thickness=settings.max_line_thickness,
            edge_gap_ratio=settings.edge_gap_ratio,
        ):
            continue
        left = np.array([x1, y1, split_x, y2], dtype=np.float32)
        right = np.array([split_x, y1, x2, y2], dtype=np.float32)
        consider(left, right, score)

    return best[1] if best is not None else None


# ---------------------------------------------------------------------------
# AI fallback
# ---------------------------------------------------------------------------


def _ai_split(
    context: _SplitContext,
    box: np.ndarray,
) -> list[Detection] | None:
    """Re-run the detector on a zoomed crop of *box* and use its frames.

    Returns a list of separate detections when the zoomed pass finds at least
    two plausible frames, otherwise ``None``.
    """
    settings = context.settings

    x1, y1, x2, y2 = (float(value) for value in box)
    width = x2 - x1
    height = y2 - y1
    if width < 8 or height < 8:
        return None

    # Give the model a little context around the box.
    pad_x = width * 0.02
    pad_y = height * 0.02
    px1 = max(0.0, x1 - pad_x)
    py1 = max(0.0, y1 - pad_y)
    px2 = min(float(context.width), x2 + pad_x)
    py2 = min(float(context.height), y2 + pad_y)

    crop = context.image.crop(
        (int(round(px1)), int(round(py1)), int(round(px2)), int(round(py2)))
    )
    crop_w, crop_h = crop.size
    if crop_w < 8 or crop_h < 8:
        return None

    factor = min(
        settings.ai_max_upscale,
        max(1.0, settings.ai_target_size / max(crop_w, crop_h)),
    )
    if factor > 1.01:
        crop = crop.resize(
            (max(1, int(round(crop_w * factor))), max(1, int(round(crop_h * factor)))),
            Image.Resampling.BILINEAR,
        )

    panels, _, _ = predict(
        context.session,
        crop,
        image_size=context.image_size,
        panel_confidence=max(0.10, context.panel_confidence - 0.10),
        text_confidence=context.text_confidence,
        body_confidence=context.body_confidence,
    )
    if len(panels) < settings.ai_min_frames:
        return None

    offset = np.array([px1, py1, px1, py1], dtype=np.float32)
    mapped = [
        Detection(
            np.asarray(panel.box, dtype=np.float32) / float(factor) + offset,
            float(panel.score),
            int(panel.class_id),
        )
        for panel in panels
    ]

    # Drop nested / duplicate frames so the remaining ones are genuinely apart.
    mapped, _ = deduplicate(
        mapped,
        iou_threshold=0.70,
        containment_threshold=0.85,
    )

    crop_area = (px2 - px1) * (py2 - py1)
    parent_area = width * height
    min_frame_area = crop_area * settings.ai_min_frame_area_ratio

    kept: list[Detection] = []
    for detection in mapped:
        clipped = clip_box(detection.box, context.width, context.height)
        area = box_area(clipped)
        # A frame that nearly covers the parent is not a split, just the merge.
        if area < min_frame_area or area >= parent_area * 0.90:
            continue
        kept.append(
            Detection(np.asarray(clipped, dtype=np.float32), detection.score, detection.class_id)
        )

    if len(kept) < settings.ai_min_frames:
        return None

    return kept[: settings.ai_max_frames]


# ---------------------------------------------------------------------------
# Recursive driver
# ---------------------------------------------------------------------------


def _split_detection(
    detection: Detection,
    context: _SplitContext,
    depth: int,
) -> list[Detection]:
    """Return *detection* split into as many panels as can be justified."""
    settings = context.settings

    if depth >= settings.max_depth:
        return [detection]

    if settings.enable_frame_split:
        pair = _frame_line_split(
            context.ink,
            detection.box,
            settings,
            context.page_area,
        )
        if pair is not None:
            first = Detection(pair[0], detection.score, detection.class_id)
            second = Detection(pair[1], detection.score, detection.class_id)
            return _split_detection(first, context, depth + 1) + _split_detection(
                second, context, depth + 1
            )

    if settings.enable_ai_fallback:
        area_ratio = box_area(detection.box) / max(1, context.page_area)
        if area_ratio >= settings.ai_min_area_ratio:
            # Only re-run the detector when there is evidence of a merge: a
            # partially visible frame line that the full-threshold test could
            # not confirm (e.g. hidden by artwork breaking out of a panel).
            evidence = _max_internal_line_ratio(
                context.ink,
                detection.box,
                settings,
            )
            if evidence >= settings.ai_min_line_ratio:
                children = _ai_split(context, detection.box)
                if children is not None:
                    return children

    return [detection]


def split_merged_panels(
    image: Image.Image,
    panels: list[Detection],
    *,
    session: "ort.InferenceSession",
    image_size: int,
    panel_confidence: float,
    text_confidence: float,
    body_confidence: float,
    settings: SplitSettings,
) -> tuple[list[Detection], int]:
    """Split merged panel boxes.

    Returns the expanded panel list and the number of extra panels created.
    The original detections are never modified.
    """
    if settings is None or not (
        settings.enable_frame_split or settings.enable_ai_fallback
    ):
        return list(panels), 0

    width, height = image.size
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    ink = gray < settings.dark_threshold

    context = _SplitContext(
        ink=ink,
        image=image,
        session=session,
        image_size=image_size,
        panel_confidence=panel_confidence,
        text_confidence=text_confidence,
        body_confidence=body_confidence,
        settings=settings,
        page_area=max(1, width * height),
        width=width,
        height=height,
    )

    result: list[Detection] = []
    extra = 0

    for detection in panels:
        children = _split_detection(detection, context, depth=0)
        if len(children) > 1:
            extra += len(children) - 1
        result.extend(children)

    return result, extra
