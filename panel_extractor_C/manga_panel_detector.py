#!/usr/bin/env python3
"""Classical-computer-vision manga panel detector.

This module is based conceptually on pakkio/comics-panel, but uses a
separator-first strategy for manga:

    page
      -> preprocessing
      -> horizontal / vertical separator detection
      -> recursive layout splitting
      -> validation
      -> classical fallbacks (contours, watershed, projection grid)
      -> RTL/LTR ordering
      -> crop export

No AI, deep learning, model weights, or OCR are used.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger("manga_panel_detector")
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

ImageInput = Union[str, Path, np.ndarray]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Basic image helpers
# ---------------------------------------------------------------------------


def _load_image(image: ImageInput) -> np.ndarray:
    if isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected a BGR image with shape (H, W, 3).")
        return image.copy()

    path = Path(image)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img


def estimate_page_background(gray: np.ndarray, border_px: int = 6) -> str:
    """Estimate whether the outer page edge is mostly white or black."""
    h, w = gray.shape
    b = max(1, min(border_px, h // 4, w // 4))
    samples = np.concatenate(
        [
            gray[:b, :].ravel(),
            gray[-b:, :].ravel(),
            gray[:, :b].ravel(),
            gray[:, -b:].ravel(),
        ]
    )
    return "white" if float(np.median(samples)) >= 127.0 else "black"


def add_bleed_border(
    img_bgr: np.ndarray,
    thickness_ratio: float = 0.015,
) -> Tuple[np.ndarray, int, Optional[List[int]]]:
    """Pad the page and add an artificial frame for edge-touching panels."""
    h, w = img_bgr.shape[:2]
    pad = max(2, int(min(h, w) * thickness_ratio))
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    bg = estimate_page_background(gray)
    fill = 255 if bg == "white" else 0
    padded = cv2.copyMakeBorder(
        img_bgr,
        pad,
        pad,
        pad,
        pad,
        cv2.BORDER_CONSTANT,
        value=(fill, fill, fill),
    )

    ink = 0 if bg == "white" else 255
    thickness = max(1, pad // 2)
    cv2.rectangle(
        padded,
        (pad, pad),
        (padded.shape[1] - pad - 1, padded.shape[0] - pad - 1),
        (ink, ink, ink),
        thickness,
    )

    frame_bbox = [pad, pad, padded.shape[1] - 2 * pad, padded.shape[0] - 2 * pad]
    return padded, pad, frame_bbox


def denoise_screentone(
    gray: np.ndarray,
    kernel_ratio: float = 0.003,
    strength: int = 1,
) -> np.ndarray:
    """Reduce screentone dots without aggressively blurring panel borders."""
    h, w = gray.shape
    k = max(3, int(min(h, w) * kernel_ratio))
    if k % 2 == 0:
        k += 1
    result = gray
    for _ in range(max(1, strength)):
        result = cv2.medianBlur(result, k)
    return result


def _safe_crop_bbox(bbox: List[int], width: int, height: int) -> List[int]:
    x, y, w, h = [int(v) for v in bbox]
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    x2 = max(x + 1, min(width, x + max(1, w)))
    y2 = max(y + 1, min(height, y + max(1, h)))
    return [x, y, x2 - x, y2 - y]


def _unpad_panels(
    panels: List[Dict[str, Any]], pad: int, original_w: int, original_h: int
) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for panel in panels:
        p = dict(panel)
        x, y, w, h = p["bbox"]
        p["bbox"] = _safe_crop_bbox(
            [x - pad, y - pad, w, h], original_w, original_h
        )
        if "rotated_box" in p:
            p["rotated_box"] = [
                [float(px) - pad, float(py) - pad]
                for px, py in p["rotated_box"]
            ]
        result.append(p)
    return result


# ---------------------------------------------------------------------------
# Separator detection
# ---------------------------------------------------------------------------


def _build_separator_masks(
    gray: np.ndarray,
    black_threshold: int,
    line_ratio: float,
    canny_low: int,
    canny_high: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return black ink, long horizontal/vertical lines and Canny edges."""
    h, w = gray.shape
    black = cv2.threshold(
        gray, black_threshold, 255, cv2.THRESH_BINARY_INV
    )[1]

    h_len = max(5, int(w * line_ratio))
    v_len = max(5, int(h * line_ratio))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))

    horizontal = cv2.morphologyEx(black, cv2.MORPH_OPEN, h_kernel)
    vertical = cv2.morphologyEx(black, cv2.MORPH_OPEN, v_kernel)
    edges = cv2.Canny(gray, canny_low, canny_high)
    return black, horizontal, vertical, edges


def _hough_lines(
    edges: np.ndarray,
    min_line_length: int,
    max_line_gap: int,
    angle_tolerance_deg: float,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    """Return near-horizontal and near-vertical Hough line segments."""
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=max(15, min_line_length // 3),
        minLineLength=max(8, min_line_length),
        maxLineGap=max(1, max_line_gap),
    )

    horizontal: List[Dict[str, float]] = []
    vertical: List[Dict[str, float]] = []
    if lines is None:
        return horizontal, vertical

    # OpenCV normally returns (N, 1, 4), but reshape each item so the code
    # stays robust across OpenCV builds.
    for raw in np.asarray(lines).reshape(-1, 4):
        x1, y1, x2, y2 = [int(v) for v in raw]
        dx = x2 - x1
        dy = y2 - y1
        length = float(np.hypot(dx, dy))
        if length < min_line_length:
            continue

        angle = abs(float(np.degrees(np.arctan2(dy, dx)))) % 180.0
        if min(angle, 180.0 - angle) <= angle_tolerance_deg:
            horizontal.append(
                {
                    "position": float((y1 + y2) * 0.5),
                    "start": float(min(x1, x2)),
                    "end": float(max(x1, x2)),
                    "length": length,
                }
            )
        elif abs(angle - 90.0) <= angle_tolerance_deg:
            vertical.append(
                {
                    "position": float((x1 + x2) * 0.5),
                    "start": float(min(y1, y2)),
                    "end": float(max(y1, y2)),
                    "length": length,
                }
            )

    return horizontal, vertical


def _connected_projection_candidates(
    mask: np.ndarray,
    orientation: str,
    min_coverage: float = 0.35,
    min_run: int = 1,
) -> List[Tuple[int, int, int]]:
    """Find sustained rows/columns with enough long-line coverage."""
    h, w = mask.shape
    if orientation == "horizontal":
        projection = np.mean(mask > 0, axis=1)
        scale = w
    else:
        projection = np.mean(mask > 0, axis=0)
        scale = h

    active = projection >= min_coverage
    candidates: List[Tuple[int, int, int]] = []
    i = 0
    n = len(active)
    while i < n:
        if not active[i]:
            i += 1
            continue
        start = i
        while i < n and active[i]:
            i += 1
        end = i - 1
        if end - start + 1 >= min_run:
            center = int(round((start + end) * 0.5))
            candidates.append((center, 0, scale - 1))
    return candidates


def _merge_position_groups(
    positions: List[float], distance: int
) -> List[int]:
    if not positions:
        return []
    positions = sorted(float(v) for v in positions)
    groups: List[List[float]] = [[positions[0]]]
    for p in positions[1:]:
        if p - groups[-1][-1] <= distance:
            groups[-1].append(p)
        else:
            groups.append([p])
    return [int(round(float(np.mean(g)))) for g in groups]


def _image_strip_stats(
    gray: np.ndarray,
    orientation: str,
    position: int,
    start: int,
    end: int,
    half_width: int,
) -> Tuple[float, float, float]:
    """Return line ink coverage, gutter whiteness and side ink contrast."""
    h, w = gray.shape
    if orientation == "horizontal":
        position = max(1, min(h - 2, position))
        x1 = max(0, min(w - 1, start))
        x2 = max(x1 + 1, min(w, end + 1))
        band = gray[max(0, position - 1):min(h, position + 2), x1:x2]
        above = gray[max(0, position - half_width):position, x1:x2]
        below = gray[position + 1:min(h, position + 1 + half_width), x1:x2]
    else:
        position = max(1, min(w - 2, position))
        y1 = max(0, min(h - 1, start))
        y2 = max(y1 + 1, min(h, end + 1))
        band = gray[y1:y2, max(0, position - 1):min(w, position + 2)]
        above = gray[y1:y2, max(0, position - half_width):position]
        below = gray[y1:y2, position + 1:min(w, position + 1 + half_width)]

    if band.size == 0 or above.size == 0 or below.size == 0:
        return 0.0, 0.0, 0.0

    line_coverage = float(np.mean(band < 90))
    white_left = float(np.mean(above > 235))
    white_right = float(np.mean(below > 235))
    gutter_whiteness = min(1.0, 0.5 * (white_left + white_right))

    ink_left = float(np.mean(above < 100))
    ink_right = float(np.mean(below < 100))
    side_contrast = min(1.0, abs(ink_left - ink_right) * 3.0)
    return line_coverage, gutter_whiteness, side_contrast


def _score_separator(
    gray: np.ndarray,
    orientation: str,
    position: int,
    start: int,
    end: int,
    min_span_ratio: float,
) -> Tuple[float, Dict[str, float]]:
    h, w = gray.shape
    dimension = w if orientation == "horizontal" else h
    span = max(1, end - start + 1)
    span_ratio = min(1.0, span / float(max(1, dimension)))
    half_width = max(3, int(min(h, w) * 0.012))

    line_cov, gutter_white, side_contrast = _image_strip_stats(
        gray, orientation, position, start, end, half_width
    )

    # Strong continuous lines are the primary signal. Gutter whiteness helps
    # distinguish panel separators from arbitrary character/object lines.
    continuity = max(0.0, min(1.0, (span_ratio - min_span_ratio) / max(1e-6, 1.0 - min_span_ratio)))
    score = (
        0.55 * line_cov
        + 0.20 * gutter_white
        + 0.10 * side_contrast
        + 0.15 * continuity
    )
    return float(score), {
        "line_coverage": line_cov,
        "gutter_whiteness": gutter_white,
        "side_contrast": side_contrast,
        "span_ratio": span_ratio,
    }


def detect_separator_candidates(
    gray: np.ndarray,
    *,
    black_threshold: int = 55,
    canny_low: int = 40,
    canny_high: int = 130,
    line_ratio: float = 0.05,
    min_span_ratio: float = 0.30,
    min_score: float = 0.42,
    angle_tolerance_deg: float = 4.0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, np.ndarray]]:
    """Detect scored horizontal and vertical panel separator candidates."""
    h, w = gray.shape
    black, horizontal, vertical, edges = _build_separator_masks(
        gray,
        black_threshold,
        line_ratio,
        canny_low,
        canny_high,
    )

    min_line_length = max(12, int(min(h, w) * min_span_ratio))
    max_line_gap = max(4, int(min(h, w) * 0.025))
    hough_h, hough_v = _hough_lines(
        edges,
        min_line_length,
        max_line_gap,
        angle_tolerance_deg,
    )

    raw_h: List[Tuple[int, int, int]] = [
        (int(round(c["position"])), int(c["start"]), int(c["end"]))
        for c in hough_h
    ] + _connected_projection_candidates(
        horizontal, "horizontal", min_coverage=0.28
    )
    raw_v: List[Tuple[int, int, int]] = [
        (int(round(c["position"])), int(c["start"]), int(c["end"]))
        for c in hough_v
    ] + _connected_projection_candidates(
        vertical, "vertical", min_coverage=0.28
    )

    def build(
        raw: List[Tuple[int, int, int]], orientation: str, merge_distance: int
    ) -> List[Dict[str, Any]]:
        if not raw:
            return []
        centers = _merge_position_groups([r[0] for r in raw], merge_distance)
        candidates: List[Dict[str, Any]] = []
        for center in centers:
            group = [r for r in raw if abs(r[0] - center) <= merge_distance]
            start = min(r[1] for r in group)
            end = max(r[2] for r in group)
            score, features = _score_separator(
                gray, orientation, center, start, end, min_span_ratio
            )
            candidates.append(
                {
                    "orientation": orientation,
                    "position": int(center),
                    "span_start": int(start),
                    "span_end": int(end),
                    "score": score,
                    "features": features,
                    "support": len(group),
                }
            )

        candidates = [c for c in candidates if c["score"] >= min_score]
        candidates.sort(key=lambda c: c["score"], reverse=True)
        return candidates

    horizontal_candidates = build(raw_h, "horizontal", max(3, h // 120))
    vertical_candidates = build(raw_v, "vertical", max(3, w // 120))

    debug = {
        "separator_black": black,
        "separator_horizontal": horizontal,
        "separator_vertical": vertical,
        "separator_edges": edges,
    }
    return horizontal_candidates, vertical_candidates, debug


# ---------------------------------------------------------------------------
# Recursive layout splitting
# ---------------------------------------------------------------------------


def _candidate_covers_region(
    candidate: Dict[str, Any], bbox: List[int], orientation: str, required_coverage: float
) -> bool:
    x, y, w, h = bbox
    if orientation == "horizontal":
        position = candidate["position"]
        if not (y < position < y + h):
            return False
        overlap = max(
            0,
            min(x + w, candidate["span_end"]) - max(x, candidate["span_start"]),
        )
        return overlap / float(max(1, w)) >= required_coverage

    position = candidate["position"]
    if not (x < position < x + w):
        return False
    overlap = max(
        0,
        min(y + h, candidate["span_end"]) - max(y, candidate["span_start"]),
    )
    return overlap / float(max(1, h)) >= required_coverage


def _child_boxes(
    bbox: List[int], candidate: Dict[str, Any], orientation: str, gap: int
) -> Optional[Tuple[List[int], List[int]]]:
    x, y, w, h = bbox
    cut = int(candidate["position"])

    if orientation == "horizontal":
        a = [x, y, w, cut - gap - y]
        b_y = cut + gap
        b = [x, b_y, w, y + h - b_y]
    else:
        a = [x, y, cut - gap - x, h]
        b_x = cut + gap
        b = [b_x, y, x + w - b_x, h]

    if a[2] <= 0 or a[3] <= 0 or b[2] <= 0 or b[3] <= 0:
        return None
    return a, b


def _region_content_score(gray: np.ndarray, bbox: List[int]) -> float:
    """Return a simple amount-of-structure score for a proposed region."""
    x, y, w, h = _safe_crop_bbox(bbox, gray.shape[1], gray.shape[0])
    roi = gray[y:y + h, x:x + w]
    if roi.size == 0:
        return 0.0
    # Both ink and moderate gray structure count. Pure white/black blanks are
    # unlikely to be useful panel leaves.
    ink = float(np.mean(roi < 210))
    variance = float(np.std(roi) / 128.0)
    return min(1.0, 0.65 * ink + 0.35 * min(1.0, variance))


def recursive_separator_split(
    gray: np.ndarray,
    horizontal_candidates: List[Dict[str, Any]],
    vertical_candidates: List[Dict[str, Any]],
    *,
    min_panel_area_ratio: float = 0.008,
    min_region_width_ratio: float = 0.08,
    min_region_height_ratio: float = 0.08,
    max_depth: int = 8,
    split_gap_ratio: float = 0.004,
    min_separator_coverage: float = 0.60,
    min_split_score: float = 0.44,
    reading_order: str = "rtl",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Recursively split the page using the strongest valid separator."""
    H, W = gray.shape
    root = [0, 0, W, H]
    min_w = max(24, int(W * min_region_width_ratio))
    min_h = max(24, int(H * min_region_height_ratio))
    min_area = min_panel_area_ratio * W * H
    gap = max(1, int(min(H, W) * split_gap_ratio))

    leaves: List[Dict[str, Any]] = []
    split_records: List[Dict[str, Any]] = []

    def recurse(bbox: List[int], depth: int, path: List[int]) -> None:
        x, y, w, h = bbox
        area_ratio = (w * h) / float(W * H)

        if depth >= max_depth or w < min_w * 2 or h < min_h * 2:
            leaves.append(
                {
                    "bbox": bbox.copy(),
                    "area": int(w * h),
                    "area_ratio": float(area_ratio),
                    "method": "separator",
                    "layout_path": path.copy(),
                    "score": 1.0,
                }
            )
            return

        options: List[Tuple[float, str, Dict[str, Any], List[int], List[int]]] = []

        for candidate in vertical_candidates:
            if not _candidate_covers_region(
                candidate, bbox, "vertical", min_separator_coverage
            ):
                continue
            children = _child_boxes(bbox, candidate, "vertical", gap)
            if children is None:
                continue
            a, b = children
            if min(a[2], b[2]) < min_w or min(a[3], b[3]) < min_h:
                continue
            if min(a[2] * a[3], b[2] * b[3]) < min_area:
                continue

            balance = min(a[2] * a[3], b[2] * b[3]) / float(w * h)
            score = 0.78 * float(candidate["score"]) + 0.22 * min(1.0, balance * 2.0)
            options.append((score, "vertical", candidate, a, b))

        for candidate in horizontal_candidates:
            if not _candidate_covers_region(
                candidate, bbox, "horizontal", min_separator_coverage
            ):
                continue
            children = _child_boxes(bbox, candidate, "horizontal", gap)
            if children is None:
                continue
            a, b = children
            if min(a[2], b[2]) < min_w or min(a[3], b[3]) < min_h:
                continue
            if min(a[2] * a[3], b[2] * b[3]) < min_area:
                continue

            balance = min(a[2] * a[3], b[2] * b[3]) / float(w * h)
            score = 0.78 * float(candidate["score"]) + 0.22 * min(1.0, balance * 2.0)
            options.append((score, "horizontal", candidate, a, b))

        options.sort(key=lambda item: item[0], reverse=True)
        chosen = next((item for item in options if item[0] >= min_split_score), None)

        if chosen is None:
            leaves.append(
                {
                    "bbox": bbox.copy(),
                    "area": int(w * h),
                    "area_ratio": float(area_ratio),
                    "method": "separator",
                    "layout_path": path.copy(),
                    "score": 1.0,
                }
            )
            return

        score, orientation, candidate, first, second = chosen
        split_records.append(
            {
                "bbox": bbox.copy(),
                "orientation": orientation,
                "position": int(candidate["position"]),
                "score": float(score),
                "candidate_score": float(candidate["score"]),
                "layout_path": path.copy(),
            }
        )

        # A horizontal split is always top -> bottom. A vertical split follows
        # the requested reading direction so layout_path already reflects the
        # intended manga reading sequence.
        if orientation == "vertical" and reading_order == "rtl":
            recurse(second, depth + 1, path + [0])
            recurse(first, depth + 1, path + [1])
        else:
            recurse(first, depth + 1, path + [0])
            recurse(second, depth + 1, path + [1])

    recurse(root, 0, [])
    return leaves, split_records


# ---------------------------------------------------------------------------
# Fallback detectors based on comics-panel
# ---------------------------------------------------------------------------


def _normalise_angle(angle: float) -> float:
    a = angle % 90.0
    return a - 90.0 if a > 45.0 else a


def _build_line_mask(
    gray: np.ndarray,
    black_threshold: int,
    line_thickness_ratio: float,
    morph_iterations: int,
    edge_method: str,
    white_threshold: int,
    debug: Dict[str, np.ndarray],
) -> np.ndarray:
    h, w = gray.shape
    kernel_size = max(1, int(min(h, w) * line_thickness_ratio))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_size, kernel_size)
    )

    black_mask = cv2.threshold(
        gray, black_threshold, 255, cv2.THRESH_BINARY_INV
    )[1]
    black_lines = cv2.morphologyEx(
        black_mask, cv2.MORPH_CLOSE, kernel, iterations=max(1, morph_iterations)
    )
    debug["fallback_black_lines"] = black_lines.copy()

    if edge_method == "black":
        return black_lines

    if edge_method == "white":
        white_mask = cv2.threshold(
            gray, white_threshold, 255, cv2.THRESH_BINARY
        )[1]
        white_lines = cv2.morphologyEx(
            white_mask, cv2.MORPH_OPEN, kernel, iterations=max(1, morph_iterations)
        )
        debug["fallback_white_lines"] = white_lines.copy()
        return white_lines

    if edge_method == "canny":
        edges = cv2.Canny(gray, 30, 100)
        debug["fallback_canny"] = edges.copy()
        return cv2.dilate(edges, kernel, iterations=max(1, morph_iterations))

    white_mask = cv2.threshold(
        gray, white_threshold, 255, cv2.THRESH_BINARY
    )[1]
    white_lines = cv2.morphologyEx(
        white_mask, cv2.MORPH_OPEN, kernel, iterations=max(1, morph_iterations)
    )
    debug["fallback_white_lines"] = white_lines.copy()
    return cv2.bitwise_or(black_lines, white_lines)


def _fallback_contours(
    gray: np.ndarray,
    line_mask: np.ndarray,
    *,
    min_area_ratio: float,
    max_area_ratio: float,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
    allow_rotated: bool,
    rotation_threshold_deg: float,
) -> List[Dict[str, Any]]:
    h, w = gray.shape
    contours, _ = cv2.findContours(
        line_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    min_area = min_area_ratio * w * h
    max_area = max_area_ratio * w * h
    panels: List[Dict[str, Any]] = []

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area or area > max_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 8 or bh < 8:
            continue
        ar = bw / float(bh)
        inv = bh / float(bw)
        if not (
            min_aspect_ratio <= ar <= max_aspect_ratio
            or min_aspect_ratio <= inv <= max_aspect_ratio
        ):
            continue

        rect_area = bw * bh
        fill = area / float(rect_area) if rect_area else 0.0
        if fill < 0.55:
            continue

        panel: Dict[str, Any] = {
            "bbox": [int(x), int(y), int(bw), int(bh)],
            "area": int(area),
            "area_ratio": float(area / (w * h)),
            "aspect_ratio": float(ar),
            "method": "contour",
            "score": float(min(1.0, fill)),
        }
        if allow_rotated:
            rect = cv2.minAreaRect(cnt)
            angle = _normalise_angle(float(rect[2]))
            if abs(angle) >= rotation_threshold_deg:
                panel["rotated_box"] = cv2.boxPoints(rect).tolist()
                panel["rotation_deg"] = float(angle)
        panels.append(panel)
    return panels


def _fallback_watershed(
    img_bgr: np.ndarray,
    gray: np.ndarray,
    line_mask: Optional[np.ndarray],
    *,
    min_area_ratio: float,
    max_area_ratio: float,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
) -> List[Dict[str, Any]]:
    h, w = gray.shape
    if line_mask is not None:
        panel_mask = cv2.bitwise_not(line_mask)
    else:
        panel_mask = cv2.threshold(
            gray, 127, 255, cv2.THRESH_BINARY
        )[1]

    dist = cv2.distanceTransform(panel_mask, cv2.DIST_L2, 5)
    if float(np.max(dist)) <= 0:
        return []
    normalized = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    seeds = cv2.threshold(normalized, 100, 255, cv2.THRESH_BINARY)[1]
    _, labels = cv2.connectedComponents(seeds)
    labels = labels.astype(np.int32) + 1
    if line_mask is not None:
        labels[line_mask > 0] = 0

    markers = cv2.watershed(img_bgr, labels)
    min_area = min_area_ratio * w * h
    max_area = max_area_ratio * w * h
    panels: List[Dict[str, Any]] = []

    max_marker = int(np.max(markers))
    for marker_id in range(2, max_marker + 1):
        mask = np.uint8(markers == marker_id) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(cnt))
        if area < min_area or area > max_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        ar = bw / float(bh) if bh else 0.0
        inv = bh / float(bw) if bw else 0.0
        if not (
            min_aspect_ratio <= ar <= max_aspect_ratio
            or min_aspect_ratio <= inv <= max_aspect_ratio
        ):
            continue
        fill = area / float(max(1, bw * bh))
        if fill < 0.55:
            continue
        panels.append(
            {
                "bbox": [int(x), int(y), int(bw), int(bh)],
                "area": int(area),
                "area_ratio": float(area / (w * h)),
                "aspect_ratio": float(ar),
                "method": "watershed",
                "score": float(min(1.0, fill)),
                "marker_id": int(marker_id),
            }
        )
    return panels


def detect_panels_by_projection_manga(
    gray: np.ndarray,
    *,
    min_area_ratio: float = 0.008,
    max_area_ratio: float = 0.65,
    black_threshold: int = 50,
) -> List[Dict[str, Any]]:
    """Regular-layout fallback using low-ink projection gaps."""
    h, w = gray.shape
    black = cv2.threshold(
        gray, black_threshold, 255, cv2.THRESH_BINARY_INV
    )[1]

    ink_y = np.mean(black > 0, axis=1)
    ink_x = np.mean(black > 0, axis=0)

    # A gutter is more likely to be a sustained low-ink run than a single
    # low-ink row/column. Use a relative threshold to handle scanned pages.
    y_threshold = max(0.01, float(np.percentile(ink_y, 25)) * 0.8)
    x_threshold = max(0.01, float(np.percentile(ink_x, 25)) * 0.8)

    def borders(profile: np.ndarray, threshold: float, length: int) -> List[int]:
        result = [0]
        i = 0
        while i < len(profile):
            if profile[i] <= threshold:
                start = i
                while i < len(profile) and profile[i] <= threshold:
                    i += 1
                end = i - 1
                if end - start + 1 >= max(2, int(length * 0.005)):
                    result.append((start + end) // 2)
            else:
                i += 1
        result.append(length - 1)
        return sorted(set(result))

    h_borders = borders(ink_y, y_threshold, h)
    v_borders = borders(ink_x, x_threshold, w)
    if len(h_borders) < 2 or len(v_borders) < 2:
        return []

    panels: List[Dict[str, Any]] = []
    min_area = min_area_ratio * w * h
    max_area = max_area_ratio * w * h
    for yi in range(len(h_borders) - 1):
        for xi in range(len(v_borders) - 1):
            x1 = h_borders[yi] if False else v_borders[xi]
            y1 = h_borders[yi]
            x2 = v_borders[xi + 1]
            y2 = h_borders[yi + 1]
            x1 += 1
            y1 += 1
            x2 -= 1
            y2 -= 1
            bw = x2 - x1
            bh = y2 - y1
            area = bw * bh
            if bw <= 0 or bh <= 0 or area < min_area or area > max_area:
                continue
            panels.append(
                {
                    "bbox": [int(x1), int(y1), int(bw), int(bh)],
                    "area": int(area),
                    "area_ratio": float(area / (w * h)),
                    "aspect_ratio": float(bw / bh),
                    "method": "grid",
                    "score": 0.35,
                }
            )
    return panels


# ---------------------------------------------------------------------------
# Candidate cleanup / ordering
# ---------------------------------------------------------------------------


def calculate_iou(box1: List[int], box2: List[int]) -> float:
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    ix1 = max(x1, x2)
    iy1 = max(y1, y2)
    ix2 = min(x1 + w1, x2 + w2)
    iy2 = min(y1 + h1, y2 + h2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = w1 * h1 + w2 * h2 - inter
    return inter / float(union) if union > 0 else 0.0


_METHOD_PRIORITY = {"separator": 4, "contour": 3, "watershed": 2, "grid": 1}


def remove_overlapping_panels(
    panels: List[Dict[str, Any]], overlap_threshold: float = 0.50
) -> List[Dict[str, Any]]:
    if not panels:
        return []

    ordered = sorted(
        panels,
        key=lambda p: (
            _METHOD_PRIORITY.get(p.get("method", ""), 0),
            float(p.get("score", 0.0)),
            int(p.get("area", 0)),
        ),
        reverse=True,
    )

    kept: List[Dict[str, Any]] = []
    for candidate in ordered:
        if all(
            calculate_iou(candidate["bbox"], other["bbox"]) <= overlap_threshold
            for other in kept
        ):
            kept.append(candidate)
    return kept


def _sort_reading_rows(
    panels: List[Dict[str, Any]], reading_order: str
) -> List[Dict[str, Any]]:
    if not panels:
        return []

    enriched = []
    for panel in panels:
        x, y, w, h = panel["bbox"]
        enriched.append(
            {
                "panel": panel,
                "x1": x,
                "y1": y,
                "x2": x + w,
                "y2": y + h,
            }
        )

    enriched.sort(key=lambda e: (e["y1"], e["x1"]))
    rows: List[List[Dict[str, Any]]] = []
    for item in enriched:
        placed = False
        center_y = (item["y1"] + item["y2"]) * 0.5
        for row in rows:
            row_centers = [((r["y1"] + r["y2"]) * 0.5) for r in row]
            reference = float(np.mean(row_centers))
            tolerance = max(
                8.0,
                0.35
                * max(
                    item["y2"] - item["y1"],
                    float(np.mean([r["y2"] - r["y1"] for r in row])),
                ),
            )
            if abs(center_y - reference) <= tolerance:
                row.append(item)
                placed = True
                break
        if not placed:
            rows.append([item])

    rows.sort(key=lambda row: min(item["y1"] for item in row))
    reverse = reading_order == "rtl"
    result: List[Dict[str, Any]] = []
    for row in rows:
        row.sort(key=lambda item: item["x1"], reverse=reverse)
        result.extend(item["panel"] for item in row)
    return result


def improve_panel_order_manga(
    panels: List[Dict[str, Any]],
    img_height: int,
    img_width: int,
    reading_order: str = "rtl",
) -> List[Dict[str, Any]]:
    """Assign final reading_index using spatial row ordering."""
    ordered = _sort_reading_rows(panels, reading_order)
    for index, panel in enumerate(ordered, start=1):
        panel["reading_index"] = index
    return ordered


# ---------------------------------------------------------------------------
# Main detection function
# ---------------------------------------------------------------------------


def _draw_separator_debug(
    image: np.ndarray,
    horizontal: List[Dict[str, Any]],
    vertical: List[Dict[str, Any]],
    accepted: List[Dict[str, Any]],
) -> np.ndarray:
    out = image.copy()
    for c in horizontal:
        y = int(c["position"])
        cv2.line(out, (0, y), (out.shape[1] - 1, y), (255, 180, 0), 1)
    for c in vertical:
        x = int(c["position"])
        cv2.line(out, (x, 0), (x, out.shape[0] - 1), (255, 180, 0), 1)
    for split in accepted:
        x, y, w, h = split["bbox"]
        if split["orientation"] == "horizontal":
            p = int(split["position"])
            cv2.line(out, (x, p), (x + w, p), (0, 255, 0), 3)
        else:
            p = int(split["position"])
            cv2.line(out, (p, y), (p, y + h), (0, 255, 0), 3)
    return out


def detect_panels_manga(
    image: ImageInput,
    out_dir: Optional[Path] = None,
    *,
    reading_order: str = "rtl",
    min_panel_area_ratio: float = 0.008,
    max_panel_area_ratio: float = 0.65,
    edge_method: str = "black",
    black_threshold: int = 50,
    white_threshold: int = 235,
    line_thickness_ratio: float = 0.004,
    morph_iterations: int = 2,
    min_aspect_ratio: float = 0.15,
    max_aspect_ratio: float = 6.0,
    denoise: bool = True,
    denoise_kernel_ratio: float = 0.003,
    denoise_strength: int = 1,
    bleed_border: bool = True,
    bleed_border_ratio: float = 0.015,
    allow_rotated: bool = True,
    rotation_threshold_deg: float = 3.0,
    enable_contour_method: bool = True,
    enable_watershed_method: bool = True,
    enable_grid_fallback: bool = True,
    separator_first: bool = True,
    separator_min_score: float = 0.42,
    separator_line_ratio: float = 0.05,
    separator_min_span_ratio: float = 0.30,
    separator_min_region_width_ratio: float = 0.08,
    separator_min_region_height_ratio: float = 0.08,
    separator_max_depth: int = 8,
    separator_min_leaf_count: int = 2,
    debug: bool = False,
    debug_stem: str = "debug",
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, np.ndarray]]:
    """Detect manga panels and return the original image, panels and debug images."""
    out_dir = Path(out_dir) if out_dir is not None else None
    debug_images: Dict[str, np.ndarray] = {}

    original = _load_image(image)
    debug_images["original"] = original.copy()
    orig_h, orig_w = original.shape[:2]

    if bleed_border:
        working, pad, frame_bbox = add_bleed_border(
            original, thickness_ratio=bleed_border_ratio
        )
    else:
        working = original.copy()
        pad = 0
        frame_bbox = None

    gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(
        clipLimit=2.0, tileGridSize=(8, 8)
    ).apply(gray)
    source = (
        denoise_screentone(enhanced, denoise_kernel_ratio, denoise_strength)
        if denoise
        else enhanced
    )
    debug_images["enhanced"] = enhanced.copy()
    debug_images["mask_source"] = source.copy()

    panels: List[Dict[str, Any]] = []

    if separator_first:
        h_candidates, v_candidates, sep_debug = detect_separator_candidates(
            source,
            black_threshold=black_threshold,
            line_ratio=separator_line_ratio,
            min_span_ratio=separator_min_span_ratio,
            min_score=separator_min_score,
        )
        debug_images.update(sep_debug)

        leaves, splits = recursive_separator_split(
            source,
            h_candidates,
            v_candidates,
            min_panel_area_ratio=min_panel_area_ratio,
            min_region_width_ratio=separator_min_region_width_ratio,
            min_region_height_ratio=separator_min_region_height_ratio,
            max_depth=separator_max_depth,
            split_gap_ratio=0.004,
            min_separator_coverage=0.60,
            min_split_score=max(separator_min_score, 0.44),
            reading_order=reading_order,
        )

        if len(leaves) >= separator_min_leaf_count:
            for leaf in leaves:
                leaf["score"] = 0.75
            panels.extend(leaves)

        debug_images["separator_debug"] = _draw_separator_debug(
            working, h_candidates, v_candidates, splits
        )
        logger.info(
            "Separator-first: %d leaves, %d accepted splits, %d horizontal candidates, %d vertical candidates",
            len(leaves),
            len(splits),
            len(h_candidates),
            len(v_candidates),
        )

    # If separator segmentation produced too few regions, use the original
    # repository-style methods to recover difficult pages.
    if len(panels) < separator_min_leaf_count:
        line_mask: Optional[np.ndarray] = None

        if enable_contour_method:
            line_mask = _build_line_mask(
                source,
                black_threshold,
                line_thickness_ratio,
                morph_iterations,
                edge_method,
                white_threshold,
                debug_images,
            )
            contour_panels = _fallback_contours(
                source,
                line_mask,
                min_area_ratio=min_panel_area_ratio,
                max_area_ratio=max_panel_area_ratio,
                min_aspect_ratio=min_aspect_ratio,
                max_aspect_ratio=max_aspect_ratio,
                allow_rotated=allow_rotated,
                rotation_threshold_deg=rotation_threshold_deg,
            )
            panels.extend(contour_panels)
            logger.info("Contour fallback: %d candidates", len(contour_panels))

        if enable_watershed_method and len(panels) < separator_min_leaf_count:
            watershed_panels = _fallback_watershed(
                working,
                source,
                line_mask,
                min_area_ratio=min_panel_area_ratio,
                max_area_ratio=max_panel_area_ratio,
                min_aspect_ratio=min_aspect_ratio,
                max_aspect_ratio=max_aspect_ratio,
            )
            panels.extend(watershed_panels)
            logger.info("Watershed fallback: %d candidates", len(watershed_panels))

        if enable_grid_fallback and len(panels) < separator_min_leaf_count:
            grid_panels = detect_panels_by_projection_manga(
                source,
                min_area_ratio=min_panel_area_ratio,
                max_area_ratio=max_panel_area_ratio,
                black_threshold=black_threshold,
            )
            panels.extend(grid_panels)
            logger.info("Grid fallback: %d candidates", len(grid_panels))

    # Remove the artificial frame candidate if one slips through.
    if frame_bbox is not None:
        panels = [
            p for p in panels
            if calculate_iou(p["bbox"], frame_bbox) < 0.90
        ]

    panels = remove_overlapping_panels(panels)
    panels = _unpad_panels(panels, pad, orig_w, orig_h) if pad else panels
    panels = improve_panel_order_manga(
        panels, orig_h, orig_w, reading_order=reading_order
    )

    if debug and out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, image_debug in debug_images.items():
            path = out_dir / f"{debug_stem}_debug_{name}.jpg"
            cv2.imwrite(str(path), image_debug)

    logger.info("Final panel count: %d", len(panels))
    return original, panels, debug_images


# ---------------------------------------------------------------------------
# Crop export
# ---------------------------------------------------------------------------


def _order_box_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    sums = points.sum(axis=1)
    diffs = points[:, 1] - points[:, 0]
    tl = points[np.argmin(sums)]
    br = points[np.argmax(sums)]
    tr = points[np.argmin(diffs)]
    bl = points[np.argmax(diffs)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _pad_bbox(
    bbox: List[int], pad_ratio: float, image_w: int, image_h: int
) -> List[int]:
    x, y, w, h = bbox
    px = int(round(w * pad_ratio))
    py = int(round(h * pad_ratio))
    return _safe_crop_bbox(
        [x - px, y - py, w + 2 * px, h + 2 * py], image_w, image_h
    )


def _export_crop(
    image: np.ndarray,
    panel: Dict[str, Any],
    pad_ratio: float,
    deskew: bool,
) -> np.ndarray:
    h, w = image.shape[:2]

    if deskew and "rotated_box" in panel:
        points = _order_box_points(np.asarray(panel["rotated_box"], dtype=np.float32))
        tl, tr, br, bl = points
        crop_w = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
        crop_h = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
        if crop_w >= 8 and crop_h >= 8:
            px = int(round(crop_w * pad_ratio))
            py = int(round(crop_h * pad_ratio))
            dst = np.array(
                [
                    [px, py],
                    [px + crop_w - 1, py],
                    [px + crop_w - 1, py + crop_h - 1],
                    [px, py + crop_h - 1],
                ],
                dtype=np.float32,
            )
            matrix = cv2.getPerspectiveTransform(points, dst)
            return cv2.warpPerspective(
                image,
                matrix,
                (crop_w + 2 * px, crop_h + 2 * py),
                borderMode=cv2.BORDER_REPLICATE,
            )

    x, y, bw, bh = _pad_bbox(panel["bbox"], pad_ratio, w, h)
    return image[y:y + bh, x:x + bw]


def export_crops(
    image: np.ndarray,
    panels: List[Dict[str, Any]],
    stem: str,
    out_dir: Path,
    *,
    fmt: str = "jpg",
    pad_ratio: float = 0.05,
    deskew: bool = True,
    start_index: int = 1,
    jpeg_quality: int = 95,
) -> List[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result: List[Path] = []

    for offset, panel in enumerate(panels):
        crop = _export_crop(image, panel, pad_ratio, deskew)
        index = start_index + offset
        path = out_dir / f"{stem}_panel_{index:03d}.{fmt}"

        if fmt.lower() in {"jpg", "jpeg"}:
            cv2.imwrite(
                str(path),
                crop,
                [cv2.IMWRITE_JPEG_QUALITY, int(max(1, min(100, jpeg_quality)))],
            )
        elif fmt.lower() == "png":
            cv2.imwrite(
                str(path), crop, [cv2.IMWRITE_PNG_COMPRESSION, 3]
            )
        else:
            raise ValueError(f"Unsupported output format: {fmt}")
        result.append(path)

    return result


def draw_panels(
    image: np.ndarray, panels: List[Dict[str, Any]], start_index: int = 1
) -> np.ndarray:
    out = image.copy()
    colors = {
        "separator": (0, 200, 0),
        "contour": (255, 0, 0),
        "watershed": (0, 0, 255),
        "grid": (0, 165, 255),
    }

    for offset, panel in enumerate(panels):
        index = start_index + offset
        x, y, w, h = panel["bbox"]
        color = colors.get(panel.get("method", ""), (180, 180, 180))
        if "rotated_box" in panel:
            points = np.asarray(panel["rotated_box"], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [points], True, color, 2)
        else:
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            out,
            str(index),
            (x + 4, min(y + 24, out.shape[0] - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return out


# ---------------------------------------------------------------------------
# Chapter / spread processing
# ---------------------------------------------------------------------------


def looks_like_spread(image: np.ndarray, aspect_threshold: float = 1.35) -> bool:
    h, w = image.shape[:2]
    return bool(h > 0 and (w / float(h)) >= aspect_threshold)


def split_spread(
    image: np.ndarray, search_ratio: float = 0.08
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Split a double-page scan near its visual center."""
    h, w = image.shape[:2]
    if w < 2:
        return image.copy(), image.copy(), 0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    center = w // 2
    radius = max(4, int(w * search_ratio))
    lo = max(1, center - radius)
    hi = min(w - 1, center + radius)
    strip = gray[:, lo:hi]
    profile = np.mean(strip, axis=0)
    baseline = float(np.median(gray))
    split_x = lo + int(np.argmax(np.abs(profile - baseline)))
    split_x = max(1, min(w - 1, split_x))
    return image[:, :split_x].copy(), image[:, split_x:].copy(), split_x


def process_page(
    image: ImageInput,
    out_dir: Path,
    stem: str,
    *,
    start_index: int = 1,
    fmt: str = "jpg",
    pad_ratio: float = 0.05,
    deskew: bool = True,
    crops: bool = True,
    save_json: bool = True,
    save_annotated: bool = True,
    jpeg_quality: int = 95,
    **detect_kwargs: Any,
) -> Tuple[List[Dict[str, Any]], List[Path], int]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_bgr, panels, _ = detect_panels_manga(
        image,
        out_dir,
        debug_stem=stem,
        **detect_kwargs,
    )

    if save_annotated:
        annotated = draw_panels(image_bgr, panels, start_index=start_index)
        cv2.imwrite(
            str(out_dir / f"{stem}_annotated.jpg"),
            annotated,
            [cv2.IMWRITE_JPEG_QUALITY, int(max(1, min(100, jpeg_quality)))],
        )

    crop_paths: List[Path] = []
    if panels and crops:
        crop_paths = export_crops(
            image_bgr,
            panels,
            stem,
            out_dir,
            fmt=fmt,
            pad_ratio=pad_ratio,
            deskew=deskew,
            start_index=start_index,
            jpeg_quality=jpeg_quality,
        )
    elif not panels:
        logger.warning("%s: no panels detected", stem)

    if save_json:
        with open(out_dir / f"{stem}_panels.json", "w", encoding="utf-8") as handle:
            json.dump(panels, handle, indent=2, ensure_ascii=False)

    return panels, crop_paths, start_index + len(panels)


def process_chapter(
    folder: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    reading_order: str = "rtl",
    split_spread_flag: bool = False,
    fmt: str = "jpg",
    pad_ratio: float = 0.05,
    deskew: bool = True,
    crops: bool = True,
    jpeg_quality: int = 95,
    save_json: bool = True,
    **detect_kwargs: Any,
) -> List[Dict[str, Any]]:
    """Process all page images in a directory."""
    folder = Path(folder)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not folder.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {folder}")

    pages = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )
    if not pages:
        raise FileNotFoundError(f"No image files found in {folder}")

    manifest: List[Dict[str, Any]] = []
    global_index = 1

    for page_path in pages:
        image = cv2.imread(str(page_path), cv2.IMREAD_COLOR)
        if image is None:
            logger.warning("Skipping unreadable image: %s", page_path)
            continue

        if split_spread_flag and looks_like_spread(image):
            left, right, _ = split_spread(image)
            sub_pages = (
                [("R", right), ("L", left)]
                if reading_order == "rtl"
                else [("L", left), ("R", right)]
            )
        else:
            sub_pages = [("", image)]

        for suffix, sub_image in sub_pages:
            stem = f"{page_path.stem}{suffix}"
            start_index = global_index
            panels, crop_paths, global_index = process_page(
                sub_image,
                out_dir,
                stem,
                start_index=start_index,
                fmt=fmt,
                pad_ratio=pad_ratio,
                deskew=deskew,
                crops=crops,
                save_json=save_json,
                save_annotated=True,
                jpeg_quality=jpeg_quality,
                reading_order=reading_order,
                **detect_kwargs,
            )

            for offset, panel in enumerate(panels):
                manifest.append(
                    {
                        "reading_index": start_index + offset,
                        "source_page": page_path.name,
                        "sub_page": suffix or None,
                        "file": crop_paths[offset].name if offset < len(crop_paths) else None,
                        "bbox": panel["bbox"],
                        "method": panel.get("method"),
                        "score": panel.get("score"),
                        "layout_path": panel.get("layout_path"),
                    }
                )

    with open(out_dir / "chapter_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    logger.info(
        "Chapter done: %d panels across %d pages -> %s",
        len(manifest),
        len(pages),
        out_dir,
    )
    return manifest


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classical-CV manga panel detector using separator-first segmentation."
    )
    parser.add_argument("input", type=Path, help="Page image or folder with --batch")
    parser.add_argument("--output", "-o", type=Path, default=Path("output"))
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--reading-order", choices=["rtl", "ltr"], default="rtl")
    parser.add_argument("--split-spread", action="store_true")
    parser.add_argument("--no-crops", dest="crops", action="store_false")
    parser.add_argument("--format", choices=["jpg", "png"], default="jpg")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--edge-method", choices=["white", "black", "dual", "canny"], default="black")
    parser.add_argument("--black-threshold", type=int, default=50)
    parser.add_argument("--white-threshold", type=int, default=235)
    parser.add_argument("--min-area-ratio", type=float, default=0.008)
    parser.add_argument("--max-area-ratio", type=float, default=0.65)
    parser.add_argument("--no-contour", dest="enable_contour", action="store_false")
    parser.add_argument("--no-watershed", dest="enable_watershed", action="store_false")
    parser.add_argument("--no-grid", dest="enable_grid", action="store_false")
    parser.add_argument("--no-denoise", dest="denoise", action="store_false")
    parser.add_argument("--denoise-kernel-ratio", type=float, default=0.003)
    parser.add_argument("--denoise-strength", type=int, default=1)
    parser.add_argument("--no-bleed-border", dest="bleed_border", action="store_false")
    parser.add_argument("--bleed-border-ratio", type=float, default=0.015)
    parser.add_argument("--no-deskew", dest="deskew", action="store_false")
    parser.add_argument("--rotation-threshold", type=float, default=3.0)
    parser.add_argument("--line-thickness-ratio", type=float, default=0.004)
    parser.add_argument("--morph-iterations", type=int, default=2)
    parser.add_argument("--no-separator", dest="separator_first", action="store_false")
    parser.add_argument("--separator-min-score", type=float, default=0.42)
    parser.add_argument("--separator-line-ratio", type=float, default=0.05)
    parser.add_argument("--separator-min-span-ratio", type=float, default=0.30)
    parser.add_argument("--separator-min-region-width-ratio", type=float, default=0.08)
    parser.add_argument("--separator-min-region-height-ratio", type=float, default=0.08)
    parser.add_argument("--separator-max-depth", type=int, default=8)
    parser.add_argument("--separator-min-leaf-count", type=int, default=2)
    parser.add_argument("--pad", type=float, default=0.05)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    detect_kwargs = {
        "reading_order": args.reading_order,
        "min_panel_area_ratio": args.min_area_ratio,
        "max_panel_area_ratio": args.max_area_ratio,
        "edge_method": args.edge_method,
        "black_threshold": args.black_threshold,
        "white_threshold": args.white_threshold,
        "line_thickness_ratio": args.line_thickness_ratio,
        "morph_iterations": args.morph_iterations,
        "denoise": args.denoise,
        "denoise_kernel_ratio": args.denoise_kernel_ratio,
        "denoise_strength": args.denoise_strength,
        "bleed_border": args.bleed_border,
        "bleed_border_ratio": args.bleed_border_ratio,
        "allow_rotated": args.deskew,
        "rotation_threshold_deg": args.rotation_threshold,
        "enable_contour_method": args.enable_contour,
        "enable_watershed_method": args.enable_watershed,
        "enable_grid_fallback": args.enable_grid,
        "separator_first": args.separator_first,
        "separator_min_score": args.separator_min_score,
        "separator_line_ratio": args.separator_line_ratio,
        "separator_min_span_ratio": args.separator_min_span_ratio,
        "separator_min_region_width_ratio": args.separator_min_region_width_ratio,
        "separator_min_region_height_ratio": args.separator_min_region_height_ratio,
        "separator_max_depth": args.separator_max_depth,
        "separator_min_leaf_count": args.separator_min_leaf_count,
        "debug": args.debug,
    }

    if args.batch:
        process_chapter(
            args.input,
            args.output,
            reading_order=args.reading_order,
            split_spread_flag=args.split_spread,
            fmt=args.format,
            pad_ratio=args.pad,
            deskew=args.deskew,
            crops=args.crops,
            jpeg_quality=args.jpeg_quality,
            **detect_kwargs,
        )
        return

    image = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {args.input}")

    args.output.mkdir(parents=True, exist_ok=True)
    if args.split_spread and looks_like_spread(image):
        left, right, _ = split_spread(image)
        sub_pages = (
            [("R", right), ("L", left)]
            if args.reading_order == "rtl"
            else [("L", left), ("R", right)]
        )
    else:
        sub_pages = [("", image)]

    index = 1
    for suffix, sub_image in sub_pages:
        stem = f"{args.input.stem}{suffix}"
        _, _, index = process_page(
            sub_image,
            args.output,
            stem,
            start_index=index,
            fmt=args.format,
            pad_ratio=args.pad,
            deskew=args.deskew,
            crops=args.crops,
            jpeg_quality=args.jpeg_quality,
            **detect_kwargs,
        )


if __name__ == "__main__":
    main()
