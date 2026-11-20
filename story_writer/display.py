"""Aspect-aware display groups, independent of narration and evidence links."""
import json
from pathlib import Path
import shutil
import time

from PIL import Image

from common.json_io import write_json


DISPLAY_POLICY = "same_page_aspect_fit_v1"
VISUAL_HOLD_SECONDS = 0.75


def is_visual_hold(segment):
    return (segment.get("narration_policy") == "visual_hold_v1" and segment.get("text") == ""
            and segment.get("used_instance_ids") == [] and segment.get("name_uses") == [])


def narration_blocks(segments):
    """Return contiguous panel indices sharing one scene narration at its final panel."""
    blocks, index = [], 0
    while index < len(segments):
        first = segments[index]
        if first.get("narration_policy") == "scene_visual_v1":
            anchor_id = first.get("narrated_by_panel_id")
            end = next((i for i in range(index, len(segments))
                        if segments[i]["primary_panel_id"] == anchor_id), None)
            if end is None or end == index:
                raise ValueError("Scene visual has no later narration anchor.")
        else:
            end = index
        rows = segments[index:end + 1]
        anchor = rows[-1]
        ids = [row["primary_panel_id"] for row in rows]
        if end > index or anchor.get("scene_panel_ids") is not None:
            if (anchor.get("scene_panel_ids") != ids or not anchor.get("text", "").strip()
                    or anchor.get("supporting_panel_ids") != ids
                    or len({(row.get("chapter"), row.get("page")) for row in rows}) != 1):
                raise ValueError("Scene narration does not match its ordered page coverage.")
            for row in rows[:-1]:
                if (row.get("narration_policy") != "scene_visual_v1" or row.get("text") != ""
                        or row.get("narrated_by_panel_id") != anchor["primary_panel_id"]
                        or row.get("used_instance_ids") != [] or row.get("name_uses") != []
                        or row.get("supporting_panel_ids") != [row["primary_panel_id"]]):
                    raise ValueError("Invalid scene visual coverage row.")
        elif first.get("narration_policy") == "scene_visual_v1":
            raise ValueError("Scene visual cannot stand alone.")
        blocks.append(list(range(index, end + 1)))
        index = end + 1
    return blocks


def _placements(rows, sizes, layout):
    width, height = 1920, 1080
    placements = []
    for index, (row, (w, h)) in enumerate(zip(rows, sizes)):
        x, y, cell_width, cell_height = 0, 0, width, height
        if layout == "side_by_side_rtl":
            cell_width = width / 2
            x = (1 - index) * cell_width
        elif layout == "stacked":
            cell_height = height / 2
            y = index * cell_height
        scale = min(cell_width / w, cell_height / h)
        placed_width, placed_height = w * scale, h * scale
        placements.append({
            "panel_id": row["panel_id"], "panel_image": row["panel_image"],
            "x": (x + (cell_width - placed_width) / 2) / width,
            "y": (y + (cell_height - placed_height) / 2) / height,
            "width": placed_width / width, "height": placed_height / height,
        })
    return placements


def _fill(placements):
    return sum(p["width"] * p["height"] for p in placements)


def build_display_groups(segments, timeline):
    """Pair awkward adjacent panels only when more frame area can show artwork."""
    if [s["primary_panel_id"] for s in segments] != [r["panel_id"] for r in timeline]:
        raise ValueError("Story segments and timeline panel order do not match.")
    sizes = []
    for segment, row in zip(segments, timeline):
        if segment["panel_image"] != row["panel_image"] or segment["chapter"] != row["chapter"]:
            raise ValueError("Story and timeline panel sources do not match.")
        if row["estimated_end_seconds"] < row["estimated_start_seconds"]:
            raise ValueError("Timeline has negative duration.")
        with Image.open(row["panel_image"]) as image:
            sizes.append(image.size)
    groups = []
    index = 0
    while index < len(timeline):
        rows = timeline[index:index + 1]
        layout = "single"
        placed = _placements(rows, sizes[index:index + 1], layout)
        if index + 1 < len(timeline):
            first, second = segments[index:index + 2]
            if (first["chapter"], first["page"]) == (second["chapter"], second["page"]):
                pair = timeline[index:index + 2]
                pair_sizes = sizes[index:index + 2]
                single_fills = [_fill(_placements([row], [size], "single"))
                                for row, size in zip(pair, pair_sizes)]
                durations = [r["estimated_end_seconds"] - r["estimated_start_seconds"] for r in pair]
                total = sum(durations)
                baseline = (sum(f * d for f, d in zip(single_fills, durations)) / total
                            if total else sum(single_fills) / 2)
                choices = [(name, _placements(pair, pair_sizes, name))
                           for name in ("side_by_side_rtl", "stacked")]
                name, candidate = max(choices, key=lambda item: _fill(item[1]))
                # Pair only underfilled frames, with a meaningful area improvement.
                if min(single_fills) < 0.6 and _fill(candidate) > baseline + 0.05:
                    rows, layout, placed = pair, name, candidate
        groups.append({
            "panel_ids": [row["panel_id"] for row in rows],
            "layout": layout, "placements": placed,
            "start_word": rows[0]["start_word"], "end_word": rows[-1]["end_word"],
            "estimated_start_seconds": rows[0]["estimated_start_seconds"],
            "estimated_end_seconds": rows[-1]["estimated_end_seconds"],
        })
        index += len(rows)
    return groups


def refresh_panel_timeline(story_path: Path, timeline_path: Path):
    """Add display groups to saved output without regenerating any narration."""
    story = json.loads(story_path.read_text(encoding="utf-8"))
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    fingerprint = story.get("metadata", {}).get("input_fingerprint")
    if (not fingerprint or fingerprint != timeline.get("metadata", {}).get("input_fingerprint")
            or story.get("metadata", {}).get("run_id") != timeline.get("metadata", {}).get("run_id")
            or story.get("manga") != timeline.get("manga")):
        raise ValueError("Story and timeline have different or missing provenance.")
    groups = build_display_groups(story["segments"], timeline["timeline"])
    if timeline.get("display_groups") == groups and timeline.get("display_policy") == DISPLAY_POLICY:
        return timeline
    backup = timeline_path.with_name(f"{timeline_path.stem}.before_layout_{time.time_ns()}.json")
    shutil.copy2(timeline_path, backup)
    timeline.update(display_groups=groups, display_policy=DISPLAY_POLICY)
    write_json(timeline_path, timeline)
    return timeline
