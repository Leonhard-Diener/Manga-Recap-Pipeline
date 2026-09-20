"""Exports for the standalone identity command."""
from __future__ import annotations

from datetime import datetime
import html
import shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps

from common.json_io import write_json
from .pipeline import read_json
from .storage import save_revision
from .matching import stable_id

CLASSIFICATION_FILE_NAME = "character_classification.json"


def export_review_gallery(result: dict, crops: list[Image.Image], directory: Path) -> Path:
    """Local, self-contained review page with exact IDs and original crop pixels."""
    directory.mkdir(parents=True, exist_ok=True)
    by_id = {}
    crop_by_id = {}
    source_links = {}
    for record, crop in zip(result["characters"], crops):
        image_name = record["instance_id"] + ".jpg"
        crop.save(directory / image_name, quality=90)
        by_id[record["instance_id"]] = record
        crop_by_id[record["instance_id"]] = crop
        source = Path(record["page"])
        if source.is_file() and str(source) not in source_links:
            pages = directory / "pages"
            pages.mkdir(exist_ok=True)
            name = stable_id("source_", str(source), record.get("page_hash")) + source.suffix
            shutil.copy2(source, pages / name)
            source_links[str(source)] = "pages/" + name
    chunks = ['<!doctype html><meta charset="utf-8"><title>Character review</title>',
              '<style>body{font:15px system-ui;background:#eee;padding:20px}section{background:white;padding:15px;margin:15px 0}article{display:inline-block;vertical-align:top;width:245px;margin:6px;overflow-wrap:anywhere}img{width:240px;height:240px;object-fit:contain}code{font-size:11px}</style>',
              '<h1>Visual proposals (not confirmed identities)</h1>',
              '<p>Review individual crops before assigning IDs. A group may contain mistakes. '
              'Copy exact instance IDs into the review file; pending model events never merge identities.</p>']
    groups = [g for g in result["visual_groups"] if g["size"] > 1]
    grouped = {identity for group in groups for identity in group["instance_ids"]}
    ungrouped = [r for r in result["characters"] if r["instance_id"] not in grouped]
    def card(record):
        identity = record["instance_id"]
        details = html.escape(str({k: record.get(k) for k in (
            "chapter", "panel_number", "quality_flags", "entity_id", "form_id", "identity_status", "name")}))
        source_link = source_links.get(record["page"])
        context = f'<a href="{html.escape(source_link, quote=True)}">Original page</a>' if source_link else ''
        return (f'<article id="{identity}"><img loading="lazy" src="{identity}.jpg"><br><code>{identity}</code>'
                f'<p>{html.escape(Path(record["page"]).name)} {context}</p><small>{details}</small></article>')
    if ungrouped:
        archive = chunks[:2] + ['<h1>Ungrouped observations</h1><p><a href="index.html">Back to groups</a></p>']
        archive.extend(card(record) for record in ungrouped)
        (directory / "ungrouped.html").write_text('\n'.join(archive), encoding="utf-8")
        group_filter = result.get("visual_group_filter", {})
        minimum = group_filter.get("minimum_occurrences", 1)
        chunks.append(f'<p>{len(groups)} groups. Groups below {minimum} distinct panel occurrence(s) removed; '
                      f'<a href="ungrouped.html">{len(ungrouped)} ungrouped observations</a> retained separately.</p>')
    events = result.get("transformation_proposals", [])
    if events:
        chunks.append('<section><h2>Possible form changes and related events — pending review</h2>')
        for event in events:
            title = f"{event.get('type')} — {event.get('chapter')}/{event.get('page')}"
            chunks.append('<h3>' + html.escape(title) + '</h3><p>' + html.escape(event.get("evidence", "")) + '</p>')
            for side in ("source", "target"):
                links = ' '.join(f'<a href="{"" if identity in grouped else "ungrouped.html"}#{html.escape(identity, quote=True)}">{html.escape(identity)}</a>'
                                 for identity in event.get("candidate_instances", {}).get(side, []))
                chunks.append('<p>' + side + ': ' + html.escape(event.get(side + "_description", "")) + '<br>' + links + '</p>')
            chunks.append('<small>' + html.escape(event.get("grounding_status", "unverified")) + '</small>')
        chunks.append('</section>')
    for group in sorted(groups, key=lambda g: (-g["size"], g["group_id"])):
        link_note = (f' — {len(group.get("style_links", []))} simple/detail subgroup link(s)'
                     if group.get("style_links") else '')
        chunks.append("<section><h2>" + html.escape(group["group_id"]) +
                      f' ({group["size"]} crops / {group.get("occurrence_count", group["size"])} panels)'
                      f'{link_note}</h2>')
        if group["size"] > 1:
            # Bounded quick overview; individual cards below include every crop.
            selected = group["instance_ids"][:96]
            sheet = Image.new("RGB", (8 * 150, ((len(selected) + 7) // 8) * 174), "white")
            draw = ImageDraw.Draw(sheet)
            for index, identity in enumerate(selected):
                x, y = (index % 8) * 150, (index // 8) * 174
                thumb = ImageOps.contain(crop_by_id[identity], (148, 148))
                sheet.paste(thumb, (x, y))
                draw.text((x, y + 150), identity[-12:], fill="black")
            sheet_name = group["group_id"] + ".jpg"
            sheet.save(directory / sheet_name, quality=90)
            chunks.append(f'<p><a href="{sheet_name}">Contact sheet (first 96 crops)</a></p>')
        for identity in group["instance_ids"]:
            chunks.append(card(by_id[identity]))
        chunks.append("</section>")
    path = directory / "index.html"
    path.write_text("\n".join(chunks), encoding="utf-8")
    return path


def publish_result(result, crops, destination: Path, chapters, *, reviews, calibration=None,
                   gallery=True, source_output: Path | None = None):
    source_output = source_output or destination
    run_dir = destination / "reid_runs" / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + result["revision"])
    # Snapshot evidence and decisions; never delete an earlier run or review file.
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "reviews_used.json", reviews)
    write_json(run_dir / "calibration_used.json", calibration)
    if gallery:
        result["review_gallery"] = str(export_review_gallery(result, crops, run_dir / "gallery").resolve())
    manifests = [read_json(source_output / chapter.name / "panel_manifest.json", {}) for chapter in chapters]
    result["identity_database"] = str((destination / "character_identities.sqlite3").resolve())
    save_revision(destination / "character_identities.sqlite3", result, reviews,
                  run_id=run_dir.name, manifests=manifests)
    write_json(run_dir / CLASSIFICATION_FILE_NAME, result)
    write_json(run_dir / "review_template.json", {
        "schema_version": 1, "assignments": [], "different": [], "names": [],
    })
    output_file = destination / CLASSIFICATION_FILE_NAME
    if output_file.is_file():
        shutil.copy2(output_file, run_dir / "previous_manga_result.json")
    write_json(output_file, result)
    for chapter in chapters:
        chapter_file = destination / chapter.name / CLASSIFICATION_FILE_NAME
        if chapter_file.is_file():
            shutil.copy2(chapter_file, run_dir / f"previous_{chapter.name}.json")
        chapter_result = {k: v for k, v in result.items() if k not in ("characters", "visual_groups")}
        chapter_result["chapter"] = chapter.name
        chapter_result["characters"] = [r for r in result["characters"] if r["chapter"] == chapter.name]
        write_json(chapter_file, chapter_result)
    for warning in result["warnings"]:
        print("WARNING:", warning)
    accepted = sum(x["entity_id"] is not None for x in result["characters"])
    print(f'{len(result["characters"])} instances; {len(result["visual_groups"])} proposed visual groups; '
          f'{accepted} accepted assignments; {len(result["identities"])} canonical entities.')
    if result.get("visual_group_filter"):
        group_filter = result["visual_group_filter"]
        print(f'Group display cutoff: {group_filter["minimum_occurrences"]} distinct panels '
              f'({group_filter["minimum_frequency_ratio"]:.0%} of largest proposal).')
    print(f'Saved {output_file}; {result["elapsed_seconds"]:.2f}s')
    if "review_gallery" in result:
        print("Review:", result["review_gallery"])
    return result
