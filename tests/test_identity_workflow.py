"""Identity regression scenarios, real SQLite/export IO and mocked detector IO."""
import copy
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

import config
from description.response import parse_page_response
from panel_extractor.pipeline import detect_panels
from panel_extractor.types import PageDetection
from story_entities.manifest import build_manifest, file_hash
from story_entities.matching import resolve
from story_entities.pipeline import collect_instances, resolve_prepared, validate_calibration
from story_entities.runner import publish_result
from story_entities.storage import load_revision, save_revision


def fixture(count=5):
    items = [{"instance_id": f"i{i}", "prototype_eligible": True,
              "panel_id": f"panel{i}", "page_role": "story", "page_hash": "hash",
              "geometry_revision": "geometry", "reading_position": [0, i, 1],
              "quality_flags": [], "chapter": "chapter_001", "page": "page.png",
              "panel_number": i + 1} for i in range(count)]
    matrix = np.full((count, count), .04, dtype=np.float32)
    np.fill_diagonal(matrix, 0)
    reviews = {"assignments": [{"instance_id": f"i{i}", "entity_id": "hero", "form_id": "human",
                                "evidence": "Reviewed source", "accepted": True} for i in range(count)],
               "names": [{"entity_id": "hero", "name": "Rin", "source_instance_id": "i0",
                           "accepted": True, "evidence": "Reviewed self-introduction"}]}
    prepared = {"instances": items, "distances": matrix, "warnings": [], "manga": "Manga",
                "chapters": ["chapter_001"], "encoder": "test", "encoder_fingerprint": "test_model",
                "provider": "test", "group_threshold": .14, "elapsed_seconds": 0}
    return prepared, reviews


class IdentitySafetyTests(unittest.TestCase):
    def test_dual_encoder_links_simple_subgroup_with_two_panel_support(self):
        p, _ = fixture(10)
        primary = np.full((10, 10), .7, dtype=np.float32)
        style = np.full((10, 10), .7, dtype=np.float32)
        np.fill_diagonal(primary, 0); np.fill_diagonal(style, 0)
        for matrix in (primary, style):
            for group in ((0, 1, 2, 3), (4, 5), (6, 7, 8, 9)):
                for i in group:
                    for j in group:
                        matrix[i, j] = 0 if i == j else .08
        for i, j in ((4, 0), (5, 1)):
            primary[i, j] = primary[j, i] = .23
            style[i, j] = style[j, i] = .12
        result = resolve(p["instances"], primary, group_threshold=.20,
                         assignment_group_threshold=.08, style_distances=style)
        self.assertEqual(sorted(g["size"] for g in result["visual_groups"]), [4, 6])
        linked = next(g for g in result["visual_groups"] if g["size"] == 6)
        self.assertEqual(len(linked["style_links"]), 1)
        self.assertEqual(linked["style_links"][0]["status"], "proposed_style_link")
        self.assertEqual(result["identities"], [])

    def test_style_link_requires_two_panels_encoder_agreement_and_no_constraint(self):
        p, reviews = fixture(10)
        primary = np.full((10, 10), .7, dtype=np.float32)
        style = np.full((10, 10), .7, dtype=np.float32)
        np.fill_diagonal(primary, 0); np.fill_diagonal(style, 0)
        for matrix in (primary, style):
            for group in ((0, 1, 2, 3), (4, 5), (6, 7, 8, 9)):
                for i in group:
                    for j in group:
                        matrix[i, j] = 0 if i == j else .08
        for i, j in ((4, 0), (5, 1)):
            primary[i, j] = primary[j, i] = .23
            style[i, j] = style[j, i] = .12
        variants = []
        same_panel = copy.deepcopy(p["instances"])
        same_panel[5]["panel_id"] = same_panel[4]["panel_id"]
        variants.append((same_panel, style, {}))
        disagree = style.copy()
        for i, j in ((4, 6), (5, 7)):
            disagree[i, j] = disagree[j, i] = .04
        variants.append((p["instances"], disagree, {}))
        blocked = {"different": [{"left": "i4", "right": "i0", "evidence": "Reviewed different"}]}
        variants.append((p["instances"], style, blocked))
        for items, style_matrix, decisions in variants:
            with self.subTest(decisions=decisions, same_panel=items is same_panel):
                result = resolve(items, primary, reviews=decisions, group_threshold=.20,
                                 assignment_group_threshold=.08, style_distances=style_matrix)
                self.assertEqual(result["style_links"], [])
                self.assertEqual(sorted(g["size"] for g in result["visual_groups"]), [2, 4, 4])

    def test_anchor_refinement_collapses_a_confirmed_same_character_competitor(self):
        p, _ = fixture(25)
        primary = np.full((25, 25), .7, dtype=np.float32)
        style = np.full((25, 25), .7, dtype=np.float32)
        np.fill_diagonal(primary, 0); np.fill_diagonal(style, 0)
        groups = ((0, 1, 2, 3, 4, 5, 6, 7), (8, 9, 10, 11, 12),
                  (13, 14, 15, 16), (17, 18, 19, 20, 21, 22, 23, 24))
        for matrix in (primary, style):
            for group in groups:
                for i in group:
                    for j in group:
                        matrix[i, j] = 0 if i == j else .08
        for i, j in ((8, 0), (9, 1)):
            primary[i, j] = primary[j, i] = .10
            style[i, j] = style[j, i] = .10
        for i, j in ((13, 0), (14, 1)):
            primary[i, j] = primary[j, i] = .14
            style[i, j] = style[j, i] = .06
        for i, j in ((13, 8), (14, 9)):
            primary[i, j] = primary[j, i] = .15
            style[i, j] = style[j, i] = .04
        result = resolve(p["instances"], primary, group_threshold=.20,
                         assignment_group_threshold=.08, style_distances=style)
        self.assertEqual(sorted(g["size"] for g in result["visual_groups"]), [8, 17])
        self.assertEqual([link["stage"] for link in result["style_links"]],
                         ["direct_dual_encoder", "anchor_refinement"])
        self.assertEqual(result["identities"], [])
        blocked = {"different": [{"left": "i13", "right": "i8",
                                   "evidence": "Reviewed different"}]}
        result = resolve(p["instances"], primary, reviews=blocked, group_threshold=.20,
                         assignment_group_threshold=.08, style_distances=style)
        self.assertEqual(sorted(g["size"] for g in result["visual_groups"]), [4, 8, 13])
        self.assertEqual([link["stage"] for link in result["style_links"]],
                         ["direct_dual_encoder"])

    def test_groups_below_five_percent_of_largest_are_unclassified(self):
        p, _ = fixture(43)
        distances = np.full((43, 43), .7, dtype=np.float32)
        np.fill_diagonal(distances, 0)
        for group in (range(41), range(41, 43)):
            for i in group:
                for j in group:
                    distances[i, j] = 0 if i == j else .08
        result = resolve(p["instances"], distances, group_threshold=.20)
        self.assertEqual([g["size"] for g in result["visual_groups"]], [41])
        self.assertEqual(result["ungrouped_instance_ids"], ["i41", "i42"])
        self.assertEqual(result["visual_group_filter"]["minimum_occurrences"], 3)
        self.assertTrue(all(result["characters"][i]["visual_group_id"] is None
                            for i in (41, 42)))

    def test_two_panel_subgroup_uses_stricter_anchor_refinement(self):
        p, _ = fixture(23)
        primary = np.full((23, 23), .7, dtype=np.float32)
        style = np.full((23, 23), .7, dtype=np.float32)
        np.fill_diagonal(primary, 0); np.fill_diagonal(style, 0)
        groups = (range(8), range(8, 13), range(13, 15), range(15, 23))
        for matrix in (primary, style):
            for group in groups:
                for i in group:
                    for j in group:
                        matrix[i, j] = 0 if i == j else .08
        for i, j in ((8, 0), (9, 1)):
            primary[i, j] = primary[j, i] = .10
            style[i, j] = style[j, i] = .10
        for i, j in ((13, 0), (14, 1)):
            primary[i, j] = primary[j, i] = .12
            style[i, j] = style[j, i] = .06
        for i, j in ((13, 8), (14, 9)):
            primary[i, j] = primary[j, i] = .13
            style[i, j] = style[j, i] = .04
        result = resolve(p["instances"], primary, group_threshold=.20,
                         assignment_group_threshold=.08, style_distances=style)
        self.assertEqual(sorted(g["size"] for g in result["visual_groups"]), [8, 15])
        self.assertEqual([link["stage"] for link in result["style_links"]],
                         ["direct_dual_encoder", "strict_small_anchor_refinement"])

    def test_style_variants_group_without_becoming_confirmed_identities(self):
        p, _ = fixture(3)
        p["distances"] = np.array([[0, .18, .8], [.18, 0, .8], [.8, .8, 0]])
        strict = resolve(p["instances"], p["distances"], group_threshold=.14)
        relaxed = resolve(p["instances"], p["distances"], group_threshold=.20)
        self.assertEqual(strict["visual_groups"], [])
        self.assertEqual([g["instance_ids"] for g in relaxed["visual_groups"]], [["i0", "i1"]])
        self.assertEqual(relaxed["ungrouped_instance_ids"], ["i2"])
        self.assertIsNone(relaxed["characters"][2]["visual_group_id"])
        self.assertEqual(len(relaxed["characters"]), 3)
        self.assertEqual(relaxed["identities"], [])

    def test_relaxed_display_does_not_remove_competing_identity_evidence(self):
        p, reviews = fixture(4)
        reviews["assignments"] = reviews["assignments"][:2]
        p["distances"] = np.array([[0,.04,.04,.18], [.04,0,.04,.18],
                                   [.04,.04,0,.06], [.18,.18,.06,0]])
        # i3 joins the display group at .20, but is still an outside competitor
        # under the unchanged .14 identity rule and defeats the required margin.
        for threshold in (.14, .20):
            result = resolve(p["instances"], p["distances"], reviews=reviews,
                             group_threshold=threshold,
                             calibration={"max_distance": .1, "min_margin": .03})
            self.assertIsNone(result["characters"][2]["entity_id"])

    def test_duplicate_imports_neither_inflate_count_nor_supply_independent_support(self):
        p, reviews = fixture()
        for item in p["instances"]:
            item["story_panel_id"] = "one_imported_panel"
        result = resolve_prepared(p, reviews=reviews)
        self.assertEqual(result["identities"][0]["appearance_count"], 1)
        self.assertIsNone(result["identities"][0]["name"])
        reviews["assignments"] = reviews["assignments"][:2]
        result = resolve(p["instances"], p["distances"], reviews=reviews,
                         calibration={"max_distance": .1, "min_margin": .02})
        self.assertIsNone(result["characters"][-1]["entity_id"])

    def test_single_competing_reference_inside_query_group_blocks_acceptance(self):
        p, reviews = fixture(4)
        reviews["assignments"] = reviews["assignments"][:3]
        reviews["assignments"][2].update(entity_id="other", form_id="other_form")
        # Put query i3 in i2's group; i2 cannot offer two references, but must
        # still defeat the weaker match against hero's two references.
        p["distances"][2, 3] = p["distances"][3, 2] = .01
        result = resolve(p["instances"], p["distances"], reviews=reviews,
                         calibration={"max_distance": .1, "min_margin": .02})
        self.assertIsNone(result["characters"][-1]["entity_id"])

    def test_tied_candidates_abstain_even_when_configured_margin_is_zero(self):
        p, reviews = fixture()
        reviews["assignments"] = reviews["assignments"][:4]
        for entry in reviews["assignments"][2:]:
            entry.update(entity_id="other", form_id="other_form")
        result = resolve(p["instances"], p["distances"], reviews=reviews,
                         calibration={"max_distance": .1, "min_margin": 0})
        self.assertIsNone(result["characters"][-1]["entity_id"])

    def test_identity_reveal_delays_name_and_stale_reveal_disables_it(self):
        p, reviews = fixture()
        reviews["assignments"][0]["identity_reveal_instance_id"] = "i3"
        result = resolve_prepared(p, reviews=reviews)
        self.assertTrue(all(r["name"] is None for r in result["characters"][:3]))
        self.assertEqual(result["characters"][-1]["name"], "Rin")
        reviews["assignments"][0]["identity_reveal_instance_id"] = "stale"
        result = resolve_prepared(p, reviews=reviews)
        self.assertTrue(all(r["name"] is None for r in result["characters"]))

    def test_provisional_regions_do_not_inflate_recurrence(self):
        p, reviews = fixture()
        for item in p["instances"]:
            item["panel_provisional"] = True
        result = resolve_prepared(p, reviews=reviews)
        self.assertEqual(result["identities"][0]["appearance_count"], 0)

    def test_api_rejects_unvalidated_or_wrong_model_calibration(self):
        p, _ = fixture()
        valid = {"encoder_fingerprint": "test_model", "validated": True,
                 "evaluation_reference": "heldout.json", "max_distance": .1, "min_margin": .02}
        for key, bad in (("encoder_fingerprint", "other"), ("validated", False),
                         ("max_distance", float("nan")), ("min_margin", True),
                         ("evaluation_reference", "")):
            with self.subTest(key=key):
                calibration = dict(valid, **{key: bad})
                with self.assertRaises(ValueError):
                    resolve_prepared(p, calibration=calibration)
        self.assertEqual(validate_calibration(valid, "test_model"), valid)

    def test_pending_event_does_not_remap_to_changed_page_or_geometry(self):
        p, _ = fixture()
        data = {"source_hash": "hash", "geometry_revision": "geometry", "extraction_status": "complete",
                "events": [{"type": "transformation", "source_panel": 1, "target_panel": 2}]}
        events = {"schema_version": 1, "chapters": {"chapter_001": {"page.png": data}}}
        current = resolve_prepared(p, transformations=events)
        self.assertEqual(current["transformation_proposals"][0]["candidate_instances"]["source"], ["i0"])
        for key in ("source_hash", "geometry_revision"):
            stale = copy.deepcopy(events)
            stale["chapters"]["chapter_001"]["page.png"][key] = "different"
            result = resolve_prepared(p, transformations=stale)
            proposal = result["transformation_proposals"][0]
            self.assertEqual(proposal["candidate_instances"], {"source": [], "target": []})
            self.assertEqual(proposal["status"], "pending_review")
            self.assertEqual(result["identities"], [])

    def test_description_role_requires_current_provenance_and_preserves_review(self):
        p, _ = fixture(2)
        for item in p["instances"]:
            item.update(page_role="uncertain", page_role_source="unknown", prototype_eligible=False,
                        quality_flags=["non_story_or_unknown_role"], embedding_eligible=True)
        events = {"schema_version": 1, "chapters": {"chapter_001": {"page.png": {
            "source_hash": "hash", "geometry_revision": "geometry", "page_role": "story",
            "extraction_status": "complete", "events": []}}}}
        p["instances"][0].update(page_role="cover", page_role_source="review")
        result = resolve_prepared(p, transformations=events)
        self.assertEqual(result["characters"][0]["page_role"], "cover")
        self.assertTrue(result["characters"][1]["prototype_eligible"])
        self.assertFalse(p["instances"][1]["prototype_eligible"])  # replay input stays immutable

    def test_missing_or_hallucinated_panels_reject_event_evidence(self):
        payload = {"panels": [{"panel_number": 1, "description": "A dragon spreads its wings."}], "events": []}
        result = parse_page_response(json.dumps(payload), expected_panels=[1, 2])
        self.assertEqual(result["status"], "invalid_output")
        self.assertEqual(result["panels"], [])


class PersistenceTests(unittest.TestCase):
    def test_single_crops_are_archived_without_singleton_group_sections(self):
        from story_entities.runner import export_review_gallery
        p, _ = fixture(2)
        p["distances"] = np.array([[0,.9],[.9,0]])
        result = resolve_prepared(p)
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            page = export_review_gallery(result, [Image.new("RGB", (60, 60))] * 2, directory)
            html = page.read_text(encoding="utf-8")
            self.assertNotIn('<section><h2>group_', html)
            self.assertNotIn('<article id="i0">', html)
            self.assertIn('ungrouped.html', html)
            self.assertIn('<article id="i0">', (directory / 'ungrouped.html').read_text(encoding="utf-8"))

    def test_review_revocation_changes_current_revision_but_preserves_prior_evidence(self):
        p, reviews = fixture()
        first = resolve_prepared(p, reviews=reviews)
        second = resolve_prepared(p, reviews={})
        with tempfile.TemporaryDirectory() as temp:
            database = Path(temp) / "identities.sqlite3"
            save_revision(database, first, reviews, run_id="first")
            save_revision(database, second, {}, run_id="second")
            self.assertEqual(load_revision(database)["result"]["identities"], [])
            old = load_revision(database, first["revision"])
            self.assertEqual(old["result"]["identities"][0]["name"], "Rin")
            self.assertEqual(old["reviews"], reviews)
            with closing(sqlite3.connect(database)) as db:
                self.assertEqual(db.execute("SELECT count(*) FROM decisions").fetchone()[0], 6)

    def test_foreign_manga_cannot_replace_current_revision(self):
        p, reviews = fixture()
        first = resolve_prepared(p, reviews=reviews)
        with tempfile.TemporaryDirectory() as temp:
            database = Path(temp) / "identities.sqlite3"
            save_revision(database, first, reviews, run_id="first")
            foreign = dict(first, manga="Other")
            with self.assertRaises(ValueError):
                save_revision(database, foreign, reviews, run_id="foreign")
            self.assertEqual(load_revision(database)["result"]["manga"], "Manga")

    def test_standalone_export_keeps_backup_gallery_and_database(self):
        p, reviews = fixture(2)
        first = resolve_prepared(p, reviews=reviews)
        second = resolve_prepared(p, reviews={})
        crops = [Image.new("RGB", (60, 60), "white") for _ in range(2)]
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            publish_result(first, crops, output, [Path("chapter_001")], reviews=reviews)
            publish_result(second, crops, output, [Path("chapter_001")], reviews={})
            saved = json.loads((output / "character_classification.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["identities"], [])
            self.assertTrue(Path(saved["review_gallery"]).is_file())
            backups = list(output.glob("reid_runs/*/previous_manga_result.json"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(json.loads(backups[0].read_text(encoding="utf-8"))["revision"], first["revision"])
            self.assertEqual(load_revision(output / "character_identities.sqlite3")["result"]["revision"], second["revision"])


class ManifestTests(unittest.TestCase):
    def test_no_panel_fallback_preserves_original_and_remains_provisional(self):
        with tempfile.TemporaryDirectory() as temp:
            chapter = Path(temp) / "Manga" / "chapter_001"
            chapter.mkdir(parents=True)
            page = chapter / "page.png"
            Image.new("RGB", (200, 300), "white").save(page)
            digest = file_hash(page)
            settings = config.panel_extraction_kwargs()
            settings.update(enable_tiled_fallback=False, split_settings=None)
            with patch("panel_extractor.pipeline.load_session"), \
                    patch("panel_extractor.pipeline.predict", return_value=([], [], [])):
                detections = detect_panels(chapter, Path("unused.onnx"), Path(temp) / "output", **settings)
            self.assertEqual(file_hash(page), digest)
            self.assertTrue(detections[0].preview_path.is_file())
            manifest = build_manifest(detections)
            self.assertEqual(manifest["pages"][0]["panels"][0]["box"], [0, 0, 200, 300])
            self.assertTrue(manifest["pages"][0]["panels"][0]["provisional"])
            self.assertEqual(manifest["pages"][0]["page_role"], "uncertain")

    def test_reviewed_role_survives_rerun_but_not_changed_source(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "page.png"
            Image.new("RGB", (100, 100), "white").save(path)
            detections = [PageDetection(path, [[0, 0, 100, 100]], [.9])]
            original = build_manifest(detections)
            original["pages"][0].update(page_role="reference", page_role_source="review")
            repeated = build_manifest(detections, original)
            self.assertEqual(repeated["pages"][0]["page_id"], original["pages"][0]["page_id"])
            self.assertEqual(repeated["pages"][0]["page_role"], "reference")
            Image.new("RGB", (100, 100), "black").save(path)
            changed = build_manifest(detections, original)
            self.assertNotEqual(changed["pages"][0]["page_id"], original["pages"][0]["page_id"])
            self.assertEqual(changed["pages"][0]["page_role"], "uncertain")


if __name__ == "__main__":
    unittest.main()
