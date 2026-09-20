"""Regression cases for dangerous identity mistakes; no model required."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from story_entities.matching import conservative_groups, resolve
from story_entities.pipeline import classify_manga, collect_instances


def instances(n=5):
    return [{"instance_id": f"i{x}", "prototype_eligible": True,
             "panel_id": f"c{x // 3}/page{x}#1", "page_role": "story",
             "reading_position": [x // 3, x, 1], "quality_flags": []} for x in range(n)]


def distances(n=5, value=.04):
    matrix = np.full((n, n), value, dtype=np.float32)
    np.fill_diagonal(matrix, 0)
    return matrix


def reviews(n=5):
    return {"schema_version": 1, "assignments": [
        {"instance_id": f"i{x}", "entity_id": "hero", "form_id": "human",
         "accepted": True, "evidence": "Reviewed source panel"} for x in range(n)],
        "names": [{"entity_id": "hero", "name": "Rin", "accepted": True,
                   "source_instance_id": "i0", "evidence": "Reviewed self-introduction"}]}


class MatchingTests(unittest.TestCase):
    def test_similarity_chain_does_not_collapse_characters(self):
        matrix = np.array([[0, .05, .4], [.05, 0, .08], [.4, .08, 0]])
        groups = conservative_groups(instances(3), matrix, .14)
        self.assertEqual(sorted(map(len, groups)), [1, 2])

    def test_permutation_does_not_change_visual_memberships(self):
        items = instances(4)
        matrix = np.array([[0,.02,.4,.4],[.02,0,.4,.4],[.4,.4,0,.03],[.4,.4,.03,0]])
        def members(items, matrix):
            return {frozenset(items[i]["instance_id"] for i in g)
                    for g in conservative_groups(items, matrix, .14)}
        order = [3, 0, 2, 1]
        self.assertEqual(members(items, matrix), members([items[i] for i in order], matrix[np.ix_(order, order)]))

    def test_equal_distance_ties_are_order_independent(self):
        items = instances(3)
        matrix = np.array([[0,.05,.5],[.05,0,.05],[.5,.05,0]])
        def members(order):
            rows = [items[i] for i in order]
            return {frozenset(rows[i]["instance_id"] for i in group) for group in
                    conservative_groups(rows, matrix[np.ix_(order, order)], .14)}
        self.assertEqual(members([0,1,2]), members([2,1,0]))

    def test_visual_groups_do_not_leak_names_or_identities(self):
        result = resolve(instances(), distances())
        self.assertEqual(len(result["visual_groups"]), 1)
        self.assertTrue(all(x["entity_id"] is None and x["name"] is None for x in result["characters"]))

    def test_three_panels_unnamed_four_panels_eligible(self):
        for count in (3, 4):
            result = resolve(instances(count), distances(count), reviews=reviews(count))
            self.assertEqual(result["identities"][0]["appearance_count"], count)
            self.assertEqual(result["identities"][0]["name"], "Rin" if count == 4 else None)

    def test_views_of_same_panel_count_once(self):
        items = instances()
        for item in items:
            item["panel_id"] = "same_panel"
        result = resolve(items, distances(), reviews=reviews())
        self.assertEqual(result["identities"][0]["appearance_count"], 1)
        self.assertIsNone(result["identities"][0]["name"])

    def test_unknown_panels_do_not_count(self):
        items = instances()
        for item in items:
            item["panel_id"] = None
        result = resolve(items, distances(), reviews=reviews())
        self.assertEqual(result["identities"][0]["appearance_count"], 0)

    def test_creature_form_shares_reviewed_entity_across_chapters(self):
        decisions = reviews()
        decisions["assignments"][3]["form_id"] = "dragon"
        decisions["assignments"][4]["form_id"] = "dragon"
        result = resolve(instances(), distances(), reviews=decisions)
        self.assertEqual(len(result["identities"]), 1)
        self.assertEqual(result["identities"][0]["form_ids"], ["dragon", "human"])
        self.assertEqual(result["identities"][0]["appearance_count"], 5)

    def test_low_quality_does_not_join_group_or_inherit_hero_name(self):
        items = instances()
        items[-1]["prototype_eligible"] = False
        result = resolve(items, distances(), reviews=reviews(4),
                         calibration={"max_distance": .1, "min_margin": .02})
        self.assertIsNone(result["characters"][-1]["entity_id"])
        self.assertIsNone(result["characters"][-1]["name"])

    def test_confirmed_different_pair_cannot_share_entity(self):
        decisions = reviews()
        decisions["different"] = [{"left": "i0", "right": "i1", "evidence": "Two different characters"}]
        with self.assertRaises(ValueError):
            resolve(instances(), distances(), reviews=decisions)

    def test_reflections_not_automatically_separated(self):
        items = instances(2)
        items[1]["panel_id"] = items[0]["panel_id"]
        self.assertEqual(len(conservative_groups(items, distances(2), .1)), 1)

    def test_automatic_match_needs_calibration_and_two_other_panels(self):
        for calibration, seeds, expected in ((None, 2, None),
                ({"max_distance": .1, "min_margin": .02}, 1, None),
                ({"max_distance": .1, "min_margin": .02}, 2, "hero")):
            result = resolve(instances(3), distances(3), reviews=reviews(seeds), calibration=calibration)
            self.assertEqual(result["characters"][-1]["entity_id"], expected)

    def test_competing_main_characters_force_abstention(self):
        items = instances(5)
        decisions = reviews(4)
        for entry in decisions["assignments"][2:]:
            entry.update(entity_id="other", form_id="other_human")
        result = resolve(items, distances(), reviews=decisions,
                         calibration={"max_distance": .1, "min_margin": .02})
        self.assertIsNone(result["characters"][-1]["entity_id"])

    def test_removing_review_revokes_identity_and_name(self):
        accepted = resolve(instances(), distances(), reviews=reviews())
        revoked = resolve(instances(), distances(), reviews={})
        self.assertTrue(accepted["identities"])
        self.assertFalse(revoked["identities"])
        self.assertTrue(all(x["name"] is None for x in revoked["characters"]))

    def test_late_name_reveal_does_not_name_earlier_actions(self):
        decisions = reviews()
        decisions["names"][0]["source_instance_id"] = "i3"
        result = resolve(instances(), distances(), reviews=decisions)
        self.assertFalse(result["characters"][0]["naming_allowed"])
        self.assertIsNone(result["characters"][0]["name"])
        self.assertTrue(result["characters"][-1]["naming_allowed"])
        self.assertFalse(any(x["narration_allowed"] for x in result["characters"]))

    def test_stale_source_cannot_release_name(self):
        decisions = reviews()
        decisions["names"][0]["source_instance_id"] = "old_crop"
        result = resolve(instances(), distances(), reviews=decisions)
        self.assertIsNone(result["identities"][0]["name"])

    def test_cover_appearances_do_not_make_character_recurring(self):
        items = instances()
        for item in items[2:]:
            item["page_role"] = "cover"
        result = resolve(items, distances(), reviews=reviews())
        self.assertEqual(result["identities"][0]["appearance_count"], 2)
        self.assertIsNone(result["identities"][0]["name"])

    def test_confirmed_separation_prevents_automatic_propagation(self):
        decisions = reviews(2)
        decisions["different"] = [{"left":"i0", "right":"i2", "evidence":"Different actor"}]
        result = resolve(instances(3), distances(3), reviews=decisions,
                         calibration={"max_distance": .1, "min_margin": .02})
        self.assertIsNone(result["characters"][2]["entity_id"])


class PipelineTests(unittest.TestCase):
    def test_events_never_merge_entities_and_bad_crops_not_encoded(self):
        items = instances(3)
        for i, item in enumerate(items):
            item.update(chapter="chapter_001", page="page.png", panel_number=i+1,
                        page_hash="hash", geometry_revision="geometry")
        items[-1]["prototype_eligible"] = False
        class FakeEncoder:
            default_threshold = .14
            fingerprint = "test_encoder"
            kind = "test"
            runtime = "test"
            def distances(self, rows, crops, cache):
                self.count = len(rows)
                return distances(len(rows))
        encoder = FakeEncoder()
        events = {"schema_version":1, "chapters":{"chapter_001":{"page.png":{
            "extraction_status":"complete", "source_hash":"hash", "geometry_revision":"geometry", "events":[{
                "type":"transformation", "source_panel":1, "target_panel":2,
                "status":"accepted", "confidence":1.0}]}}}}
        with patch("story_entities.pipeline.collect_instances", return_value=(items,[None]*3,[])):
            result, _, _ = classify_manga(Path("Manga"), [Path("chapter_001")], Path("out"),
                encoder=encoder, cache_dir=Path("cache"), transformations=events)
        self.assertEqual(encoder.count, 2)
        self.assertFalse(result["identities"])
        self.assertEqual(result["transformation_proposals"][0]["status"], "pending_review")
        self.assertEqual(result["transformation_proposals"][0]["candidate_instances"]["source"], ["i0"])


class InputTests(unittest.TestCase):
    def test_small_nonhuman_retained_deduplicated_and_panel_mapped(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manga = root / "Manga"
            chapter = manga / "chapter_001"
            chapter.mkdir(parents=True)
            image = Image.new("RGB", (200, 200), "white")
            ImageDraw.Draw(image).rectangle((10, 10, 50, 50), fill="black")
            page = chapter / "00001.png"
            image.save(page)
            out = root / "output"
            chapter_out = out / chapter.name
            chapter_out.mkdir(parents=True)
            data = {"detections": [
                {"page": str(page), "box": [0, 0, 70, 70], "score": .9, "species": "dragon"},
                {"page": str(page), "box": [1, 1, 70, 70], "score": .8},
                {"page": str(page), "box": [100, 100, 120, 120], "score": .9},
            ]}
            for detection in data["detections"]:
                detection["source_hash"] = hashlib.sha256(page.read_bytes()).hexdigest()
            (chapter_out / "entity_detections.json").write_text(json.dumps(data))
            manifest = {"pages": [{"page": str(page), "page_role": "story",
                "source_hash": hashlib.sha256(page.read_bytes()).hexdigest(),
                "panels": [{"panel_number": 1, "box": [0, 0, 200, 200]}]}]}
            (chapter_out / "panel_manifest.json").write_text(json.dumps(manifest))
            records, crops, warnings = collect_instances(manga, [chapter], out)
            self.assertEqual(len(records), 2)
            self.assertTrue(records[0]["prototype_eligible"])
            self.assertIn("tiny", records[1]["quality_flags"])
            self.assertIsNotNone(records[0]["panel_id"])
            first_id = records[0]["instance_id"]
            records2, _, _ = collect_instances(manga, [chapter], out)
            self.assertEqual(records2[0]["instance_id"], first_id)
            image.putpixel((0, 0), (1, 2, 3))
            image.save(page)
            records3, _, _ = collect_instances(manga, [chapter], out)
            self.assertEqual(records3, [])  # Stale proposals cannot move to new pixels.
            for detection in data["detections"]:
                detection["source_hash"] = hashlib.sha256(page.read_bytes()).hexdigest()
            (chapter_out / "entity_detections.json").write_text(json.dumps(data))
            records3, _, _ = collect_instances(manga, [chapter], out)
            self.assertNotEqual(records3[0]["instance_id"], first_id)
            self.assertIsNone(records3[0]["panel_id"])


if __name__ == "__main__":
    unittest.main()
