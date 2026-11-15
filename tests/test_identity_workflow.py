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
from panel_extractor.runner import extract_chapter, _apply_reviewed_panel_corrections
from panel_extractor.types import Detection, PageDetection
from story_entities.manifest import build_manifest, file_hash, serialise_proposals
from story_entities.matching import resolve
from story_entities.pipeline import collect_instances, resolve_prepared, validate_calibration, _build_story_events
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

    def test_certain_visible_transformation_connects_forms_and_display_name(self):
        p, _ = fixture(4)
        p["distances"] = np.array([
            [0, .8, .04, .8],
            [.8, 0, .8, .04],
            [.04, .8, 0, .8],
            [.8, .04, .8, 0],
        ])
        events = {"schema_version": 1, "chapters": {"chapter_001": {"page.png": {
            "source_hash": "hash", "geometry_revision": "geometry",
            "page_role": "story", "extraction_status": "complete",
            "events": [{
                "type": "transformation", "source_panel": 1, "target_panel": 2,
                "source_description": "a horned figure",
                "target_description": "a winged dragon",
                "evidence": "The same body visibly grows wings.",
                "evidence_type": "visible_change", "confidence_percent": 100,
                "status": "pending_review",
            }],
        }}}}

        result = resolve_prepared(p, transformations=events)

        proposal = result["transformation_proposals"][0]
        self.assertEqual(proposal["status"], "accepted_automatic")
        self.assertEqual(proposal["acceptance_policy"], "qwen_direct_visible_change_v1")
        transformed = result["characters"][:2]
        self.assertEqual(transformed[0]["entity_id"], transformed[1]["entity_id"])
        self.assertTrue(transformed[0]["entity_id"].startswith("entity_qwen_"))
        self.assertTrue(all(row["identity_status"] == "accepted_qwen_transformation"
                            for row in transformed))
        self.assertEqual(len(result["visual_groups"]), 2)
        self.assertEqual(
            result["visual_groups"][0]["display_name"],
            result["visual_groups"][1]["display_name"],
        )

    def test_automatic_transformation_fails_closed(self):
        base_event = {
            "type": "transformation", "source_panel": 1, "target_panel": 2,
            "source_description": "a horned figure",
            "target_description": "a winged dragon",
            "evidence": "The same body visibly grows wings.",
            "evidence_type": "visible_change", "confidence_percent": 100,
        }
        variants = [
            ("confidence_percent", 99, "model_certainty_below_100"),
            ("evidence_type", "explicit_text", "direct_visible_change_required"),
            ("target_panel", 3, "adjacent_endpoint_panels_required"),
            ("type", "summoning", "event_type_does_not_prove_same_identity"),
        ]
        for key, value, reason in variants:
            with self.subTest(key=key, value=value):
                p, _ = fixture(3)
                event = dict(base_event, **{key: value})
                data = {"source_hash": "hash", "geometry_revision": "geometry",
                        "page_role": "story", "extraction_status": "complete",
                        "events": [event]}
                transformations = {"schema_version": 1, "chapters": {
                    "chapter_001": {"page.png": data}}}
                result = resolve_prepared(p, transformations=transformations)
                proposal = result["transformation_proposals"][0]
                self.assertEqual(proposal["status"], "pending_review")
                self.assertEqual(proposal["automatic_rejection_reason"], reason)
                self.assertEqual(result["identities"], [])

    def test_automatic_transformation_rejects_ambiguous_or_conflicting_endpoints(self):
        event = {
            "type": "transformation", "source_panel": 1, "target_panel": 2,
            "source_description": "a horned figure",
            "target_description": "a winged dragon",
            "evidence": "The same body visibly grows wings.",
            "evidence_type": "visible_change", "confidence_percent": 100,
        }
        data = {"source_hash": "hash", "geometry_revision": "geometry",
                "page_role": "story", "extraction_status": "complete",
                "events": [event]}
        transformations = {"schema_version": 1, "chapters": {
            "chapter_001": {"page.png": data}}}

        ambiguous, _ = fixture(3)
        ambiguous["instances"][2]["panel_number"] = 1
        result = resolve_prepared(ambiguous, transformations=transformations)
        self.assertEqual(
            result["transformation_proposals"][0]["automatic_rejection_reason"],
            "exactly_one_instance_per_endpoint_required",
        )
        self.assertEqual(result["identities"], [])

        conflicting, reviews = fixture(2)
        reviews["assignments"][1].update(entity_id="villain", form_id="dragon")
        result = resolve_prepared(
            conflicting,
            reviews=reviews,
            transformations=transformations,
        )
        self.assertEqual(
            result["transformation_proposals"][0]["automatic_rejection_reason"],
            "conflicting_accepted_entities",
        )
        self.assertEqual(len(result["identities"]), 2)

    def test_labeled_transformation_selects_exact_people_on_crowded_panels(self):
        p, _ = fixture(4)
        p["instances"][0]["grounding_ids"] = ["det_source"]
        p["instances"][1]["grounding_ids"] = ["det_target"]
        p["instances"][2]["grounding_ids"] = ["det_bystander"]
        p["instances"][2]["panel_number"] = 1
        event = {
            "type": "transformation", "source_panel": 1, "target_panel": 2,
            "source_description": "a horned figure", "target_description": "a dragon",
            "evidence": "The same body visibly grows wings.",
            "evidence_type": "visible_change", "confidence_percent": 100,
            "source_detection_id": "det_source", "target_detection_id": "det_target",
        }
        transformations = {"schema_version": 1, "chapters": {"chapter_001": {"page.png": {
            "source_hash": "hash", "geometry_revision": "geometry", "page_role": "story",
            "extraction_status": "complete", "events": [event],
        }}}}
        result = resolve_prepared(p, transformations=transformations)
        proposal = result["transformation_proposals"][0]
        self.assertEqual(proposal["grounding_method"], "labeled_detection")
        self.assertEqual(proposal["status"], "accepted_automatic")
        self.assertEqual(proposal["accepted_instance_ids"], ["i0", "i1"])

    def test_grounded_actions_are_accepted_without_model_confidence_or_review(self):
        p, _ = fixture(2)
        p["instances"][0]["grounding_ids"] = ["det_actor"]
        p["instances"][1].update(
            grounding_ids=["det_target"], panel_number=1, panel_id="panel0",
            reading_position=[0, 0, 1],
        )
        descriptions = {"chapter_001": {"page.png": {
            "source_hash": "hash", "geometry_revision": "geometry",
            "page_role": "story", "extraction_status": "complete",
            "characters": [
                {"detection_id": "det_actor", "visual_description": "horned girl"},
                {"detection_id": "det_target", "visual_description": "uniformed student"},
            ],
            "panels": [{"panel_number": 1,
                         "description": "A horned girl points a sword at a student.",
                         "actions": [{
                "actor_detection_id": "det_actor", "target_detection_id": "det_target",
                "text": "points a sword at",
            }]}],
        }}}
        result = resolve_prepared(p, descriptions=descriptions)
        event = result["story_events"][0]
        self.assertEqual(event["status"], "accepted_automatic")
        self.assertEqual(event["acceptance_policy"], "grounded_visible_action_format_v3")
        self.assertEqual(event["actor_instance_id"], "i0")
        self.assertNotIn("confidence_percent", event)
        self.assertEqual(result["characters"][0]["stable_description"], "horned girl")

        reviews = {"reviewed_actions": [{
            "status": "accepted_reviewed", "fact_id": "reviewed_action_1",
            "panel_id": "panel0", "actor_instance_id": "i0",
            "target_instance_id": "i1", "text": "points a sword at",
            "source_hash": "hash", "geometry_revision": "geometry",
            "grounding_revision": None, "reason": "Checked against panel",
        }]}
        reviewed = resolve_prepared(p, descriptions=descriptions, reviews=reviews)
        accepted = [row for row in reviewed["story_events"]
                    if row["status"] == "accepted_reviewed"]
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]["actor_instance_id"], "i0")
        self.assertEqual(accepted[0]["target_instance_id"], "i1")

    def test_grounded_action_does_not_need_its_verb_repeated_in_summary(self):
        prepared, _ = fixture(1)
        actor = prepared["instances"][0]
        actor.update(grounding_ids=["det_actor"], narration_eligibility="eligible_automatic")
        descriptions = {"chapter_001": {"page.png": {
            "source_hash": "hash", "geometry_revision": "geometry", "extraction_status": "complete",
            "panels": [{"panel_number": 1, "description": "A student stands outside a school.",
                        "actions": [{"actor_detection_id": "det_actor", "target_detection_id": None,
                                     "text": "raises an umbrella above the head"}]}],
        }}}
        result = {"characters": [actor]}
        event = _build_story_events(result, descriptions)[0]
        self.assertEqual(event["status"], "accepted_automatic")
        self.assertEqual(event["text"], "raises an umbrella above the head")
        self.assertEqual(event["acceptance_policy"], "grounded_visible_action_format_v3")

        descriptions["chapter_001"]["page.png"]["source_hash"] = "stale"
        stale = _build_story_events(result, descriptions)[0]
        self.assertEqual(stale["status"], "pending_review")
        self.assertEqual(stale["automatic_rejection_reason"], "stale_or_unverified_grounding")

    def test_repeated_depictions_cannot_automatically_interact(self):
        prepared, _ = fixture(2)
        for index, row in enumerate(prepared["instances"]):
            row.update(panel_number=1, panel_id="panel0", grounding_ids=[f"det_{index}"],
                       entity_id="same_student", narration_eligibility="eligible_automatic")
        descriptions = {"chapter_001": {"page.png": {
            "source_hash": "hash", "geometry_revision": "geometry", "extraction_status": "complete",
            "panels": [{"panel_number": 1, "description": "Two depictions of a uniformed student.",
                        "actions": [{"actor_detection_id": "det_0", "target_detection_id": "det_1",
                                     "text": "points at the other figure"}]}],
        }}}
        result = {"characters": prepared["instances"]}
        event = _build_story_events(result, descriptions)[0]
        self.assertEqual(event["status"], "pending_review")
        self.assertEqual(event["automatic_rejection_reason"], "same_identity_occurrences_require_review")
        self.assertEqual(event["text"], "points at the other figure")
        self.assertNotIn("acceptance_policy", event)

        result["characters"][1]["entity_id"] = "other_student"
        self.assertEqual(_build_story_events(result, descriptions)[0]["status"], "accepted_automatic")
        action = descriptions["chapter_001"]["page.png"]["panels"][0]["actions"][0]
        action.update(target_detection_id="det_0", text="scratches own arm")
        self.assertEqual(_build_story_events(result, descriptions)[0]["status"], "accepted_automatic")

    def test_valid_text_transcriptions_are_accepted_automatically(self):
        p, _ = fixture(1)
        p["instances"][0]["grounding_ids"] = ["det_speaker"]
        p["instances"][0]["grounding_revision"] = "grounding"
        descriptions = {"chapter_001": {"page.png": {
            "source_hash": "hash", "geometry_revision": "geometry",
            "grounding_revision": "grounding", "extraction_status": "complete",
            "panels": [], "characters": [{
                "detection_id": "det_speaker",
                "visual_description": "dark-haired boy in a jacket",
                "representation_type": "live_character",
            }],
            "text_proposals": [{
                "proposal_id": "text_1", "panel_id": "panel0", "panel_number": 1,
                "kind": "dialogue", "text": "We need to leave.",
                "speaker_detection_id": "det_speaker", "status": "pending_review",
            }],
        }}}
        result = resolve_prepared(p, descriptions=descriptions)
        proposal = result["text_proposals"][0]
        self.assertEqual(proposal["status"], "accepted_automatic")
        self.assertEqual(proposal["acceptance_policy"], "grounded_text_transcription_format_v2")
        self.assertEqual(proposal["speaker_instance_id"], "i0")
        self.assertEqual(proposal["candidate_speaker_instance_id"], "i0")
        self.assertEqual(proposal["source_hash"], "hash")
        self.assertNotIn("reviewed_text_facts", proposal)
        changed = copy.deepcopy(descriptions)
        changed["chapter_001"]["page.png"]["text_proposals"][0]["text"] = "We need to run."
        self.assertNotEqual(
            result["revision"], resolve_prepared(p, descriptions=changed)["revision"],
        )
        publication = copy.deepcopy(descriptions)
        publication["chapter_001"]["page.png"]["text_proposals"][0].update(
            kind="publication", text="TWO-PART SPECIAL ONE-SHOT", speaker_detection_id=None,
        )
        exported = resolve_prepared(p, descriptions=publication)["text_proposals"][0]
        self.assertEqual(exported["status"], "accepted_automatic")
        self.assertEqual(exported["kind"], "publication")
        self.assertEqual(exported["text"], "TWO-PART SPECIAL ONE-SHOT")
        self.assertIsNone(exported["speaker_instance_id"])

    def test_recurrent_live_character_becomes_automatic_identity(self):
        p, _ = fixture(4)
        for index, item in enumerate(p["instances"]):
            item.update(
                grounding_ids=[f"det{index}"], grounding_revision="grounding",
                box=[index * 20, 0, index * 20 + 10, 20],
            )
        descriptions = {"chapter_001": {"page.png": {
            "source_hash": "hash", "geometry_revision": "geometry",
            "grounding_revision": "grounding", "extraction_status": "complete",
            "panels": [],
            "characters": [{
                "detection_id": f"det{index}", "visual_description": "dark-haired boy in a jacket",
            } for index in range(4)],
        }}}
        result = resolve_prepared(p, descriptions=descriptions)
        self.assertEqual(len(result["identities"]), 1)
        self.assertTrue(all(
            row["identity_status"] == "accepted_automatic"
            for row in result["characters"]
        ))
        self.assertEqual(
            result["identities"][0]["acceptance_policy"],
            "multi_signal_identity_v1",
        )

    def test_toy_is_not_merged_into_automatic_live_identity(self):
        p, _ = fixture(4)
        for index, item in enumerate(p["instances"]):
            item.update(
                grounding_ids=[f"det{index}"], grounding_revision="grounding",
                box=[index * 20, 0, index * 20 + 10, 20],
            )
        descriptions = {"chapter_001": {"page.png": {
            "source_hash": "hash", "geometry_revision": "geometry",
            "grounding_revision": "grounding", "extraction_status": "complete",
            "panels": [],
            "characters": [{
                "detection_id": f"det{index}",
                "visual_description": (
                    "small toy figure of a dark-haired boy"
                    if index == 3 else "dark-haired boy in a jacket"
                ),
            } for index in range(4)],
        }}}
        result = resolve_prepared(p, descriptions=descriptions)
        self.assertEqual(result["characters"][3]["representation_type"], "toy")
        self.assertEqual(result["characters"][3]["identity_status"], "unknown")
        self.assertTrue(all(
            row["identity_status"] == "accepted_automatic"
            for row in result["characters"][:3]
        ))

    def test_explicitly_different_simultaneous_body_is_not_merged(self):
        p, _ = fixture(4)
        p["instances"][1]["panel_id"] = p["instances"][0]["panel_id"]
        for index, item in enumerate(p["instances"]):
            item.update(
                grounding_ids=[f"det{index}"], grounding_revision="grounding",
                box=[index * 20, 0, index * 20 + 10, 20],
            )
        descriptions = {"chapter_001": {"page.png": {
            "source_hash": "hash", "geometry_revision": "geometry",
            "grounding_revision": "grounding", "extraction_status": "complete",
            "panels": [],
            "characters": [{
                "detection_id": f"det{index}",
                "visual_description": (
                    "light-haired girl in a dress"
                    if index == 1 else "dark-haired boy in a jacket"
                ),
            } for index in range(4)],
        }}}
        result = resolve_prepared(p, descriptions=descriptions)
        self.assertEqual(len(result["identities"]), 1)
        self.assertEqual(result["characters"][1]["identity_status"], "unknown")
        self.assertTrue(all(
            result["characters"][index]["identity_status"] == "accepted_automatic"
            for index in (0, 2, 3)
        ))

    def test_reviewed_cross_page_transformation_keeps_explicit_endpoints(self):
        p, _ = fixture(2)
        p["instances"][0]["grounding_revision"] = "grounding_1"
        p["instances"][1].update(
            page="other.png", page_hash="hash_2", geometry_revision="geometry_2",
            grounding_revision="grounding_2", panel_id="panel_other",
        )
        reviews = {"reviewed_transformations": [{
            "status": "accepted_reviewed", "fact_id": "reviewed_change_1",
            "type": "transformation", "source_instance_id": "i0",
            "target_instance_id": "i1", "evidence": "Reviewed visible change",
            "source_page_hash": "hash", "source_geometry_revision": "geometry",
            "source_grounding_revision": "grounding_1",
            "target_page_hash": "hash_2", "target_geometry_revision": "geometry_2",
            "target_grounding_revision": "grounding_2",
        }]}
        result = resolve_prepared(p, reviews=reviews)
        event = result["transformation_proposals"][0]
        self.assertEqual(event["status"], "accepted_reviewed")
        self.assertEqual(event["accepted_instance_ids"], ["i0", "i1"])
        self.assertNotEqual(event["source_panel_id"], event["target_panel_id"])

    def test_rejected_body_remains_addressable_but_not_narration_eligible(self):
        p, _ = fixture(1)
        p["instances"][0]["grounding_revision"] = "grounding"
        descriptions = {"chapter_001": {"page.png": {
            "source_hash": "hash", "geometry_revision": "geometry",
            "grounding_revision": "grounding", "extraction_status": "complete",
            "characters": [], "panels": [],
        }}}
        reviews = {"eligibility": [{
            "instance_id": "i0", "decision": "rejected_non_character",
            "source_hash": "hash", "geometry_revision": "geometry",
            "grounding_revision": "grounding", "reason": "Reviewed object detection",
        }]}
        result = resolve_prepared(p, descriptions=descriptions, reviews=reviews)
        self.assertEqual(len(result["characters"]), 1)
        self.assertEqual(
            result["characters"][0]["narration_eligibility"],
            "rejected_non_character",
        )

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

    def test_page_role_does_not_depend_on_transformation_success(self):
        p, _ = fixture(1)
        p["instances"][0].update(
            page_role="uncertain", page_role_source="unknown", prototype_eligible=False,
            quality_flags=["non_story_or_unknown_role"], embedding_eligible=True,
        )
        descriptions = {"chapter_001": {"page.png": {
            "source_hash": "hash", "geometry_revision": "geometry",
            "extraction_status": "complete", "page_role": "story",
            "characters": [], "panels": [],
        }}}
        transformations = {"schema_version": 1, "chapters": {"chapter_001": {
            "page.png": {
                "source_hash": "hash", "geometry_revision": "geometry",
                "extraction_status": "invalid_output", "page_role": "story", "events": [],
            },
        }}}
        result = resolve_prepared(
            p, descriptions=descriptions, transformations=transformations,
        )
        self.assertEqual(result["characters"][0]["page_role"], "story")
        self.assertTrue(result["characters"][0]["prototype_eligible"])

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
            templates = list(output.glob("reid_runs/*/review_template.json"))
            self.assertTrue(templates)
            template = json.loads(templates[0].read_text(encoding="utf-8"))
            self.assertEqual(template["reviewed_text_facts"], [])
            self.assertIn("text_proposals", template)


class ManifestTests(unittest.TestCase):
    def test_reviewed_panel_boxes_and_order_are_source_revision_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            page = root / "input" / "Manga" / "chapter_001" / "page.png"
            page.parent.mkdir(parents=True)
            Image.new("RGB", (100, 120), "white").save(page)
            detection = PageDetection(
                page_path=page,
                boxes=[[0, 0, 50, 60], [50, 60, 100, 120]],
                scores=[.9, .8],
            )
            raw = build_manifest([detection])
            correction = root / "panel_review_corrections.json"
            correction.write_text(json.dumps({
                "schema_version": 1,
                "pages": [{
                    "page": page.name, "status": "accepted_reviewed",
                    "source_hash": raw["pages"][0]["source_hash"],
                    "base_geometry_revision": raw["pages"][0]["geometry_revision"],
                    "boxes": [[0, 0, 40, 120], [40, 0, 100, 120]],
                    "reading_order": [1, 0], "page_role": "story",
                }],
            }), encoding="utf-8")
            applied = _apply_reviewed_panel_corrections(
                [detection], raw, correction, root, 90,
            )
            self.assertEqual(detection.boxes, [[40, 0, 100, 120], [0, 0, 40, 120]])
            self.assertIn(page.name, applied)

    def test_chapter_export_writes_grounded_preview_and_mapping(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            chapter = root / "Manga" / "chapter_001"
            destination = root / "output" / "Manga" / "chapter_001"
            chapter.mkdir(parents=True)
            page = chapter / "page.jpg"
            Image.new("RGB", (300, 300), "white").save(page)

            def fake_detect(input_dir, model_path, output_dir, *, body_detections, **settings):
                body_detections[page] = [
                    Detection(np.array([170, 20, 260, 140]), .95, 0),
                ]
                return [PageDetection(page, [[0, 0, 300, 300]], [.9])]

            with patch("panel_extractor.runner.detect_panels", side_effect=fake_detect), \
                    patch("panel_extractor.runner.crop_panels", return_value=[]):
                extract_chapter(chapter, destination, Path("unused.onnx"), jpeg_quality=92)

            manifest = json.loads((destination / "panel_manifest.json").read_text(encoding="utf-8"))
            page_record = manifest["pages"][0]
            self.assertTrue(Path(page_record["grounded_preview"]).is_file())
            self.assertEqual(page_record["character_labels"][0]["label"], "C1")
            proposals = json.loads((destination / "character_detections.json").read_text(encoding="utf-8"))
            self.assertEqual(proposals[0]["characters"][0]["detection_id"],
                             page_record["character_labels"][0]["detection_id"])

    def test_character_proposals_get_deterministic_labels_and_dedupe(self):
        with tempfile.TemporaryDirectory() as temp:
            chapter = Path(temp) / "Manga" / "chapter_001"
            chapter.mkdir(parents=True)
            page = chapter / "page.png"
            Image.new("RGB", (300, 300), "white").save(page)
            detections = [PageDetection(page, [[0, 0, 300, 150], [0, 150, 300, 300]], [.9, .9])]
            manifest = build_manifest(detections)
            bodies = {page: [
                Detection(np.array([180, 20, 260, 140]), .90, 0),
                Detection(np.array([181, 20, 261, 140]), .80, 0),
                Detection(np.array([20, 170, 100, 290]), .95, 0),
            ]}
            first = serialise_proposals(bodies, manifest)
            second = serialise_proposals(bodies, manifest)
            characters = first[0]["characters"]
            self.assertEqual(first, second)
            self.assertEqual([row["label"] for row in characters], ["C1", "C2"])
            self.assertEqual([row["panel_number"] for row in characters], [1, 2])
            self.assertEqual(len({row["detection_id"] for row in characters}), 2)

    def test_character_labels_drop_low_scores_and_nested_duplicate_boxes(self):
        with tempfile.TemporaryDirectory() as temp:
            chapter = Path(temp) / "Manga" / "chapter_001"
            chapter.mkdir(parents=True)
            page = chapter / "page.png"
            Image.new("RGB", (300, 300), "white").save(page)
            manifest = build_manifest([
                PageDetection(page, [[0, 0, 300, 300]], [.9])
            ])
            bodies = {page: [
                Detection(np.array([10, 10, 120, 260]), .95, 0),
                Detection(np.array([12, 12, 118, 258]), .94, 0),
                Detection(np.array([20, 20, 110, 250]), .93, 0),
                Detection(np.array([105, 10, 220, 260]), .92, 0),
                Detection(np.array([230, 20, 290, 260]), .49, 0),
            ]}

            characters = serialise_proposals(bodies, manifest)[0]["characters"]

            self.assertEqual(len(characters), 2)
            self.assertEqual(sorted(row["score"] for row in characters), [.92, .95])
            self.assertTrue(all(row["panel_number"] == 1 for row in characters))

    def test_small_contained_character_is_not_merged_into_large_character(self):
        with tempfile.TemporaryDirectory() as temp:
            chapter = Path(temp) / "Manga" / "chapter_001"
            chapter.mkdir(parents=True)
            page = chapter / "page.png"
            Image.new("RGB", (300, 300), "white").save(page)
            manifest = build_manifest([
                PageDetection(page, [[0, 0, 300, 300]], [.9])
            ])
            bodies = {page: [
                Detection(np.array([5, 5, 260, 295]), .95, 0),
                Detection(np.array([20, 20, 70, 120]), .90, 0),
            ]}

            characters = serialise_proposals(bodies, manifest)[0]["characters"]

            self.assertEqual(len(characters), 2)

    def test_aligned_tile_fragment_is_removed_without_merging_overlapping_people(self):
        with tempfile.TemporaryDirectory() as temp:
            chapter = Path(temp) / "Manga" / "chapter_001"
            chapter.mkdir(parents=True)
            page = chapter / "page.png"
            Image.new("RGB", (400, 400), "white").save(page)
            manifest = build_manifest([
                PageDetection(page, [[0, 0, 400, 400]], [.9])
            ])
            bodies = {page: [
                Detection(np.array([100, 60, 300, 360]), .95, 0),
                # Same body, clipped on its right by tiled inference.
                Detection(np.array([101, 60, 245, 358]), .90, 0),
                # A genuinely separate overlapping person: only one edge aligns.
                Detection(np.array([220, 90, 380, 360]), .85, 0),
            ]}

            characters = serialise_proposals(bodies, manifest)[0]["characters"]

            self.assertEqual(len(characters), 2)
            self.assertEqual(sorted(row["score"] for row in characters), [.85, .95])

    def test_duplicate_selection_prefers_full_box_only_when_score_is_close(self):
        with tempfile.TemporaryDirectory() as temp:
            chapter = Path(temp) / "Manga" / "chapter_001"
            chapter.mkdir(parents=True)
            page = chapter / "page.png"
            Image.new("RGB", (400, 400), "white").save(page)
            manifest = build_manifest([
                PageDetection(page, [[0, 0, 400, 400]], [.9])
            ])
            fuller = [80, 40, 300, 360]
            bodies = {page: [
                Detection(np.array([100, 80, 280, 350]), .95, 0),
                Detection(np.array(fuller), .90, 0),
                # This very weak large box must be discarded by the threshold.
                Detection(np.array([20, 10, 350, 390]), .40, 0),
            ]}

            characters = serialise_proposals(bodies, manifest)[0]["characters"]

            self.assertEqual(len(characters), 1)
            self.assertEqual(characters[0]["box"], fuller)

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
