"""Writer-only regression cases for plot coverage and exact occurrence bindings."""
import copy
import json
import unittest

from story_writer.evidence import (build_ledger, draft_schema, incidental_pose, validate_draft,
                                   validate_review, outside_quotations, prompt_record)


def panel(panel_id="p1"):
    return {"panel_id": panel_id, "chapter": "chapter_001", "page": "1.jpg", "panel_number": 1,
            "description": "A student runs through a corridor.", "characters": [
                {"instance_id": "boy", "allowed_name": "Akira", "visual_description": "a young man", "entity_id": None},
                {"instance_id": "girl", "allowed_name": None, "visual_description": "a woman", "entity_id": None}],
            "accepted_text_facts": [], "accepted_actions": [], "accepted_transformations": []}


class StoryEvidenceTests(unittest.TestCase):
    def sentence(self, text, facts, characters=(), speaker=None):
        return {"text": text, "facts": list(facts), "characters": list(characters), "speaker": speaker}

    def test_game_exposition_cannot_disappear_behind_spell_quotes(self):
        source = panel()
        source["accepted_text_facts"] = [
            {"fact_id": "popularity", "kind": "narration", "text": "It became an overnight sensation online."},
            {"fact_id": "climax", "kind": "narration", "text": "The protagonist does not deliver the final blow to the evil god."},
            {"fact_id": "spell", "kind": "dialogue", "text": "Pyro Burgundy!", "speaker_instance_id": None}]
        ledger = build_ledger([source])
        draft = {"sentences": [self.sentence('“Pyro Burgundy!”', ["F3"])]}
        errors = validate_draft(draft, ledger)
        self.assertTrue(any("F1" in error and "F2" in error for error in errors), errors)

    def test_popularity_fact_survives_first_person_narrator_handling(self):
        source = panel()
        source["accepted_text_facts"] = [
            {"kind": "narration", "text": "His heroic deeds put him at the top of the popularity poll."},
            {"kind": "narration", "text": "I came to love him and read every guide."}]
        ledger = build_ledger([source])
        errors = validate_draft({"sentences": [self.sentence("Every guide gets read.", ["F2"])]}, ledger)
        self.assertIn("Missing required facts: F1", errors)

    def test_unknown_line_cannot_be_assigned_to_the_only_visible_character(self):
        source = panel()
        source["characters"] = source["characters"][:1]
        source["accepted_actions"] = [{"actor_instance_id": "boy", "text": "raises a dagger"}]
        source["accepted_text_facts"] = [{"kind": "dialogue", "text": "Then I'll take her.", "speaker_instance_id": None}]
        ledger = build_ledger([source])
        draft = {"sentences": [self.sentence("Akira declares that he will take her.", ["F1", "F2"], ["C1"], "C1")]}
        self.assertTrue(any("unknown speech" in error for error in validate_draft(draft, ledger)))

    def test_known_speaker_is_not_replaced_with_another_occurrence(self):
        source = panel()
        source["accepted_text_facts"] = [{"kind": "dialogue", "text": "Do you know who the heir is?", "speaker_instance_id": "boy"}]
        source["accepted_actions"] = [{"actor_instance_id": "girl", "text": "opens the door"}]
        ledger = build_ledger([source])
        draft = {"sentences": [self.sentence("She asks who the heir is.", ["F1", "F2"], ["C2"], "C2")]}
        self.assertTrue(any("known speaker" in error for error in validate_draft(draft, ledger)))

    def test_narration_does_not_gain_a_visible_actor(self):
        source = panel()
        source["accepted_text_facts"] = [{"kind": "narration", "text": "Arthur Klarvein"}]
        source["accepted_actions"] = [{"actor_instance_id": "boy", "text": "opens the door"}]
        ledger = build_ledger([source])
        self.assertTrue(ledger["facts"][0]["required"])
        errors = validate_draft({"sentences": [self.sentence("Arthur speaks.", ["F1"], ["C1"], "C1")]}, ledger)
        self.assertTrue(any("visible participant" in error for error in errors))

    def test_optional_pose_does_not_hide_meaningful_quiet_action(self):
        self.assertTrue(incidental_pose("standing nearby"))
        self.assertTrue(incidental_pose("touching his hair"))
        for text in ("standing guard", "standing between her and the sword", "looking for an escape",
                     "kneeling to surrender", "sitting on the wounded guard to restrain him"):
            with self.subTest(text=text):
                self.assertFalse(incidental_pose(text))

    def test_position_can_be_omitted_without_losing_intervention(self):
        source = panel()
        source["accepted_actions"] = [dict(actor_instance_id="girl", target_instance_id="boy", text=text)
                                      for text in ("stands beside him", "stands beside him to block the exit")]
        ledger = build_ledger([source])
        self.assertEqual([f["required"] for f in ledger["facts"]], [False, True])
        draft = {"sentences": [self.sentence("She blocks his exit.", ["F2"], ["C1", "C2"])]}
        self.assertEqual(validate_draft(draft, ledger), [])

    def test_ledger_preserves_original_evidence_and_source_ids(self):
        source = panel()
        source["accepted_text_facts"] = [{"kind": "dialogue", "fact_id": "text_123", "text": "Honestly,\nyou...", "speaker_instance_id": None}]
        before = copy.deepcopy(source)
        first = build_ledger([source])
        self.assertEqual(first, build_ledger([source]))
        self.assertEqual(source, before)
        self.assertEqual(first["facts"][0]["text"], "Honestly,\nyou...")
        self.assertEqual(first["facts"][0]["source_id"], "text_123")
        source["accepted_text_facts"][0]["text"] = "Changed evidence."
        self.assertNotEqual(first["fingerprint"], build_ledger([source])["fingerprint"])

    def test_internal_identity_ids_do_not_leak_name_permissions_into_prompts(self):
        source = panel()
        source["characters"][0].update(entity_id="entity_UnpermittedCanonicalName", instance_id="private_occurrence_id")
        source["accepted_actions"] = [{"actor_instance_id": "private_occurrence_id", "event_id": "private_event_id", "text": "opens the door"}]
        ledger = build_ledger([source])
        compact = json.dumps(prompt_record(ledger))
        for secret_id in ("UnpermittedCanonicalName", "private_occurrence_id", "private_event_id", ledger["fingerprint"]):
            self.assertNotIn(secret_id, compact)
        self.assertIn("Akira", compact)
        self.assertIn("opens the door", compact)

    def test_fact_ids_do_not_depend_on_request_local_character_numbers(self):
        source = panel()
        source["accepted_actions"] = [{"actor_instance_id": "boy", "event_id": "run", "text": "runs away"}]
        before = build_ledger([source])["facts"][0]
        source["accepted_actions"].insert(0, {"actor_instance_id": "girl", "event_id": "open", "text": "opens the door"})
        after = build_ledger([source])["facts"][1]
        self.assertNotEqual(before["actor"], after["actor"])
        self.assertEqual(before["id"], after["id"])

    def test_unknown_thought_and_publication_are_not_citable_story_facts(self):
        source = panel()
        source["accepted_text_facts"] = [
            {"kind": "thought", "text": "I know the secret.", "speaker_instance_id": None},
            {"kind": "publication", "text": "Subscribe now!"}]
        ledger = build_ledger([source])
        errors = validate_draft({"sentences": [self.sentence("He knows the secret.", ["F1", "F2"])]}, ledger)
        self.assertTrue(any("unusable evidence" in error for error in errors))

    def test_stale_and_future_event_endpoints_fail_closed(self):
        first, second = panel(), panel("p2")
        second["characters"] = [{"instance_id": "future", "visual_description": "a girl"}]
        first["accepted_actions"] = [{"actor_instance_id": "future", "text": "grabs a sword"}]
        with self.assertRaisesRegex(ValueError, "future occurrence"):
            build_ledger([first], [first, second])
        first["accepted_actions"][0]["actor_instance_id"] = "missing"
        with self.assertRaisesRegex(ValueError, "Missing grounded occurrence"):
            build_ledger([first])

    def test_transformation_keeps_an_explicit_earlier_source_endpoint(self):
        first, second = panel(), panel("p2")
        first["characters"] = first["characters"][:1]
        second["characters"] = [{"instance_id": "changed", "visual_description": "a creature"}]
        second["accepted_transformations"] = [{"event_id": "event_1", "source_instance_id": "boy",
            "target_instance_id": "changed", "source_panel_id": "p1", "target_panel_id": "p2",
            "evidence": "The boy changes into a creature."}]
        ledger = build_ledger([second], [first, second])
        fact = ledger["facts"][0]
        self.assertEqual(fact["kind"], "transformation")
        self.assertEqual((fact["source"], fact["target"]), ("C1", "C2"))
        self.assertEqual(ledger["characters"][0]["panel_id"], "p1")

    def test_forged_coverage_can_pass_bindings_but_is_not_semantic_proof(self):
        source = panel()
        source["accepted_text_facts"] = [{"kind": "narration", "text": "The vow lasts for life."}]
        ledger = build_ledger([source])
        draft = {"sentences": [self.sentence("The vow expires tomorrow.", ["F1"])]}
        self.assertEqual(validate_draft(draft, ledger), [])
        # The separate reviewer must reject this; declarations alone are intentionally not called proof.

    def test_malformed_bindings_and_extra_output_fields_are_rejected(self):
        source = panel()
        source["accepted_text_facts"] = [{"kind": "narration", "text": "A war begins."}]
        ledger = build_ledger([source])
        for change in ({"facts": ["F1", "F1"]}, {"facts": ["missing"]}, {"facts": [{}]},
                       {"characters": [{}]}, {"speaker": {}}, {"invented": True}):
            sentence = dict(self.sentence("A war begins.", ["F1"]), **change)
            with self.subTest(change=change):
                self.assertTrue(validate_draft({"sentences": [sentence]}, ledger))
        self.assertEqual(draft_schema(ledger)["properties"]["sentences"]["items"]["anyOf"][0]["additionalProperties"], False)

    def test_schema_binds_background_unknown_speech_and_known_actor_separately(self):
        source = panel()
        source["accepted_text_facts"] = [
            {"kind": "narration", "text": "A war begins."},
            {"kind": "dialogue", "text": "Stop!", "speaker_instance_id": None},
            {"kind": "dialogue", "text": "Run!", "speaker_instance_id": "boy"}]
        source["accepted_actions"] = [{"actor_instance_id": "boy", "text": "runs away"}]
        variants = draft_schema(build_ledger([source]))["properties"]["sentences"]["items"]["anyOf"]
        properties = [v["properties"] for v in variants]
        self.assertEqual(len(properties), 4)
        self.assertEqual(properties[0]["characters"]["maxItems"], 0)
        self.assertEqual(properties[1]["speaker"]["enum"], [None])
        self.assertEqual(properties[1]["characters"]["maxItems"], 0)
        self.assertEqual(properties[2]["speaker"]["enum"], ["C1"])
        self.assertEqual(properties[2]["characters"]["items"]["enum"], ["C1"])
        self.assertEqual(properties[3]["speaker"]["enum"], [None])
        self.assertEqual(properties[3]["characters"]["items"]["enum"], ["C1"])

    def test_title_case_is_not_evidence_that_a_fact_is_just_a_name(self):
        source = panel()
        source["accepted_text_facts"] = [{"kind": "narration", "text": "The World Ends"}]
        self.assertTrue(build_ledger([source])["facts"][0]["required"])

    def test_narration_cannot_reverse_source_panel_order(self):
        first, second = panel(), panel("p2")
        second["characters"] = []
        first["accepted_text_facts"] = [{"kind": "narration", "text": "The war begins."}]
        second["accepted_text_facts"] = [{"kind": "narration", "text": "The city falls."}]
        ledger = build_ledger([first, second])
        draft = {"sentences": [self.sentence("The city falls.", ["F2"]), self.sentence("The war begins.", ["F1"])]}
        self.assertTrue(any("backward" in error for error in validate_draft(draft, ledger)))

    def test_narration_does_not_print_internal_character_keys(self):
        source = panel()
        source["accepted_actions"] = [{"actor_instance_id": "boy", "text": "runs away"}]
        ledger = build_ledger([source])
        draft = {"sentences": [self.sentence("C1 runs away.", ["F1"], ["C1"])]}
        self.assertTrue(any("prints metadata key" in error for error in validate_draft(draft, ledger)))

    def test_quotes_and_contractions_preserve_perspective_checks(self):
        for text in ('He says, “I can’t.”', "He says, 'I can't.'", 'He can’t escape.', '“I am ready.” “We go now.”',
                     "The boys' promise holds.", "The boys’ promise holds.", "He calls it 'the boys'."):
            outside, errors = outside_quotations(text)
            with self.subTest(text=text):
                self.assertFalse(errors)
                self.assertNotIn("I ", outside)
        for text in ('He says, “I go.', 'He says, I go.”'):
            self.assertTrue(outside_quotations(text)[1])

    def test_quoted_source_name_does_not_assign_a_body_identity(self):
        from story_writer.writer import validate_story_response
        for name in ("Akira", "Lucia"):
            with self.subTest(name=name):
                source = panel()
                source["accepted_text_facts"] = [{"kind": "dialogue", "text": f"{name}, wait!", "speaker_instance_id": None}]
                row = {"primary_panel_id": "p1", "supporting_panel_ids": ["p1"], "text": f'“{name}, wait!”',
                       "used_instance_ids": [], "name_uses": []}
                _, errors = validate_story_response(json.dumps({"segments": [row]}), [source], known_names={"Lucia"})
                self.assertEqual(errors, [])
                row["text"] = f"{name} orders him to wait."
                _, errors = validate_story_response(json.dumps({"segments": [row]}), [source], known_names={"Lucia"})
                self.assertTrue(any("does not ground the name" in error for error in errors), errors)

    def test_quotation_marks_cannot_launder_an_invented_canonical_name(self):
        from story_writer.writer import validate_story_response
        source = panel()
        source["accepted_text_facts"] = [{"kind": "dialogue", "text": "I am ready.", "speaker_instance_id": "boy"}]
        row = {"primary_panel_id": "p1", "supporting_panel_ids": ["p1"], "text": 'He says, “I am Lucia.”',
               "used_instance_ids": ["boy"], "name_uses": []}
        for text in ('He says, “I am Lucia.”', 'He says, "I am\nLucia."'):
            row["text"] = text
            _, errors = validate_story_response(json.dumps({"segments": [row]}), [source], known_names={"Lucia"})
            self.assertTrue(any("not supported by exact source speech" in error for error in errors), errors)

    def test_review_must_reject_distorted_meaning_even_with_complete_citations(self):
        source = panel()
        source["accepted_text_facts"] = [{"kind": "narration", "text": "The vow lasts for life."}]
        ledger = build_ledger([source])
        draft = {"sentences": [self.sentence("The vow expires tomorrow.", ["F1"])]}
        review = {"fact_checks": [{"fact": "F1", "sentences": [1], "verdict": "distorted", "reason": "Lifetime changed to tomorrow."}],
                  "sentence_checks": [{"sentence": 1, "verdict": "unsupported", "reason": "Tomorrow is invented."}],
                  "paragraph_issues": []}
        self.assertEqual(len(validate_review(review, draft, ledger)), 2)
        for change in ({"fact_checks": []}, {"sentence_checks": []}, {"paragraph_issues": "ok"}):
            self.assertTrue(validate_review(dict(review, **change), draft, ledger))

    def test_review_cannot_waive_required_evidence_or_forge_sentence_indices(self):
        source = panel()
        source["accepted_text_facts"] = [{"kind": "narration", "text": "A war begins."}]
        ledger = build_ledger([source])
        draft = {"sentences": []}
        review = {"fact_checks": [{"fact": "F1", "sentences": [], "verdict": "omitted_incidental", "reason": "Too long."}],
                  "sentence_checks": [], "paragraph_issues": []}
        self.assertTrue(validate_review(review, draft, ledger))
        draft["sentences"] = [self.sentence("A war begins.", ["F1"])]
        review["sentence_checks"] = [{"sentence": 1, "verdict": "supported", "reason": ""}]
        review["fact_checks"][0]["verdict"] = "preserved"
        self.assertTrue(validate_review(review, draft, ledger))
        review["fact_checks"][0]["sentences"] = [1]
        self.assertEqual(validate_review(review, draft, ledger), [])
        review["sentence_checks"][0]["sentence"] = True
        self.assertTrue(validate_review(review, draft, ledger))
