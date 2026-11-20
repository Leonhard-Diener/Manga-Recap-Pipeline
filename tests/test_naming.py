"""CPU-only checks for deterministic gender-neutral fallback naming."""

import copy
import unittest

from story_entities.naming import assign_group_display_names, load_fallback_names


class FallbackNamingTests(unittest.TestCase):
    def test_pool_contains_200_unique_common_surnames(self):
        names = load_fallback_names()
        self.assertEqual(len(names), 200)
        self.assertEqual(len({name.casefold() for name in names}), 200)

    def test_only_retained_groups_receive_stable_unique_names(self):
        source = {
            "visual_groups": [
                {"group_id": "group_b", "instance_ids": ["b"]},
                {"group_id": "group_a", "instance_ids": ["a"]},
            ],
            "characters": [
                {"instance_id": "a", "visual_group_id": "group_a"},
                {"instance_id": "b", "visual_group_id": "group_b"},
                {"instance_id": "rare", "visual_group_id": None,
                 "display_name": "stale", "display_name_source": "stale"},
            ],
        }
        first = copy.deepcopy(source)
        second = copy.deepcopy(source)
        second["visual_groups"].reverse()

        assign_group_display_names(first, "Manga")
        assign_group_display_names(second, "Manga")

        first_names = {row["group_id"]: row["display_name"]
                       for row in first["visual_groups"]}
        second_names = {row["group_id"]: row["display_name"]
                        for row in second["visual_groups"]}
        self.assertEqual(first_names, second_names)
        self.assertEqual(len(set(first_names.values())), 2)
        self.assertTrue(all(row["display_name_source"] == "fallback_pool"
                            for row in first["visual_groups"]))
        self.assertTrue(all(row["display_name_scope"] == "visual_group"
                            for row in first["visual_groups"]))
        self.assertNotIn("display_name", first["characters"][2])
        self.assertFalse(first["display_naming_policy"]["ungrouped_characters_named"])
        self.assertEqual(
            first["display_naming_policy"]["pool_kind"],
            "common_japanese_surnames",
        )

    def test_accepted_entity_forms_share_one_name(self):
        source = {
            "visual_groups": [
                {"group_id": "human_form", "instance_ids": ["human"]},
                {"group_id": "dragon_form", "instance_ids": ["dragon"]},
                {"group_id": "unrelated", "instance_ids": ["other"]},
            ],
            "characters": [
                {"instance_id": "human", "visual_group_id": "human_form",
                 "entity_id": "hero"},
                {"instance_id": "dragon", "visual_group_id": "dragon_form",
                 "entity_id": "hero"},
                {"instance_id": "other", "visual_group_id": "unrelated",
                 "entity_id": None},
            ],
            "identities": [{"entity_id": "hero"}],
        }

        assign_group_display_names(source, "Manga")

        by_group = {
            row["group_id"]: row
            for row in source["visual_groups"]
        }
        self.assertEqual(
            by_group["human_form"]["display_name"],
            by_group["dragon_form"]["display_name"],
        )
        self.assertNotEqual(
            by_group["human_form"]["display_name"],
            by_group["unrelated"]["display_name"],
        )
        self.assertEqual(by_group["human_form"]["display_name_scope"], "entity")
        self.assertEqual(by_group["dragon_form"]["display_name_scope"], "entity")
        self.assertEqual(
            source["identities"][0]["display_name"],
            by_group["human_form"]["display_name"],
        )
        self.assertTrue(
            source["display_naming_policy"]["accepted_entity_forms_share_name"]
        )

    def test_unconnected_forms_keep_separate_names(self):
        source = {
            "visual_groups": [
                {"group_id": "form_a", "instance_ids": ["a"]},
                {"group_id": "form_b", "instance_ids": ["b"]},
            ],
            "characters": [
                {"instance_id": "a", "visual_group_id": "form_a",
                 "entity_id": None},
                {"instance_id": "b", "visual_group_id": "form_b",
                 "entity_id": None},
            ],
        }

        assign_group_display_names(source, "Manga")

        self.assertNotEqual(
            source["visual_groups"][0]["display_name"],
            source["visual_groups"][1]["display_name"],
        )

    def test_one_accepted_member_cannot_give_two_unresolved_groups_the_same_alias(self):
        source = {
            "visual_groups": [{"group_id": g, "instance_ids": [g + "1", g + "2"]}
                              for g in ("girl", "boy")],
            "characters": [{"instance_id": g + str(i), "visual_group_id": g,
                            "entity_id": "shared" if i == 1 else None}
                           for g in ("girl", "boy") for i in (1, 2)],
        }
        assign_group_display_names(source, "Manga")
        self.assertNotEqual(source["visual_groups"][0]["display_name"],
                            source["visual_groups"][1]["display_name"])

    def test_aliases_do_not_require_canonical_names_or_name_reveals(self):
        from story_writer.writer import _character_fact

        source = {
            "visual_groups": [{"group_id": "group", "instance_ids": ["before", "after"]}],
            "identities": [{"entity_id": "lucia", "name": "Lucia",
                            "name_record": {"source_instance_id": "after"}}],
            "characters": [
                {"instance_id": instance, "visual_group_id": "group", "entity_id": "lucia",
                 "identity_status": "reviewed", "naming_allowed": allowed,
                 "name": "Lucia" if allowed else None, "stable_description": "a horned girl"}
                for instance, allowed in (("before", False), ("after", True))
            ],
        }
        assign_group_display_names(source, "Manga")
        before, after = source["characters"]
        self.assertTrue(before["display_name_allowed"])
        self.assertEqual(_character_fact(before)["allowed_name"], before["display_name"])
        self.assertEqual(_character_fact(after)["allowed_name"], before["display_name"])
        self.assertNotEqual(_character_fact(after)["allowed_name"], "Lucia")

    def test_unretained_occurrences_stay_unnamed_even_with_an_entity_or_stale_alias(self):
        from story_writer.writer import _character_fact

        source = {
            "visual_groups": [{"group_id": "retained", "instance_ids": ["person"]}],
            "characters": [
                {"instance_id": "person", "visual_group_id": "retained", "identity_status": "unknown"},
                {"instance_id": "solo", "visual_group_id": None, "entity_id": "hero",
                 "display_name": "Stale", "display_name_allowed": True},
                {"instance_id": "removed", "visual_group_id": "filtered_out",
                 "display_name": "Stale", "display_name_allowed": True},
            ],
        }
        assign_group_display_names(source, "Manga")
        person, solo, removed = source["characters"]
        self.assertEqual(_character_fact(person)["allowed_name"], person["display_name"])
        self.assertEqual(person["identity_status"], "unknown")
        for row in (solo, removed):
            self.assertNotIn("display_name", row)
            self.assertIsNone(_character_fact(row)["allowed_name"])


if __name__ == "__main__":
    unittest.main()
