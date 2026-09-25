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


if __name__ == "__main__":
    unittest.main()
