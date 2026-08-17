from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from tests._paths import REPO_ROOT


CHARACTER_PACK_DIR = REPO_ROOT / "template" / "wildcard" / "research_v6" / "characters"
EXPECTED_PACKS = {
    "fgo": "fgo.txt",
    "sao": "sao.txt",
    "ggo": "ggo.txt",
    "zzz": "zzz.txt",
    "genshin_impact": "genshin_impact.txt",
    "blue_archive": "blue_archive.txt",
    "nikke": "nikke.txt",
    "arknights": "arknights.txt",
    "honkai_star_rail": "honkai_star_rail.txt",
    "azur_lane": "azur_lane.txt",
    "punishing_gray_raven": "punishing_gray_raven.txt",
}


class V6CharacterPackTests(unittest.TestCase):
    def test_manifest_matches_expected_character_pack_set(self) -> None:
        manifest = yaml.safe_load((CHARACTER_PACK_DIR / "manifest.yaml").read_text(encoding="utf-8"))
        manifest_pairs = {row["game_id"]: row["file"] for row in manifest["packs"]}
        discovered_files = sorted(path.name for path in CHARACTER_PACK_DIR.glob("*.txt"))

        self.assertEqual(manifest_pairs, EXPECTED_PACKS)
        self.assertEqual(discovered_files, sorted(EXPECTED_PACKS.values()))
        self.assertEqual(manifest["format"], "neutral_extended_identity")
        self.assertTrue(manifest["wired_into_v6_templates"])

    def test_character_pack_files_are_non_empty_and_deduplicated(self) -> None:
        for path in sorted(CHARACTER_PACK_DIR.glob("*.txt")):
            with self.subTest(path=path.name):
                lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
                self.assertTrue(lines)
                self.assertGreaterEqual(len(lines), 70)
                self.assertEqual(len(lines), len(set(lines)))
                self.assertTrue(all(", " in line for line in lines))


if __name__ == "__main__":
    unittest.main()
