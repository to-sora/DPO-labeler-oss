from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from checkpoint_registry import (
    CheckpointRegistryError,
    get_checkpoint_registry,
    load_checkpoint_registry,
)
from generators.generate_dpo_workflow_matrix import ALL_CHECKPOINTS
from generators.sv_checkpoint_pools import SELF_TRAINED_CHECKPOINTS


EXPECTED_FAMILIES = {
    "sdxl_albedobaseXL_v31Large.safetensors": "sdxl_anime_base",
    "sdxl_animagineXL40_v4Opt.safetensors": "sdxl_anime_base",
    "sdxl_animagineXLV31_v31.safetensors": "sdxl_anime_base",
    "sdxl_AnythingXL_xl.safetensors": "sdxl_anime_base",
    "sdxl_astolfokarmixXL_256cBased.safetensors": "sdxl_anime_base",
    "sdxl_bismuthIllustrious_v60.safetensors": "illustration",
    "sdxl_bluePencilXL_v700.safetensors": "sdxl_anime_base",
    "sdxl_boleromixIllustrious_v601.safetensors": "illustration",
    "sdxl_counterfeitxl_v25.safetensors": "sdxl_anime_base",
    "sdxl_cyberrealisticPony_v150__2.safetensors": "pony",
    "sdxl_cyberrealisticPony_v150.safetensors": "pony",
    "sdxl_divingIllustriousReal_v70VAE.safetensors": "illustration",
    "sdxl_epicrealismXL_pureFix.safetensors": "realistic",
    "sdxl_hassakuXLIllustrious_v32.safetensors": "illustration",
    "sdxl_hikariNoobVPred_124.safetensors": "sdxl_anime_base",
    "sdxl_hosekiLustrousmix_illustNoobaiEPS11V2.safetensors": "illustration",
    "sdxl_illustrij_v20.safetensors": "illustration",
    "sdxl_iniverseMixSFWNSFW_ponyRealGuofengV50A__2.safetensors": "pony",
    "sdxl_iniverseMixSFWNSFW_ponyRealGuofengV50A.safetensors": "pony",
    "sdxl_iniverseMixSFWNSFW_realXLV1.safetensors": "realistic",
    "sdxl_JANKUTrainedNoobaiRouwei_v60.safetensors": "sdxl_anime_base",
    "sdxl_jibMixRealisticXL_v180SkinSupreme.safetensors": "realistic",
    "sdxl_juggernautXL_ragnarokBy.safetensors": "realistic",
    "sdxl_meichidarkmixReload_meichidarkanimv2Lust.safetensors": "sdxl_anime_base",
    "sdxl_mistoonAnime_v10Illustrious.safetensors": "illustration",
    "sdxl_mritualIllustrious_v201.safetensors": "illustration",
    "sdxl_novaAnimeXL_ilV140.safetensors": "sdxl_anime_base",
    "sdxl_novaAnimeXL_ilV150.safetensors": "sdxl_anime_base",
    "sdxl_novaUnrealXL_v100.safetensors": "realistic",
    "sdxl_ntrMIXIllustriousXL_xiii.safetensors": "illustration",
    "sdxl_obsessionIllustrious_vPredV11.safetensors": "illustration",
    "sdxl_perfectdeliberate_v60.safetensors": "realistic",
    "sdxl_pieModels_applePieV2.safetensors": "realistic",
    "sdxl_ponyDiffusionV6XL_v6StartWithThisOne.safetensors": "pony",
    "sdxl_pornmaster_proSDXLV8.safetensors": "sdxl_anime_base",
    "sdxl_prefectIllustriousXL_v60.safetensors": "illustration",
    "sdxl_prefectiousXLNSFW_v10.safetensors": "sdxl_anime_base",
    "sdxl_prefectPonyXL_v6.safetensors": "pony",
    "sdxl_realismByStableYogi_v5XLFP16.safetensors": "realistic",
    "sdxl_realPony_illustriousPony.safetensors": "pony",
    "sdxl_sdxl10ArienmixxlAsian_v45Pruned.safetensors": "sdxl_anime_base",
    "sdxl_sdxlNijiSeven_sdxlNijiSeven.safetensors": "sdxl_anime_base",
    "sdxl_steincustom_V13__2.safetensors": "sdxl_anime_base",
    "sdxl_steincustom_V13.safetensors": "sdxl_anime_base",
    "sdxl_uncannyValley_VPredV1.safetensors": "realistic",
    "sdxl_waiIllustriousSDXL_v160.safetensors": "illustration",
    "sdxl_waiREALCN_v150.safetensors": "sdxl_anime_base",
    "sdxl_waiREALMIX_v11.safetensors": "sdxl_anime_base",
    "sdxl_xxmix9realisticsdxl_v10.safetensors": "realistic",
    "kohyass/keep/Animagine_XL_4.0_base/test3_ver4.ckpt": "sdxl_anime_base",
    "kohyass/keep/Animagine_XL_4.0_base/test4_ver4-000008.ckpt": "sdxl_anime_base",
    "kohyass/keep/Animagine_XL_4.0_base/test4_ver4.ckpt": "sdxl_anime_base",
    "kohyass/keep/illlustion_base/path_2/test1_ver4-000014.ckpt": "illustration",
    "kohyass/keep/illlustion_base/path_2/test1_ver4.ckpt": "illustration",
    "kohyass/keep/illlustion_base/path_4/test1_ver4-000014.ckpt": "illustration",
    "kohyass/keep/illlustion_base/path_4/test1_ver4.ckpt": "illustration",
}


class CheckpointRegistryTests(unittest.TestCase):
    def test_registry_covers_every_parent_checkpoint(self) -> None:
        inventory = set(ALL_CHECKPOINTS) | set(SELF_TRAINED_CHECKPOINTS)
        self.assertEqual(inventory, set(EXPECTED_FAMILIES))

        registry = get_checkpoint_registry()
        for checkpoint, expected_family in EXPECTED_FAMILIES.items():
            with self.subTest(checkpoint=checkpoint):
                resolved = registry.resolve(checkpoint)
                self.assertIsNotNone(resolved.model_id)
                self.assertEqual(resolved.family, expected_family)
                self.assertNotEqual(resolved.family_source, "default")

    def test_keyword_classifier_exceeds_ninety_percent_accuracy(self) -> None:
        registry = get_checkpoint_registry()
        correct = 0
        for checkpoint, expected_family in EXPECTED_FAMILIES.items():
            classified = registry.classify_family_by_keywords(checkpoint)
            if classified is not None and classified.family == expected_family:
                correct += 1
        accuracy = correct / len(EXPECTED_FAMILIES)
        self.assertGreaterEqual(accuracy, 0.90, f"keyword accuracy was {accuracy:.1%}")

    def test_private_and_public_metadata_are_distinct(self) -> None:
        registry = get_checkpoint_registry()
        for checkpoint in ALL_CHECKPOINTS:
            resolved = registry.resolve(checkpoint)
            self.assertEqual(resolved.visibility, "public")
            self.assertTrue(resolved.publish)
        for checkpoint in SELF_TRAINED_CHECKPOINTS:
            resolved = registry.resolve(checkpoint)
            self.assertEqual(resolved.visibility, "private")
            self.assertFalse(resolved.publish)

    def test_aliases_share_one_registry_identity(self) -> None:
        registry = get_checkpoint_registry()
        first = registry.resolve("sdxl_steincustom_V13.safetensors")
        second = registry.resolve("sdxl_steincustom_V13__2.safetensors")
        self.assertEqual(first.model_id, second.model_id)
        self.assertEqual(first.family, second.family)

    def test_entry_without_family_uses_keyword_classifier(self) -> None:
        registry = get_checkpoint_registry()
        resolved = registry.resolve("sdxl_novaAnimeXL_ilV150.safetensors")
        self.assertEqual(resolved.family, "sdxl_anime_base")
        self.assertEqual(resolved.family_source, "keyword")
        self.assertEqual(resolved.matched_keyword, "anime")

    def test_unknown_checkpoint_preserves_compatibility_default(self) -> None:
        registry = get_checkpoint_registry()
        resolved = registry.resolve("renamed_private_model.safetensors")
        self.assertEqual(resolved.family, "sdxl_anime_base")
        self.assertEqual(resolved.family_source, "default")
        self.assertEqual(resolved.visibility, "unknown")
        self.assertFalse(resolved.publish)

    def test_overlay_can_register_a_private_alias_without_family(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            overlay = Path(tmpdir) / "overlay.yaml"
            overlay.write_text(
                "\n".join(
                    [
                        "version: 1",
                        "models:",
                        "  local-pony-test:",
                        "    visibility: private",
                        "    publish: false",
                        "    aliases:",
                        "      - private/renamed_pony_test.safetensors",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            registry = load_checkpoint_registry(overlay_paths=[overlay])
            resolved = registry.resolve("private/renamed_pony_test.safetensors")
            self.assertEqual(resolved.family, "pony")
            self.assertEqual(resolved.family_source, "keyword")
            self.assertEqual(resolved.visibility, "private")

    def test_invalid_family_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "invalid.yaml"
            registry_path.write_text(
                "\n".join(
                    [
                        "version: 1",
                        "default_family: invalid",
                        "keyword_rules:",
                        "  - family: pony",
                        "    keywords: [pony]",
                        "models:",
                        "  model:",
                        "    aliases: [model.safetensors]",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(CheckpointRegistryError):
                load_checkpoint_registry(registry_path)


if __name__ == "__main__":
    unittest.main()
