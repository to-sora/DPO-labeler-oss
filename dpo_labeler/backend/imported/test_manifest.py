import yaml

from .test_support import ImportCase


class ManifestTests(ImportCase):
    def test_external_images_prompt_file_and_inline_fallback(self) -> None:
        manifest = self.manifest()
        (self.images / "prompt.txt").write_text("file prompt 貓", encoding="utf-8")
        manifest["image"][0]["prompt-path"] = "prompt.txt"
        manifest["image"][1]["prompt-path"] = "missing.txt"
        task = self.service.import_yaml(yaml.safe_dump(manifest))
        self.assertEqual(task["images"][0]["prompt"], "file prompt 貓")
        self.assertEqual(task["images"][1]["prompt"], "prompt 1")
        self.assertEqual(self.service.image(task["task_id"], 0), self.images / "0.png")
        self.assertEqual(self.service.catalog()["tasks"][0]["character_name"], "Character 貓")

    def test_bad_imports_leave_no_cache(self) -> None:
        base = self.manifest()
        cases = [dict(base, image_dir="relative"), dict(base, image=[]),
                 dict(base, image=[base["image"][0]] * 2),
                 dict(base, image=[{"image-path": "missing.png"}]),
                 dict(base, **{"dim-name": ["same", "same"]})]
        for manifest in cases:
            with self.subTest(manifest=manifest), self.assertRaises(ValueError):
                self.service.import_yaml(yaml.safe_dump(manifest))
        self.assertEqual(self.service.catalog()["tasks"], [])

    def test_each_import_is_independent_and_one_directory_is_removable(self) -> None:
        import shutil
        source = yaml.safe_dump(self.manifest())
        first = self.service.import_yaml(source)
        second = self.service.import_yaml(source)
        self.assertNotEqual(first["task_id"], second["task_id"])
        self.service.submit(first["task_id"], self.vote(first))
        self.assertEqual(self.service.get(second["task_id"])["comparisons"], 0)
        shutil.rmtree(first["output_dir"])
        self.assertEqual(len(self.service.catalog()["tasks"]), 1)
        self.assertTrue((self.images / "0.png").is_file())

    def test_single_image_is_immediately_certified(self) -> None:
        task = self.create(1)
        self.assertTrue(task["complete"])
        self.assertIsNone(task["pair"])
