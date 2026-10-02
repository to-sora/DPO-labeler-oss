import yaml

from .service import ImportedTasks
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
        self.assertEqual(task["prompt_dir"], str(self.images))
        self.assertEqual(self.service.image(task["task_id"], 0), self.images / "0.png")
        self.assertEqual(self.service.catalog()["tasks"][0]["character_name"], "Character 貓")

    def test_separate_prompt_directory_keeps_file_and_inline_behavior(self) -> None:
        manifest = self.manifest()
        prompts = self.root / "prompts"
        prompts.mkdir()
        (prompts / "prompt.txt").write_text("separate prompt 貓", encoding="utf-8")
        (prompts / "link.txt").symlink_to(prompts / "prompt.txt")
        self.service = ImportedTasks(self.root / "state", [self.images, prompts])
        manifest["prompt_dir"] = str(prompts)
        for entry, path in zip(manifest["image"],
                               ["prompt.txt", str(prompts / "prompt.txt"), "link.txt", "missing.txt"]):
            entry["prompt-path"] = path
        task = self.service.import_yaml(yaml.safe_dump(manifest))
        self.assertEqual(task["prompt_dir"], str(prompts))
        self.assertEqual([image["prompt"] for image in task["images"][:4]],
                         ["separate prompt 貓"] * 3 + ["prompt 3"])
        self.assertEqual(task["images"][4]["prompt"], "prompt 4")

    def test_prompt_paths_cannot_escape_default_or_explicit_directory(self) -> None:
        manifest = self.manifest()
        prompts = self.images / "prompts"
        prompts.mkdir()
        for prompt_root in (self.images, prompts):
            secret = prompt_root.parent / "outside-secret.txt"
            secret.write_text("private fixture", encoding="utf-8")
            (prompt_root / "secret-link.txt").symlink_to(secret)
            (prompt_root / "outside").symlink_to(prompt_root.parent, target_is_directory=True)
            if prompt_root == prompts:
                manifest["prompt_dir"] = str(prompts)
            for path in ("../outside-secret.txt", str(secret), "secret-link.txt",
                         "outside/outside-secret.txt", "../missing-secret.txt"):
                manifest["image"][0]["prompt-path"] = path
                with self.subTest(root=prompt_root, path=path), self.assertRaisesRegex(
                        ValueError, "prompt-path must stay inside prompt_dir"):
                    self.service.import_yaml(yaml.safe_dump(manifest))
        self.assertEqual(self.service.catalog()["tasks"], [])

    def test_import_directories_must_stay_inside_configured_roots(self) -> None:
        manifest = self.manifest(1)
        outside = self.root / "external-sibling"
        outside.mkdir()
        (outside / "0.png").write_bytes((self.images / "0.png").read_bytes())
        link = self.images / "outside"
        link.symlink_to(outside, target_is_directory=True)
        for field in ("image_dir", "prompt_dir"):
            for directory in (outside, link):
                with self.subTest(field=field, directory=directory), self.assertRaisesRegex(
                        ValueError, f"{field} must stay inside a configured image root"):
                    self.service.import_yaml(yaml.safe_dump({**manifest, field: str(directory)}))
        self.assertEqual(self.service.catalog()["tasks"], [])
        self.service = ImportedTasks(self.root / "state", [self.images, outside])
        task = self.service.import_yaml(yaml.safe_dump({**manifest, "image_dir": str(outside)}))
        self.assertEqual(self.service.image(task["task_id"], 0), outside / "0.png")

    def test_empty_allowlist_rejects_imports(self) -> None:
        self.service = ImportedTasks(self.root / "state", [])
        with self.assertRaisesRegex(ValueError, "configured image root"):
            self.create()
        self.assertEqual(self.service.catalog()["tasks"], [])

    def test_bad_imports_leave_no_cache(self) -> None:
        base = self.manifest()
        cases = [dict(base, image_dir="relative"), dict(base, image=[]),
                 dict(base, image=[base["image"][0]] * 2),
                 dict(base, image=[{"image-path": "missing.png"}]),
                 dict(base, **{"dim-name": ["same", "same"]}),
                 dict(base, prompt_dir="relative"),
                 dict(base, prompt_dir=str(self.images / "missing")),
                 dict(base, prompt_dir=str(self.images / "0.png"))]
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
        self.service.submit(first["task_id"], self.vote(first), reviewer_username="reviewer")
        self.assertEqual(self.service.get(second["task_id"])["comparisons"], 0)
        shutil.rmtree(first["output_dir"])
        self.assertEqual(len(self.service.catalog()["tasks"]), 1)
        self.assertTrue((self.images / "0.png").is_file())

    def test_single_image_is_immediately_certified(self) -> None:
        task = self.create(1)
        self.assertTrue(task["complete"])
        self.assertIsNone(task["pair"])
