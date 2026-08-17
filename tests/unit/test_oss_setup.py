from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from run_workflows_jsonl import load_dotenv_defaults
from scripts import oss_setup


class OssSetupTests(unittest.TestCase):
    def test_default_config_preserves_active_python_environment(self) -> None:
        config = oss_setup.default_config()

        self.assertEqual(config["PYTHON_BIN"], str(Path(sys.executable).absolute()))
        self.assertEqual(config["GRADER_PYTHON_BIN"], str(Path(sys.executable).absolute()))

    def test_write_runner_env_configures_local_comfyui_http(self) -> None:
        config = oss_setup.default_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            with mock.patch.object(oss_setup, "RUNNER_ENV_PATH", env_path):
                oss_setup.write_runner_env(config)

            values = load_dotenv_defaults(env_path)

        self.assertEqual(values["RUN_WORKFLOWS_URL"], "http://127.0.0.1:8188/")
        self.assertEqual(values["RUN_WORKFLOWS_ALLOW_INSECURE"], "true")
        self.assertEqual(values["RUN_WORKFLOWS_TIMEOUT_SECONDS"], "300")

    def test_setup_forwards_explicit_civitai_auth_routing(self) -> None:
        with (
            mock.patch.object(oss_setup.install_wildcards, "load_manifest", return_value={"terms": {}}),
            mock.patch.object(oss_setup.install_wildcards, "install") as install,
            mock.patch.object(oss_setup, "write_grader_config"),
            mock.patch.object(oss_setup, "write_env"),
            mock.patch.object(oss_setup, "write_runner_env"),
            mock.patch.object(oss_setup, "chmod_executable"),
            mock.patch.dict(oss_setup.os.environ, {"CIVITAI_API_TOKEN": "test-token"}),
        ):
            result = oss_setup.main(
                [
                    "--non-interactive",
                    "--accept-third-party-terms",
                    "--civitai-proxy",
                    "http://proxy.example:8080",
                    "--civitai-token-query",
                    "--civitai-direct-redirects",
                ]
            )

        self.assertEqual(result, 0)
        self.assertTrue(install.call_args.kwargs["civitai_token_query"])
        self.assertTrue(install.call_args.kwargs["civitai_direct_redirects"])

    def test_non_interactive_setup_can_install_cpu_grader_dependencies(self) -> None:
        with (
            mock.patch.object(oss_setup, "install_grader_dependencies") as install_dependencies,
            mock.patch.object(oss_setup, "write_grader_config"),
            mock.patch.object(oss_setup, "write_env"),
            mock.patch.object(oss_setup, "write_runner_env"),
            mock.patch.object(oss_setup, "chmod_executable"),
        ):
            result = oss_setup.main(
                ["--non-interactive", "--skip-wildcards", "--install-grader-deps", "cpu"]
            )

        self.assertEqual(result, 0)
        install_dependencies.assert_called_once_with(str(Path(sys.executable).absolute()), "cpu")

    def test_cpu_grader_dependency_profile_installs_base_then_cpu_torch(self) -> None:
        with mock.patch.object(oss_setup.subprocess, "run") as run:
            oss_setup.install_grader_dependencies("grader-python", "cpu")

        self.assertEqual(run.call_count, 2)
        first_command = run.call_args_list[0].args[0]
        second_command = run.call_args_list[1].args[0]
        self.assertEqual(first_command[:4], ["grader-python", "-m", "pip", "install"])
        self.assertTrue(first_command[-1].endswith("image_grader/requirements.txt"))
        self.assertTrue(second_command[-1].endswith("image_grader/requirements-cpu.txt"))
        self.assertTrue(all(call.kwargs["check"] for call in run.call_args_list))


if __name__ == "__main__":
    unittest.main()
