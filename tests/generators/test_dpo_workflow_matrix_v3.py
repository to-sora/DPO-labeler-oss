from __future__ import annotations

import unittest

from generators.generate_dpo_workflow_matrix_v3 import generate_matrix
from tests._paths import REPO_ROOT
from tests.generators._consolidated_matrix_contract import assert_consolidated_matrix


class DpoWorkflowMatrixV3Tests(unittest.TestCase):
    def test_generate_matrix_writes_consolidated_v3_contract(self) -> None:
        assert_consolidated_matrix(
            self,
            generate_matrix=generate_matrix,
            version_tag="v3",
            template_short_names=("compact", "cinematic"),
            template_prefix="research_v3/",
        )

    def test_research_v3_templates_do_not_reference_v2_wildcards(self) -> None:
        for path in (REPO_ROOT / "template" / "prompt_templates" / "research_v3").glob("*.txt"):
            with self.subTest(path=path.name):
                text = path.read_text(encoding="utf-8")
                self.assertNotIn("__research_v2/", text)
                self.assertIn("__research_v3/", text)


if __name__ == "__main__":
    unittest.main()
