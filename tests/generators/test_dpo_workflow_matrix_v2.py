from __future__ import annotations

import unittest

from generators.generate_dpo_workflow_matrix_v2 import generate_matrix
from tests.generators._consolidated_matrix_contract import assert_consolidated_matrix


class DpoWorkflowMatrixV2Tests(unittest.TestCase):
    def test_generate_matrix_writes_consolidated_v2_contract(self) -> None:
        assert_consolidated_matrix(
            self,
            generate_matrix=generate_matrix,
            version_tag="v2",
            template_short_names=("compact", "cinematic"),
            template_prefix="research_v2/",
        )


if __name__ == "__main__":
    unittest.main()
