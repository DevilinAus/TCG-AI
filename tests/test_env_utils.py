from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from backend.tcg_ai.env_utils import load_project_env, parse_env_line


class EnvUtilsTests(unittest.TestCase):
    def test_parse_env_line_supports_export_prefix_and_quotes(self) -> None:
        self.assertEqual(
            parse_env_line('export TCG_AI_STANDARD_REMOTE_TIMEOUT_MS="1800000"'),
            ("TCG_AI_STANDARD_REMOTE_TIMEOUT_MS", "1800000"),
        )
        self.assertIsNone(parse_env_line("   # comment"))
        self.assertIsNone(parse_env_line("MALFORMED"))

    def test_load_project_env_prefers_env_local_over_stale_process_env(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            (project_root / ".env").write_text(
                "TCG_AI_STANDARD_REMOTE_TIMEOUT_MS=1500\nIGNORED_KEY=value\n",
                encoding="utf-8",
            )
            (project_root / ".env.local").write_text(
                'export TCG_AI_STANDARD_REMOTE_TIMEOUT_MS="1800000"\n',
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"TCG_AI_STANDARD_REMOTE_TIMEOUT_MS": "1500"},
                clear=False,
            ):
                env_sources = load_project_env(project_root=project_root)

                self.assertEqual(os.environ["TCG_AI_STANDARD_REMOTE_TIMEOUT_MS"], "1800000")
                self.assertEqual(
                    env_sources["TCG_AI_STANDARD_REMOTE_TIMEOUT_MS"],
                    str(project_root / ".env.local"),
                )
                self.assertNotIn("IGNORED_KEY", os.environ)


if __name__ == "__main__":
    unittest.main()
