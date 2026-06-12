from __future__ import annotations

import unittest
from unittest.mock import patch

from aimusic.app import main as app_main


class TestAppMain(unittest.TestCase):
    def test_main_delegates_to_cli_entrypoint(self) -> None:
        with patch("aimusic.app.main.cli_main") as cli_main:
            app_main.main()

        cli_main.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
