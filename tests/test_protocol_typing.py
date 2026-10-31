from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPOSITORY = Path(__file__).resolve().parents[1]


class TestProtocolTyping(unittest.TestCase):
    def run_mypy(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable, "-m", "mypy",
                "--config-file", str(REPOSITORY / "mypy.ini"),
                "--follow-imports=silent", "--no-incremental",
                "--show-error-codes", "--no-error-summary", *arguments,
            ],
            cwd=REPOSITORY,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

    def test_actual_defaults_and_all_doubles_match_protocols(self) -> None:
        result = self.run_mypy("tests/typecheck_protocols.py")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_incompatible_implementations_are_rejected_for_each_protocol(self) -> None:
        fixture = REPOSITORY / "tests/typing_fixtures/incompatible.py.txt"
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "incompatible.py"
            source.write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")
            result = self.run_mypy(str(source))
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 1, output)
        self.assertEqual(output.count("[assignment]"), 5, output)
        self.assertNotIn("[import-not-found]", output)
        self.assertNotIn("[attr-defined]", output)
        for interface in (
            "TonalMetric", "CandidateGenerator", "ScoreDecoder", "ScoreRenderer", "ReportingScoreRenderer"
        ):
            self.assertIn(f'"{interface}"', output)
