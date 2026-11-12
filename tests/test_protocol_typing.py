from __future__ import annotations

from pathlib import Path
import re
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
            markers = {
                match.group(1): index
                for index, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1)
                if (match := re.search(r"# invalid: (\w+)", line))
            }
            self.assertEqual(set(markers), {"metric", "candidate", "decoder", "renderer"})
            expected_lines = set(markers.values())
            errors = re.findall(r"^(.+?):(\d+): error: (.+)$", result.stdout + result.stderr, re.M)
            assignment_lines = {
                int(line) for filename, line, message in errors
                if Path(filename).resolve() == source.resolve() and message.endswith("[assignment]")
            }
            self.assertEqual(assignment_lines, expected_lines, result.stdout + result.stderr)
            self.assertEqual(len(errors), len(expected_lines), result.stdout + result.stderr)
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 1, output)
        self.assertNotIn("[import-not-found]", output)
        self.assertNotIn("[attr-defined]", output)
        for interface in (
            "TonalMetric", "CandidateGenerator", "ScoreDecoder", "ScoreRenderer"
        ):
            self.assertIn(f'"{interface}"', output)
