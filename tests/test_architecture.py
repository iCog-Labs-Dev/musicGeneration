"""Architecture guardrails for the audio-pipeline quarantine (PDF §2.4).

Three rules:

1. No module under ``aimusic/`` outside ``aimusic/audio/`` may import ``aimusic.audio``.
2. ``aimusic.audio`` may import only ``aimusic.core.*`` and ``aimusic.theory.*``
   from the symbolic half (never ``planning``, ``decode``, ``scoring``, ``ml``, …).
3. Importing ``aimusic.core`` must not pull heavy audio/ML binaries
   (``torch``, ``librosa``, ``madmom``, ``demucs``).
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
AIMUSIC = REPO_ROOT / "aimusic"
ALLOWED_AUDIO_AIMUSIC_PREFIXES = ("aimusic.core", "aimusic.theory", "aimusic.audio")


def _iter_py_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _imported_modules(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


class TestArchitectureGuardrails(unittest.TestCase):
    def test_no_non_audio_module_imports_aimusic_audio(self) -> None:
        """Rule 1: symbolic packages must not import the audio quarantine."""
        violations: list[str] = []
        for path in _iter_py_files(AIMUSIC):
            try:
                path.relative_to(AIMUSIC / "audio")
                continue
            except ValueError:
                pass
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for name in _imported_modules(tree):
                if name == "aimusic.audio" or name.startswith("aimusic.audio."):
                    violations.append(f"{path.relative_to(REPO_ROOT)} imports {name}")
        self.assertEqual(violations, [], msg="Rule 1 violated:\n" + "\n".join(violations))

    def test_audio_only_imports_core_and_theory_from_aimusic(self) -> None:
        """Rule 2: audio may only reach into core and theory."""
        audio_root = AIMUSIC / "audio"
        if not audio_root.is_dir():
            self.skipTest("aimusic.audio not present yet")
        violations: list[str] = []
        for path in _iter_py_files(audio_root):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for name in _imported_modules(tree):
                if not name.startswith("aimusic."):
                    continue
                if any(name == p or name.startswith(p + ".") for p in ALLOWED_AUDIO_AIMUSIC_PREFIXES):
                    continue
                violations.append(f"{path.relative_to(REPO_ROOT)} imports {name}")
        self.assertEqual(violations, [], msg="Rule 2 violated:\n" + "\n".join(violations))

    def test_core_import_does_not_pull_heavy_audio_deps(self) -> None:
        """Rule 3: a fresh subprocess importing aimusic.core stays light."""
        code = (
            "import sys\n"
            "import aimusic.core\n"
            "forbidden = {'torch', 'librosa', 'madmom', 'demucs'}\n"
            "loaded = forbidden.intersection(sys.modules)\n"
            "assert not loaded, loaded\n"
            "print('ok')\n"
        )
        env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr + proc.stdout)
        self.assertIn("ok", proc.stdout)


if __name__ == "__main__":
    unittest.main()
