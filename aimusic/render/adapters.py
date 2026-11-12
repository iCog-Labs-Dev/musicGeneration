"""Result reporting around existing renderer callables."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from stat import S_ISREG
from tempfile import TemporaryDirectory
from aimusic.core.component_types import RenderRequest, RenderResult
from aimusic.core.core_types import Score
from aimusic.core.protocols import ScoreRenderer


@dataclass(frozen=True)
class RenderResultAdapter:
    """Validate a fresh artifact and atomically publish it in any format."""

    renderer: ScoreRenderer

    def __call__(self, score: Score, request: RenderRequest) -> RenderResult:
        destination = Path(request.output_path)
        with TemporaryDirectory(prefix=".aimusic-render-", dir=destination.parent) as temporary:
            staged = Path(temporary) / destination.name
            reported = self.renderer(score, replace(request, output_path=str(staged)))
            if not isinstance(reported, RenderResult):
                raise TypeError("renderer must return a RenderResult.")
            written = staged.lstat()
            if not S_ISREG(written.st_mode):
                raise ValueError("renderer must write a regular file at the requested destination.")
            actual = RenderResult(str(staged), request.format, written.st_size)
            if reported != actual:
                raise ValueError("renderer result must match the requested file and its actual details.")
            staged.replace(destination)
            return replace(actual, output_path=request.output_path)
