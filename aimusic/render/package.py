"""Thin re-export of the RenderPackage contract producer.

Canonical implementation: ``aimusic.core.render_package``.
"""

from __future__ import annotations

from aimusic.core.render_package import (
    STRUCTURE_SCHEMA,
    MICROTONAL_TOLERANCE_CENTS,
    ContractViolation,
    RenderPackage,
    StructureDoc,
    assert_contract_invariants,
    build_structure,
    build_tuning,
    load_render_package,
    write_render_package,
)

__all__ = [
    "STRUCTURE_SCHEMA",
    "MICROTONAL_TOLERANCE_CENTS",
    "ContractViolation",
    "RenderPackage",
    "StructureDoc",
    "assert_contract_invariants",
    "build_structure",
    "build_tuning",
    "load_render_package",
    "write_render_package",
]
