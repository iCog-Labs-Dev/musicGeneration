"""Import every executable Python surface so import failures are fatal in CI."""

from __future__ import annotations

import argparse
import importlib
import pkgutil

import aimusic


OPTIONAL_MODULE_PREFIXES = ("aimusic.ml.",)


def import_all_executable_modules(*, include_optional: bool = False, include_ui: bool = True) -> tuple[str, ...]:
    """Import executable surfaces available in the selected installation."""
    module_names = tuple(
        sorted(
            module.name
            for module in pkgutil.walk_packages(
                aimusic.__path__,
                prefix=f"{aimusic.__name__}.",
            )
            if include_optional or not module.name.startswith(OPTIONAL_MODULE_PREFIXES)
        )
    )
    for module_name in module_names:
        importlib.import_module(module_name)
    if include_ui:
        importlib.import_module("ui")
        return (*module_names, "ui")
    return module_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="also import modules whose optional dependencies are installed",
    )
    args = parser.parse_args()
    imported = import_all_executable_modules(include_optional=args.include_optional)
    print(f"Imported {len(imported)} executable modules.")
