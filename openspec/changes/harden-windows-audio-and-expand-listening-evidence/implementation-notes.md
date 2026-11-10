# Implementation verification notes

## Windows CI group

- Fork PR #5 run `42789164898` (2026-11-10) passed Linux lint, typecheck,
  core tests, and ML tests, but its hosted Windows job failed before test
  execution at the standard-user `Start-Process` bootstrap with "The parameter
  is incorrect." The gate is explicitly skipped pending task 8.0a; this run is
  not native Windows acceptance evidence.
- Equivalent clean native run ID: `bb4fb51962c64969bb20afaa5ac6a03e`.
- Source revision before this checkpoint: `fb7c0a252648761560d54f9f1b044db91aa78ad4`.
- Wheel SHA-256: `2686f1f0188290a96c8cee257e83431dce631268c5ef0182ed3623bbd2561d45`.
- Result-record SHA-256: `ceedafbdb34a54cee02357ba7997ea29d88ba4ca2cbce98de696dae5f09732e6`.
- Result: 56 installed-wheel tests run under a non-administrator, non-elevated medium-integrity token; 55 passed and one symlink case was narrowly unavailable with `WinError 1314`. Mandatory substitute invariants passed, the asset environment was clean, and no unexpected skip occurred.
- The local run used a new virtual environment in an external path containing spaces and non-ASCII characters. A hosted `windows-latest` run remains required before relying on GitHub-specific account/ACL behavior.

## Linux CI-equivalent group

- Run ID: `wsl-ubuntu-20261105-dd690b020da0`.
- Full unittest log SHA-256: `dd690b020da06d11c7b451d72fdd13a4ec31b9084efe4a85f066f14ca9ecc783`.
- `flake8 aimusic tests ui.py`: passed.
- `mypy`: 159 source files passed.
- Compile/import checks: 68 executable modules imported.
- Unit discovery: 825 passed, 43 skipped in 442.716 seconds.
- ML job targets: 2 passed, 7 dependency/capability skips.

These are local equivalent runs, not hosted GitHub run identifiers. No organization PR or workflow dispatch was performed.

## Asset-free reviewer-page browser smoke

- Runtime: OpenCode 2.2.5, Electron 44.9.1, Chromium 152.0.7977.167 on macOS.
- Served an extracted reviewer-only synthetic tree on loopback; no `private/` tree was exposed.
- All six pages loaded under their generated CSP with no console errors. Lighthouse reported accessibility 1.0 and best-practices 1.0.
- A same-origin WAV reached `readyState=4`, emitted `loadedmetadata`, `canplay`, and `playing`, and advanced playback time. Focusing the native audio control and pressing Space started playback with the visible focus outline applied.
- Removing one WAV produced the expected HTTP 404 and media error while the visible recovery text continued to instruct the reviewer to mark the clip not reviewed and avoid substitution.
- This smoke run proves browser mechanics for synthetic media only. It is not native FluidSynth evidence or human listening acceptance.
