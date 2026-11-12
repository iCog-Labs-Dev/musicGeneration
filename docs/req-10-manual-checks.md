# Manual checks before presenting REQ-10

Automated tests cover component substitution and successful CLI/UI-helper runs with all four replacements selected together in 12-EDO and 19-EDO. They inspect the selected path, metric-derived diagnostics, known decoder output, actual MIDI, and CLI RenderPackage. The combined test deliberately uses a one-note decoder double; it verifies plumbing, not musical quality.

Historical browser-generation/download acceptance is complete using the user's six downloaded files and settings screenshots. The user confirmed playing both examples. These files came from the original submission, before the later upstream planner changes. They establish basic playback, not musical quality or synthesizer tuning for every future run.

The mentor-review revision adds JSON-based selection and a shared CLI/UI factory. Automated checks exercise all four configured replacements in both tunings. The final verification record in [the implementation report](req-10-implementation.md) identifies the current revision's evidence. No extra manual component-substitution test is needed. Before a live presentation, generate and listen to the particular examples you will present.

## Check the browser controls and downloads

1. Activate the project's Python environment and run `python ui.py` from the repository root.
2. Open `http://localhost:7860` if the browser does not open automatically.
3. Select seed 11, four beats, and 12-EDO. Turn off section-wise planning for this short example, then generate. Current upstream enables section-wise planning by default; use a longer piece to demonstrate multiple sections.
4. Confirm generation completes, the summary appears, and MIDI, score JSON, and manifest downloads work.
5. Repeat with 19-EDO using MPE rendering.

If the optional MIDI-to-audio converter is unavailable, the page can show a preview warning while still offering the generated files. That is distinct from failure to create or download MIDI. The UI helper writes flat score/MIDI/manifest files; the CLI additionally writes a RenderPackage.

Completion: **passed through user-provided browser downloads**. The 12-EDO run is `3ef170d5-573e-42f5-a6da-879a932710d6`; the 19-EDO run is `dc7523c8-888a-434b-96e2-649b5565da2d`. Both contain 28 notes across bass, comping, drums, and lead, with seed 11, four beats, 105 BPM, and the requested tuning. MIDI notes, timing, and velocity match the saved scores; all six downloads match the generated files byte for byte. The settings screenshots confirm 12/19 EDO and MPE selection.

## Open and listen to the generated MIDI

1. Open a default-generated 12-EDO MIDI file in your MIDI player or DAW.
2. Confirm it loads, contains the expected tracks, and plays without stuck notes or obvious timing problems.
3. If demonstrating microtonal playback, open the 19-EDO MPE file in a player configured to honor its pitch bends. A player that ignores them cannot validate the tuning.

If you do not have a MIDI player, use the UI's audio playback control when its preview is available. Confirm that audio starts and stops normally for each example you will present.

Use a default-generated piece for this check, rather than the one-note component-test fixture. Listening evaluates the presentation experience; it does not establish that a new distance formula produces better music.

Completion: **user confirmed playing both the 12-EDO and 19-EDO examples**, with no issue reported. The playback application and tuning behavior were not independently inspected. No additional manual component-substitution check is required for this issue.

## Checks not required for the component-interface submission

You do not need to repeat every replacement combination by hand. To rerun the combined automated test:

```bash
python -m unittest tests.test_artifact_injection.TestArtifactInjection.test_all_four_replacements_complete_cli_and_ui_workflows -v
```

To check selection through the actual command-line parser and UI helper, also run:

```bash
python -m unittest tests.test_component_selection_workflows -v
```

Long-piece performance, optional trained models, external audio synthesis and a deployed-service security audit remain separate validation scopes. Remote CI must be checked on the updated PR; a local test pass is not a remote CI pass.
