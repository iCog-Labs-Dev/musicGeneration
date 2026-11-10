# Event-preserving groove

`aimusic.audio.midi_events` owns MIDI transformation. The symbolic core and its
file contract are unchanged. PrettyMIDI is **not** the serialization authority
for `apply_groove`: reconstructing only instruments/notes loses channels, RPN,
SysEx and tempo maps. The private `_apply_to_midi` adapter still accepts/returns
PrettyMIDI objects for existing analysis callers; it cannot recover information
already discarded before the call.

## Representation and round trips

`MidiEvents.load(path)` / `from_mido(midi)` produce frozen `MidiEvent` and
`MidiNote` records. An event retains its absolute integer tick, original track,
channel, original order, frozen message payload and associated note IDs. Notes
retain both key-release and sustain-release ticks. Velocity-zero note-ons are
paired as releases without converting their payloads to note-offs.

Supported inputs are synchronous **type 0/1, positive PPQN, MIDI port 0**, with
one source note-owning track per channel. Chords with distinct keys are allowed;
per-channel MPE member notes remain in their original source track. Conductor
tracks, track names, global/meta messages, tempo/meter changes, controllers,
program/bank changes, bends, channel/poly pressure and SysEx (including MTS) are
retained. Conflicting global changes at the same tick, unmatched/zero-duration
notes, overlapping key-down instances of the same channel/key, unreleased pedal
holds, cross-track note ownership, invalid deltas and events after EOT fail.

`to_mido()` / `save(path)` restore track-local delta times. Untouched equal-tick
order is `(track, original order)`, not message-type sorting. Retimed channel
reuse may move a release/reset bundle before the next setup/onset, without
reordering unrelated channels. Added replay events have separate negative IDs.
This is **event/payload fidelity**, not byte-identical SMF encoding: mido may
normalize running status and append a missing EOT. Existing EOT positions are
preserved unless a retimed event requires extending its track.

## Timing and expression

- Swing/grid/accent positions use source musical ticks, not seconds divided by
  the first tempo. Anticipation uses the active meter. Tempo/meter events never
  move. Tempo conversion uses rational integration across every tempo change.
- Offsets use **round-half-even on the tick displacement**, not on the absolute
  destination tick. No cumulative floating-point tick integration is used.
- Note duration remains constant in seconds unless articulation scales it.
  Key-down expression follows the same onset/duration mapping. Sustain-tail
  expression follows the shifted key release; the pedal release stays fixed.
- Pitch wheel, CC74, channel pressure and key-specific poly pressure attach to
  active voices (including pedal-held voices), same-tick pre-onset setup, or
  unambiguous same-tick post-release resets. Shared channel expression can move
  only when every affected voice maps it to the same destination tick.
- Ordinary persistent controllers keep their original timeline. An earlier note
  may need additive initial-state replay. Later notes do not rewind volume/pan/
  sustain automation. Replay that changes another sounding voice is rejected;
  every voice (including retimed voices) is audited against an independent source
  controller timeline, not replay-contaminated state. An earlier onset may use its
  own initial source controls; subsequent source writes, even redundant writes,
  remain absolute. Program/bank and pitch/expression state are reconstructed
  for each intended voice. Bank selects latch on program change; replay also
  preserves pending bank selects for later source programs. RPN replay is an
  ordered select/data transaction;
  it restores the **current** parameter selector so later source data-entry
  messages still address the right parameter.
- Ghosts preserve the existing probability/velocity API but require an empty
  channel gap. Their state is restored before subsequent source setup. An
  overlapping ghost is rejected, not assigned an invented channel or made safe
  by shortening the original note.

After transformation, note pairing and sustain releases are checked again.
Every voice's source pitch/expression trajectory is compared at its transformed
onset and at every channel-event boundary through its sounding lifetime. Matching
initial bends alone is insufficient: two voices may subsequently require
incompatible bends or pressure. No automatic channel reallocation is attempted.

## Supported tuning subset and explicit failures

The verified MPE subset is **independent per-channel pitch wheel**, absolute
RPN data entry for bend sensitivity (including cents), fine/coarse tuning,
CC74 and pressure. Tests reconstruct sequential and overlapping-channel 19-EDO
notes independently of the transform, including a generated fractional-base,
48-semitone-range fixture, within **one cent** of intended pitches.

Opaque SysEx is preserved, not interpreted. Every moved channel event, not just
note endpoints, is checked against opaque SysEx boundaries (including same-tick
order). SysEx invalidates channel snapshots: defaults/pre-reset values are no
longer known. Retiming within a region may proceed without replay, but replay
requiring unknown state fails closed. Parameter data entry after SysEx requires
an explicitly known RPN/NRPN selector. Partial RPN/NRPN MSB/LSB values and
post-SysEx selector state participate in lifetime/trajectory validation; an unknown
component is not treated as zero. Ghosts cannot copy state across MTS/SysEx.
Preserving MTS bytes does **not**
prove MTS rendering capability. Active MPE master-zone interpretation, alternate
hold pedals/channel-mode resets, device-dependent relative parameter increment/
decrement and replay of unknown/device-specific RPN/NRPN parameters are not
silently approximated: retiming rejects them. Unchanged valid event streams
still round-trip without interpreting these messages.

`MidiEventError` describes malformed/ambiguous inputs. `ChannelStateConflict`
identifies unsafe retiming; voice conflicts include source track, zero-based
channel, key, note ID and, where relevant, the conflicting tick. For example:

```text
track 1, channel 2, note 60 (id 0): incompatible simultaneous pitch/expression state at tick 250; retry with groove disabled
```

Retry a **valid original** with groove disabled/light cleanup; malformed input
must first be repaired. Failures occur before MIDI/diff/manifest publication.
MIDI and both sidecars are preflighted for resolved-path/inode aliases to the
source and each other. All three are serialized and staged before publication;
leaf entries are atomically replaced, never opened for writing through a link.
Publication errors roll back existing artifacts; if the filesystem also refuses
rollback, the error identifies retained private recovery files. This is per-file atomicity, not
a crash-atomic multi-file transaction; concurrent writers to one bundle are not
supported. These checks certify MIDI-event fidelity only, not audible f0, instrument quality, CLAP or FAD.
Stem isolation and renderer capability enforcement are separate later tasks.

## Verification

```bash
python -m pytest tests/test_audio_midi_events.py tests/test_audio_midi_scaling.py tests/test_audio_groove.py tests/test_architecture.py -q
python -m unittest discover -s tests -p 'test_*.py'
flake8 aimusic tests ui.py
mypy aimusic tests ui.py
```

No native synth, SoundFont, network, GPU or key is needed. The regression suite
includes randomized controller order and MPE shift properties, tempo-boundary
and half-even offsets, payload comparisons, source immutability, safe ghosts,
same-tick channel reuse, incompatible trajectories and pedal-held conflicts.
Expression association uses an active-voice sweep; tempo/meter, event identity and
channel boundary lookups are indexed. Safe ghosts are reserved and inserted as a
batch with a single snapshot pass and reparse. See the deterministic 2k/4k
[scaling guard](audio-testing.md#deterministic-scaling-guard).

References: [mido SMF delta-time/type semantics](https://mido.readthedocs.io/en/latest/files/midi.html),
[parameter selection and relative-controller ambiguity](https://philrees.co.uk/nrpnq.htm),
[design decision 3](../openspec/changes/deliver-midi2audio-audible-demo/design.md).
