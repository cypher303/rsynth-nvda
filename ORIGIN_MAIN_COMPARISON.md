# Diff notes: `work` vs last known `origin/main`

Reference point for `origin/main`: commit `bf79cc3` ("Merge pull request #3 from cypher303/codex/identify-root-cause-of-garbled-wav-output"), the last mainline commit before the `work` branch changes.

## Changes introduced on `work`

- **Elements.def parser tweaks** (`_rsynth/elements.py`):
  - Header regex now captures the `features` field as a raw token (not forced to digits) and keeps state so we do not misinterpret lines before the parameter block.
  - Potential mismatch: the `Element.features` field remains typed as `int`, so storing a non-numeric token could violate the original contract even though the value is not currently consumed.
- **Default gain lowered** (`_rsynth/klatt.py`): `Speaker.Gain0` drops from 62 dB to 57 dB to mirror the C defaults; this reduces loudness and should reduce clipping/harshness.
- **Parameter name map expanded** (`_rsynth/klatt.py`): `PARAM_NAMES` now exposes all 24 parameters (including `an`, `a1`, `kopen`, `tlt`, `aturb`, `kskew`, `b1p`) instead of the previous 18-name subset. This matches `Param.COUNT` but may diverge from any tooling that assumed the shorter list.
- **Frame output clipping** (`_rsynth/klatt.py`): samples are now explicitly truncated to `int16` after clipping to ±32767, aligning with the C `clip()` behavior and avoiding float residue in the output buffer.
- **Test harness adjustments**:
  - `click_test.py` moved into `tests/` with default `flat_intonation=True` and writes to `tests/output/`.
  - New `tests/simple_smoke_wav.py` generates a basic WAV at 16 kHz for manual listening.

## Notes on compatibility/harshness

- The gain reduction and explicit int16 clipping both move the pipeline closer to the C reference and should *reduce* harshness rather than introduce it.
- The broadened `PARAM_NAMES` list is compatible with the existing `Param.COUNT` padding logic; it mainly improves reporting/inspection. Only external callers relying on the 18-entry ordering would notice the change.
- Parsing `features` as a free-form token could allow non-numeric values to flow into `Element.features`, which was originally an integer. If later code assumes an `int`, that mismatch could cause errors or inconsistent behavior once the field is consumed.
- The test config change to `flat_intonation=True` affects only the click stress harness and not synthesis defaults, but it means those test WAVs no longer exercise the default intonation path from `origin/main`.
