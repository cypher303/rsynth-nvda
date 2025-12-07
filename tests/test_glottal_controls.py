import math

from _rsynth import KlattSynth, Param


def _make_synth(f0=120.0):
    synth = KlattSynth(sample_rate=16000, ms_per_frame=10.0)
    synth.jitter = 0.0
    synth.shimmer = 0.0
    synth.flutter = 0
    synth.F0Hz = f0
    synth.ns = 0
    return synth


def test_kopen_scales_open_phase_by_four():
    synth = _make_synth()
    params = [0.0] * Param.COUNT
    params[Param.av] = 60.0  # enable voicing path
    params[Param.kopen] = 30.0
    synth.params = params

    synth._pitch_sync()

    expected_T0 = int((4 * synth.sample_rate) / synth.F0Hz)
    assert synth.T0 == expected_T0
    assert synth.nopen == min(expected_T0, 4 * int(params[Param.kopen]))


def test_kopen_override_wins_and_is_scaled():
    synth = _make_synth()
    params = [0.0] * Param.COUNT
    params[Param.av] = 60.0
    params[Param.kopen] = 10.0
    synth.kopen_override = 15  # should override and be scaled by 4
    synth.params = params

    synth._pitch_sync()

    expected_T0 = int((4 * synth.sample_rate) / synth.F0Hz)
    assert synth.nopen == min(expected_T0, 4 * synth.kopen_override)


def test_nmod_defaults_to_half_period_and_respects_kskew():
    synth = _make_synth()
    params = [0.0] * Param.COUNT
    params[Param.av] = 60.0
    synth.params = params

    synth._pitch_sync()
    half_period = synth.T0 // 2
    assert synth.nmod == half_period

    params[Param.kskew] = 30.0
    synth.params = params
    synth._pitch_sync()
    assert synth.nmod == max(0, half_period - int(params[Param.kskew]))


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
