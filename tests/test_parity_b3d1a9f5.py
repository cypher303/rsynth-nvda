import math

import pytest

from _rsynth import klatt as klatt_mod
from _rsynth import Param, text_to_phonemes
from _rsynth.klatt import KlattSynth
from _rsynth.elements import ELEMENTS
from _rsynth.phonemes import FrameGenerator, phonemes_to_elements


def _make_synth(sample_rate=16000, f0=120.0, nfcascade=5):
    synth = KlattSynth(sample_rate=sample_rate, ms_per_frame=10.0, nfcascade=nfcascade)
    synth.jitter = 0.0
    synth.shimmer = 0.0
    synth.flutter = 0
    synth.F0Hz = f0
    return synth


def test_db_to_linear_matches_c_amptable_and_clamps():
    amptable = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 13.0,
        14.0, 16.0, 18.0, 20.0, 22.0, 25.0, 28.0, 32.0, 35.0, 40.0,
        45.0, 51.0, 57.0, 64.0, 71.0, 80.0, 90.0, 101.0, 114.0, 128.0,
        142.0, 159.0, 179.0, 202.0, 227.0, 256.0, 284.0, 318.0, 359.0, 405.0,
        455.0, 512.0, 568.0, 638.0, 719.0, 811.0, 911.0, 1024.0, 1137.0, 1276.0,
        1437.0, 1613.0, 1795.0, 2029.0, 2278.0, 2560.0, 2844.0, 3180.0, 3590.0, 4050.0,
        4550.0, 5120.0, 5680.0, 6380.0, 7190.0, 8110.0, 9110.0, 10240.0, 11370.0, 12760.0,
        14370.0, 16130.0, 17950.0, 20290.0, 22780.0, 25600.0, 28440.0, 31800.0
    ]

    for idx, val in enumerate(amptable):
        assert math.isclose(klatt_mod.db_to_linear(idx), val * 0.001, rel_tol=1e-12)

    assert klatt_mod.db_to_linear(-5) == 0.0
    assert math.isclose(klatt_mod.db_to_linear(200), amptable[-1] * 0.001, rel_tol=1e-12)


def test_av_minus_seven_db_fudge_and_negative_floor():
    synth = _make_synth()
    params = [0.0] * Param.COUNT
    params[Param.av] = 60.0
    synth.params = params
    synth._pitch_sync()
    expected = klatt_mod.db_to_linear(53.0)
    assert math.isclose(synth.amp_av, expected, rel_tol=1e-9)

    params[Param.av] = 5.0
    synth.params = params
    synth._pitch_sync()
    assert synth.amp_av == 0.0


def test_nmod_voiced_half_period_and_voiceless_full_period():
    synth = _make_synth()
    params = [0.0] * Param.COUNT

    # Voiced path
    params[Param.av] = 50.0
    synth.params = params
    synth._pitch_sync()
    assert synth.nmod == synth.T0 // 2

    # Voiceless path
    params[Param.av] = 0.0
    params[Param.avc] = 0.0
    synth.params = params
    synth._pitch_sync()
    assert synth.nmod == synth.T0 == 4


def test_breathiness_from_aturb_only_not_avc():
    synth = _make_synth()
    params = [0.0] * Param.COUNT
    params[Param.av] = 60.0
    params[Param.avc] = 30.0
    params[Param.aturb] = 0.0
    synth.params = params
    synth._pitch_sync()
    assert synth.amp_breth == 0.0

    params[Param.aturb] = 15.0
    synth.params = params
    synth._pitch_sync()
    expected = klatt_mod.db_to_linear(15.0) * 0.1
    assert math.isclose(synth.amp_breth, expected, rel_tol=1e-9)

    params[Param.avc] = 60.0
    synth.params = params
    synth._pitch_sync()
    assert math.isclose(synth.amp_breth, expected, rel_tol=1e-9)


def test_kopen_clamps_to_period_bounds():
    synth = _make_synth(f0=400.0)
    params = [0.0] * Param.COUNT
    params[Param.av] = 40.0
    params[Param.kopen] = 200.0  # scaled by 4 -> way above T0
    synth.params = params
    synth._pitch_sync()
    assert synth.nopen == synth.T0  # upper clamp

    params[Param.kopen] = 1.0  # scaled to 4
    synth.params = params
    synth._pitch_sync()
    assert synth.nopen >= 4


def test_clip_truncates_overflow_to_int16_bounds(monkeypatch):
    synth = _make_synth()
    params = [0.0] * Param.COUNT
    params[Param.av] = 0.0
    synth.params = params

    def fake_filter(_voice, _noise, _raw):
        return 40000.7

    monkeypatch.setattr(synth, "_filter_sample", fake_filter)
    samples = synth.generate_frame(0.0, params)
    assert samples[0] == 32767


def test_parallel_fudge_factors_and_noise_scalars():
    synth = _make_synth()
    params = [0.0] * Param.COUNT
    params[Param.a2] = 20.0
    params[Param.a3] = 20.0
    params[Param.a4] = 20.0
    params[Param.a5] = 20.0
    params[Param.a6] = 20.0
    params[Param.ab] = 20.0
    params[Param.asp] = 20.0
    params[Param.af] = 20.0
    synth.params = params
    synth._setup_frame()

    base_a2, _, _ = klatt_mod.set_resonator_coeffs(synth.sample_rate, synth.params[Param.f2], synth.params[Param.b2], False)
    base_a3, _, _ = klatt_mod.set_resonator_coeffs(synth.sample_rate, synth.params[Param.f3], synth.params[Param.b3], False)
    base_a4, _, _ = klatt_mod.set_resonator_coeffs(synth.sample_rate, synth.speaker.F4hz, synth.speaker.B4phz, False)
    base_a5, _, _ = klatt_mod.set_resonator_coeffs(synth.sample_rate, synth.speaker.F5hz, synth.speaker.B5phz, False)
    base_a6, _, _ = klatt_mod.set_resonator_coeffs(synth.sample_rate, synth.speaker.F6hz, synth.speaker.B6phz, False)

    gain = klatt_mod.db_to_linear(20.0)
    assert math.isclose(synth.r2p.a, base_a2 * gain * 0.15, rel_tol=1e-9)
    assert math.isclose(synth.r3p.a, base_a3 * gain * 0.06, rel_tol=1e-9)
    assert math.isclose(synth.r4p.a, base_a4 * gain * 0.04, rel_tol=1e-9)
    assert math.isclose(synth.r5p.a, base_a5 * gain * 0.022, rel_tol=1e-9)
    assert math.isclose(synth.r6p.a, base_a6 * gain * 0.03, rel_tol=1e-9)

    assert math.isclose(synth.amp_bypass, gain * 0.05, rel_tol=1e-9)
    assert math.isclose(synth.amp_asp, gain * 0.05, rel_tol=1e-9)
    assert math.isclose(synth.amp_af, gain * 0.25, rel_tol=1e-9)


def test_cascade_count_bounds_under_nyquist_and_effective_flag():
    synth = _make_synth(sample_rate=8000, nfcascade=8)
    params = [0.0] * Param.COUNT
    params[Param.av] = 0.0
    synth.params = params
    synth._set_cascade_resonators()

    assert synth._effective_nfcascade == 6
    assert synth.r7c.a == 0.0 and synth.r8c.a == 0.0


def test_element_padding_and_derived_params():
    for elem in ELEMENTS.values():
        assert len(elem.params) == Param.COUNT
        aturb = elem.params[Param.aturb]
        avc = elem.params[Param.avc]
        assert aturb.stdy == avc.stdy
        b1p = elem.params[Param.b1p]
        b1 = elem.params[Param.b1]
        assert b1p.stdy == b1.stdy


def test_generate_frame_clips_underflow(monkeypatch):
    synth = _make_synth()
    params = [0.0] * Param.COUNT

    def fake_filter(_voice, _noise, _raw):
        return -40000.0

    monkeypatch.setattr(synth, "_filter_sample", fake_filter)
    samples = synth.generate_frame(0.0, params)
    assert samples[0] == -32767


def test_generate_frame_produces_ints():
    phonemes = text_to_phonemes("a")
    elements, f0_contour, _ = phonemes_to_elements(phonemes)
    frame_gen = FrameGenerator()
    frames = list(frame_gen.generate_frames(elements, f0_contour))
    f0_hz, params = frames[0]

    synth = _make_synth()
    samples = synth.generate_frame(f0_hz, params)
    assert all(isinstance(s, int) for s in samples)

