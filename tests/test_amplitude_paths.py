import math

from _rsynth import Param, text_to_phonemes
from _rsynth import klatt as klatt_mod
from _rsynth.klatt import KlattSynth
from _rsynth.phonemes import phonemes_to_elements, FrameGenerator


def _make_synth():
    synth = KlattSynth(sample_rate=16000, ms_per_frame=10.0)
    synth.jitter = 0.0
    synth.shimmer = 0.0
    synth.flutter = 0
    return synth


def test_av_uses_minus_seven_db_fudge():
    synth = _make_synth()
    params = [0.0] * Param.COUNT
    params[Param.av] = 60.0
    synth.params = params

    synth._pitch_sync()

    expected = klatt_mod.db_to_linear(max(0.0, params[Param.av] - 7.0))
    assert math.isclose(synth.amp_av, expected, rel_tol=1e-9)


def test_breathiness_respects_aturb_only():
    synth = _make_synth()
    params = [0.0] * Param.COUNT
    params[Param.av] = 60.0
    params[Param.avc] = 30.0
    params[Param.aturb] = 0.0
    synth.params = params

    synth._pitch_sync()
    assert synth.amp_breth == 0.0

    params[Param.aturb] = 20.0
    synth.params = params
    synth._pitch_sync()
    expected = klatt_mod.db_to_linear(20.0) * 0.1
    assert math.isclose(synth.amp_breth, expected, rel_tol=1e-9)


def test_generate_frame_outputs_int16_range():
    phonemes = text_to_phonemes("a")
    elements, f0_contour, _ = phonemes_to_elements(phonemes)
    frame_gen = FrameGenerator()
    frames = list(frame_gen.generate_frames(elements, f0_contour))
    assert frames
    f0_hz, params = frames[0]

    synth = _make_synth()
    samples = synth.generate_frame(f0_hz, params)

    assert len(samples) == synth.samples_per_frame
    assert all(isinstance(s, int) for s in samples)
    assert min(samples) >= -32767
    assert max(samples) <= 32767
