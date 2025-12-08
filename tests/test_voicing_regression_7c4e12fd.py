import statistics

from _rsynth import Param, FrameGenerator, phonemes_to_elements, text_to_phonemes
from _rsynth.klatt import KlattSynth


def _frames_for_text(text: str):
    phonemes = text_to_phonemes(text)
    elements, f0_contour, _ = phonemes_to_elements(phonemes)
    frame_gen = FrameGenerator()
    return list(frame_gen.generate_frames(elements, f0_contour))


def _voiced_mask(frames):
    return [params[Param.av] > 0 or params[Param.avc] > 0 for _, params in frames]


def _af_values(frames):
    return [params[Param.af] for _, params in frames]


def _synth_for_frames(frames):
    synth = KlattSynth()
    return synth


def test_hackers_keeps_voicing_and_tapers_end():
    frames = _frames_for_text("hackers")
    voiced = _voiced_mask(frames)
    voiced_count = sum(voiced)
    # Dev observed ~42 voiced of 59; ensure we keep at least 40 voiced frames.
    assert voiced_count >= 40
    # Final 5 frames should stay voiced (regression: main ended voiceless).
    assert all(voiced[-5:])
    # Early frication should not be a flat 60 throughout first 10 frames.
    first10_af = _af_values(frames[:10])
    assert len(set(round(v, 2) for v in first10_af)) > 1


def test_sentence_maintains_voicing_and_voiced_tail():
    text = "All the cats are hackers"
    frames = _frames_for_text(text)
    voiced = _voiced_mask(frames)
    voiced_count = sum(voiced)
    # Dev observed ~130 voiced of 185; guard at 120 to catch regressions toward main (98 voiced).
    assert voiced_count >= 120
    # Final 5 frames should be voiced (main had voiceless frication tail).
    assert all(voiced[-5:])
    # Mid-sentence frication block should show decay, not flat maxima.
    mid_af = _af_values(frames[80:95])
    assert statistics.pstdev(mid_af) > 0


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
