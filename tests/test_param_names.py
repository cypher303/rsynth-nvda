import math

from _rsynth import klatt, FrameGenerator, phonemes_to_elements
from _rsynth.text2phone import text_to_phonemes


def test_param_names_cover_all_params():
    assert len(klatt.PARAM_NAMES) == klatt.Param.COUNT
    # Ensure ordering aligns by sampling a few key indices
    assert klatt.PARAM_NAMES[klatt.Param.fn] == "fn"
    assert klatt.PARAM_NAMES[klatt.Param.kopen] == "kopen"
    assert klatt.PARAM_NAMES[klatt.Param.b1p] == "b1p"


def test_generated_frames_use_full_param_count():
    phonemes = text_to_phonemes("a")
    elements, f0_contour, _ = phonemes_to_elements(phonemes)
    frame_gen = FrameGenerator()

    frames = list(frame_gen.generate_frames(elements, f0_contour))
    assert frames, "Expected at least one frame from sample text"
    for _, params in frames:
        assert len(params) == klatt.Param.COUNT
        # Existing defaults should remain finite even for padded slots
        assert all(math.isfinite(value) for value in params)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
