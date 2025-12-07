import array
import pytest

from _rsynth import KlattSynth
from _rsynth.phonemes import FrameGenerator, phonemes_to_elements
from _rsynth.text2phone import text_to_phonemes


def test_generate_frame_returns_python_ints_not_bytes():
    synth = KlattSynth(sample_rate=8000, ms_per_frame=10.0)
    frame_gen = FrameGenerator(sample_rate=8000, ms_per_frame=10.0, smooth=0.0)

    phonemes = text_to_phonemes("a")
    elements, f0_contour, _ = phonemes_to_elements(phonemes)

    f0_hz, params = next(frame_gen.generate_frames(elements, f0_contour))
    frame = synth.generate_frame(f0_hz, params)

    assert isinstance(frame, list)
    assert frame, "Expected at least one sample from frame generation"
    assert len(frame) == synth.samples_per_frame

    assert all(isinstance(sample, int) for sample in frame)
    assert not any(isinstance(sample, (bytes, bytearray)) for sample in frame)

    # Python 3 bytes join should reject integer samples, making Python 2 style
    # implicit coercion impossible.
    with pytest.raises(TypeError):
        b"".join(frame)  # type: ignore[arg-type]

    # But the Python 3 array module should happily pack the ints for audio I/O.
    packed = array.array("h", frame)
    assert len(packed) == len(frame)
