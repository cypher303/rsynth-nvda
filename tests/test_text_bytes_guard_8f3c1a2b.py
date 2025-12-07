import pytest

from _rsynth.text2phone import text_to_phonemes


def test_text_to_phonemes_rejects_bytes():
    """Bytes inputs should not be silently coerced like Python 2 strings."""
    with pytest.raises(TypeError):
        text_to_phonemes(b"hello world")


def test_text_to_phonemes_returns_str_type():
    result = text_to_phonemes("hello")
    assert isinstance(result, str)
    assert result, "Expected non-empty phoneme output for simple text"
