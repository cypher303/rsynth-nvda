import pytest

from _rsynth.text2phone import normalize_text, text_to_phonemes


def test_normalize_text_rejects_bytes():
    # Python 3 regex helpers should fail fast when handed bytes instead of str.
    with pytest.raises(TypeError):
        normalize_text(b"Hello")


def test_text_to_phonemes_rejects_bytes():
    # End-to-end phoneme conversion should keep Python 3 str expectations.
    with pytest.raises(TypeError):
        text_to_phonemes(b"hi")


def test_normalize_text_returns_lowercase_string():
    normalized = normalize_text("Hello, WORLD! 2024")
    assert isinstance(normalized, str)
    assert normalized == "hello , world . twenty twenty-four"


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
