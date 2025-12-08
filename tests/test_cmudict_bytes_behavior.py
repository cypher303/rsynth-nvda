from _rsynth.cmudict import cmu_lookup


def test_cmu_lookup_returns_unicode_string():
    # The CMU dictionary should give Python 3 str results, not byte strings.
    result = cmu_lookup("hello")
    assert isinstance(result, str)
    assert "'" in result  # stress marker appears as a normal string character


def test_cmu_lookup_does_not_implicitly_decode_bytes():
    # Bytes are not automatically decoded in Python 3; a bytes lookup should fail to match.
    assert cmu_lookup(b"hello") is None


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
