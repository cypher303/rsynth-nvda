from _rsynth.text2phone import NumeralProcessor


def test_process_text_keeps_unicode_and_returns_str():
    # Emoji and other non-ASCII characters should flow through as Python 3 str data.
    text = "Pay $5 😀"
    processed = NumeralProcessor.process_text(text)
    assert isinstance(processed, str)
    assert "five dollars" in processed
    assert "😀" in processed
