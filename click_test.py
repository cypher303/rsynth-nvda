"""
Ad-hoc click stress test generator for RSynth (Python port).

Run: python click_test.py

Generates click_test.wav with phrases likely to expose boundary clicks:
- endings without punctuation vs with punctuation
- voiceless/fricative finals
- long hissy sequences
"""

import array
import wave
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _rsynth import (  # noqa: E402
    KlattSynth,
    FrameGenerator,
    phonemes_to_elements,
    text_to_phonemes,
)


def synth_phrase(synth: KlattSynth, frame_gen: FrameGenerator, text: str):
    """Synthesize a phrase into int16 samples."""
    phonemes = text_to_phonemes(text)
    elements, f0_contour, _ = phonemes_to_elements(
        phonemes,
        track_word_boundaries=False,
    )
    synth.reset()
    frame_gen.reset()
    audio = array.array("h")
    for f0_hz, params in frame_gen.generate_frames(elements, f0_contour):
        frame_samples = synth.generate_frame(f0_hz, params)
        for sample in frame_samples:
            audio.append(int(max(-32767, min(32767, sample))))
    return audio


def main():
    phrases = [
        "hello world",
        "hello world.",
        "laugh",
        "scratch",
        "book",
        "roof",
        "tough cough plough dough",
        "ssssssss",
        "final s hiss",
        "click check at end",
    ]

    synth = KlattSynth(sample_rate=16000)
    frame_gen = FrameGenerator(sample_rate=16000)

    combined = array.array("h")
    gap = array.array("h", [0] * int(0.2 * synth.sample_rate))  # 200ms silence

    for phrase in phrases:
        combined.extend(synth_phrase(synth, frame_gen, phrase))
        combined.extend(gap)

    out_path = Path(__file__).with_name("click_test.wav")
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(synth.sample_rate)
        wf.writeframes(combined.tobytes())

    print(f"Wrote {out_path} with {len(phrases)} phrases.")


if __name__ == "__main__":
    main()
