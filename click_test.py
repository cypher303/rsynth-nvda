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
        "The experimental narrator articulated each extravagantly elongated syllable with deliberate precision, navigating polysyllabic terminology as if it were a casual conversation. Occasionally, an idiosyncratic inflection appeared on otherwise unremarkable vocabulary, producing a curious but comprehensible rhythm. Listeners who appreciated meticulously enunciated language found the performance simultaneously fascinating and exhausting. Nevertheless, the demonstration successfully highlighted characteristic weaknesses in the underlying synthesis algorithm, particularly when confronted with unexpectedly intricate consonant clusters.",
        "Sophisticated observers speculated that successive sequences of sizzling sibilants would stress the system more severely than straightforward sentences. As the session progressed, these softly hissing phrases, suffused with subtle shifts in emphasis, served as a strenuous stress test. Successive “s” and “sh” sounds, sliding seamlessly into “z” and “zh” variants, sometimes generated scarcely noticeable artifacts. Sensitive participants, however, perceived faint static, suggesting a systemic susceptibility to specific spectral features."
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
