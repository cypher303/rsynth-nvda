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
    VOICE_IMPULSIVE,
    VOICE_NATURAL,
    VOICE_SOOTHING
)


CONFIG = {
    # Audio quality
    "sample_rate": 22050,       # Common options: 8000, 16000, 22050
    "ms_per_frame": 10.0,
    # Glottal source
    "voice_source": VOICE_NATURAL,  # Options: VOICE_IMPULSIVE, VOICE_SOOTHING, VOICE_NATURAL
    # Voice quality tweaks
    "jitter": 0.015,
    "shimmer": 0.04,
    "flutter": 20,
    # Prosody
    "flat_intonation": False,
}


def synth_phrase(synth: KlattSynth, frame_gen: FrameGenerator, text: str):
    """Synthesize a phrase into int16 samples."""
    phonemes = text_to_phonemes(text)
    elements, f0_contour, _ = phonemes_to_elements(
        phonemes,
        track_word_boundaries=False,
        flat_intonation=CONFIG["flat_intonation"],
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
        "The experimental narrator articulated each extravagantly elongated syllable with deliberate precision, navigating polysyllabic terminology as if it were a casual conversation. Occasionally, an idiosyncratic inflection appeared on otherwise unremarkable vocabulary, producing a curious but comprehensible rhythm. Listeners who appreciated meticulously enunciated language found the performance simultaneously fascinating and exhausting. Nevertheless, the demonstration successfully highlighted characteristic weaknesses in the underlying synthesis algorithm, particularly when confronted with unexpectedly intricate consonant clusters."
    ]

    synth = KlattSynth(
        sample_rate=CONFIG["sample_rate"],
        ms_per_frame=CONFIG["ms_per_frame"],
        voice_source=CONFIG["voice_source"],
    )
    synth.jitter = CONFIG["jitter"]
    synth.shimmer = CONFIG["shimmer"]
    synth.flutter = CONFIG["flutter"]

    frame_gen = FrameGenerator(
        sample_rate=CONFIG["sample_rate"],
    )

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

    source_label = (
        "NATURAL" if CONFIG["voice_source"] == VOICE_NATURAL else "IMPULSIVE"
    )
    print(
        f"Wrote {out_path} with {len(phrases)} phrases "
        f"at {CONFIG['sample_rate']} Hz, source={source_label}, "
        f"jitter={CONFIG['jitter']}, shimmer={CONFIG['shimmer']}, "
        f"flutter={CONFIG['flutter']}, flat_intonation={CONFIG['flat_intonation']}"
    )


if __name__ == "__main__":
    main()
