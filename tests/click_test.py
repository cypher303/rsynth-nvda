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
    "sample_rate": 16000,       # Match C reference defaults
    "ms_per_frame": 10.0,
    # Glottal source
    "voice_source": VOICE_NATURAL,  # Options: VOICE_IMPULSIVE, VOICE_SOOTHING, VOICE_NATURAL
    # Voice quality tweaks
    "jitter": 0.0,    # C default: none
    "shimmer": 0.0,   # C default: none
    "flutter": 0,     # C default: none
    # Prosody
    "flat_intonation": True,
    "speed": 1.0,     # 1.0 = C pacing; >1.0 slows down (longer durations), <1.0 speeds up
}

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def synth_phrase(synth: KlattSynth, frame_gen: FrameGenerator, text: str):
    """Synthesize a phrase into int16 samples."""
    phonemes = text_to_phonemes(text)
    elements, f0_contour, _ = phonemes_to_elements(
        phonemes,
        track_word_boundaries=False,
        flat_intonation=CONFIG["flat_intonation"],
        speed=CONFIG["speed"],
    )
    synth.reset()
    frame_gen.reset()
    audio = array.array("h")
    for f0_hz, params in frame_gen.generate_frames(elements, f0_contour):
        frame_samples = synth.generate_frame(f0_hz, params)
        for sample in frame_samples:
            # Clip and truncate to int16 to match the C clip() behavior
            clipped = max(-32767, min(32767, int(sample)))
            audio.append(clipped)
    return audio


def main():
    phrases = [
        "The experimental narrator articulated each extravagantly elongated syllable with deliberate precision, navigating polysyllabic terminology as if it were a casual conversation."
    ]

    synth = KlattSynth(
        sample_rate=CONFIG["sample_rate"],
        ms_per_frame=CONFIG["ms_per_frame"],
        voice_source=CONFIG["voice_source"],
        synthesis_model="cascade_parallel",
        nfcascade=6,
    )
    synth.jitter = CONFIG["jitter"]
    synth.shimmer = CONFIG["shimmer"]
    synth.flutter = CONFIG["flutter"]

    frame_gen = FrameGenerator(
        sample_rate=CONFIG["sample_rate"],
        ms_per_frame=CONFIG["ms_per_frame"],
        smooth=1.0,  # Match Holmes/C default (no extra smoothing)
        speed=CONFIG["speed"],
    )

    combined = array.array("h")
    gap = array.array("h", [0] * int(0.2 * synth.sample_rate))  # 200ms silence

    for phrase in phrases:
        combined.extend(synth_phrase(synth, frame_gen, phrase))
        combined.extend(gap)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "click_test.wav"
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(synth.sample_rate)
        wf.writeframes(combined.tobytes())

    if CONFIG["voice_source"] == VOICE_SOOTHING:
        source_label = "SOOTHING"
    elif CONFIG["voice_source"] == VOICE_IMPULSIVE:
        source_label = "IMPULSIVE"
    else:
        source_label = "NATURAL"
    print(
        f"Wrote {out_path} with {len(phrases)} phrases "
        f"at {CONFIG['sample_rate']} Hz, source={source_label}, "
        f"jitter={CONFIG['jitter']}, shimmer={CONFIG['shimmer']}, "
        f"flutter={CONFIG['flutter']}, flat_intonation={CONFIG['flat_intonation']}"
    )


if __name__ == "__main__":
    main()
