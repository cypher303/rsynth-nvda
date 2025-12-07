"""
Minimal smoke test to generate a short WAV with default settings.

Run: python simple_smoke_wav.py
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
    VOICE_NATURAL,
)


CONFIG = {
    "sample_rate": 16000,
    "ms_per_frame": 10.0,
    "voice_source": VOICE_NATURAL,
    "flat_intonation": False,
    "speed": 1.0,
}

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def main():
    phrases = [
        "Hello world.",
        "Testing one two three.",
    ]

    synth = KlattSynth(
        sample_rate=CONFIG["sample_rate"],
        ms_per_frame=CONFIG["ms_per_frame"],
        voice_source=CONFIG["voice_source"],
        synthesis_model="cascade_parallel",
        nfcascade=6,
    )

    frame_gen = FrameGenerator(
        sample_rate=CONFIG["sample_rate"],
        ms_per_frame=CONFIG["ms_per_frame"],
        speed=CONFIG["speed"],
    )

    combined = array.array("h")
    gap = array.array("h", [0] * int(0.1 * synth.sample_rate))  # 100ms gap

    for phrase in phrases:
        phonemes = text_to_phonemes(phrase)
        elements, f0_contour, _ = phonemes_to_elements(
            phonemes,
            track_word_boundaries=False,
            flat_intonation=CONFIG["flat_intonation"],
            speed=CONFIG["speed"],
        )
        synth.reset()
        frame_gen.reset()
        for f0_hz, params in frame_gen.generate_frames(elements, f0_contour):
            frame_samples = synth.generate_frame(f0_hz, params)
            for sample in frame_samples:
                combined.append(max(-32767, min(32767, int(sample))))
        combined.extend(gap)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "simple_smoke.wav"
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(synth.sample_rate)
        wf.writeframes(combined.tobytes())

    print(f"Wrote {out_path} with {len(phrases)} phrases at {CONFIG['sample_rate']} Hz")


if __name__ == "__main__":
    main()
