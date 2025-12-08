"""
Generate WAV outputs for side-by-side listening of dev vs. “main-like” voicing/noise behaviors.

Variants:
  - dev_current: use current pipeline as-is.
  - mainish_tail_and_onset: mimic harsher main traits by forcing noise-heavy onsets and voiceless tails
    (strong af/asp at start, zero av/avc in last 5 frames).
"""

import os
import wave
import array
from pathlib import Path
from copy import deepcopy

from _rsynth import text_to_phonemes, Param
from _rsynth.klatt import KlattSynth
from _rsynth.phonemes import FrameGenerator, phonemes_to_elements


OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(exist_ok=True)


def _frames_for_text(text: str):
    phonemes = text_to_phonemes(text)
    elements, f0_contour, _ = phonemes_to_elements(phonemes)
    frame_gen = FrameGenerator()
    return list(frame_gen.generate_frames(elements, f0_contour))


def apply_variant(frames, variant: str):
    if variant == "dev_current":
        return frames

    frames = deepcopy(frames)

    if variant == "mainish_tail_and_onset":
        # Force heavy frication/aspiration and no voicing in the first 10 frames
        for i in range(min(10, len(frames))):
            f0, params = frames[i]
            params[Param.av] = 0.0
            params[Param.avc] = 0.0
            params[Param.af] = max(params[Param.af], 60.0)
            params[Param.asp] = max(params[Param.asp], 34.0)
            frames[i] = (f0, params)

        # Force voiceless tail on last 5 frames
        tail = 5
        for i in range(1, min(tail, len(frames)) + 1):
            f0, params = frames[-i]
            params[Param.av] = 0.0
            params[Param.avc] = 0.0
            # Leave af/asp to decay naturally
            frames[-i] = (f0, params)

    return frames


def synth_frames(frames, sample_rate=16000):
    synth = KlattSynth(sample_rate=sample_rate, ms_per_frame=10.0)
    samples = []
    for f0_hz, params in frames:
        samples.extend(synth.generate_frame(f0_hz, params))
    return samples


def write_wav(path: Path, samples, sample_rate=16000):
    """Write PCM16 little-endian WAV."""
    data = array.array("h", samples)
    with path.open("wb") as fh, wave.open(fh, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(data.tobytes())


def main():
    texts = [
        "hackers",
        "All the cats are hackers",
    ]
    variants = ["dev_current", "mainish_tail_and_onset"]

    for text in texts:
        frames = _frames_for_text(text)
        for variant in variants:
            variant_frames = apply_variant(frames, variant)
            samples = synth_frames(variant_frames)
            safe_text = text.replace(" ", "_").replace("'", "")
            filename = f"{safe_text}_{variant}.wav"
            out_path = OUT_DIR / filename
            write_wav(out_path, samples)
            print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
