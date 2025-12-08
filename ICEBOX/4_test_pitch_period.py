import unittest

from _rsynth import (
    KlattSynth,
    FrameGenerator,
    phonemes_to_elements,
    text_to_phonemes,
    VOICE_NATURAL,
)


class PitchPeriodTests(unittest.TestCase):
    """Regression checks for pitch period scaling."""

    def test_pitch_period_tracks_f0_contour(self):
        # Use a simple voiced utterance to get a mostly steady contour.
        text = "saw"
        phonemes = text_to_phonemes(text)
        elements, f0_contour, _ = phonemes_to_elements(
            phonemes,
            flat_intonation=True,
            speed=1.0,
        )

        synth = KlattSynth(
            sample_rate=16000,
            ms_per_frame=10.0,
            voice_source=VOICE_NATURAL,
        )
        frame_gen = FrameGenerator(
            sample_rate=16000,
            ms_per_frame=10.0,
            speed=1.0,
        )

        base_f0 = f0_contour[0]
        periods = []

        original_pitch_sync = synth._pitch_sync

        def _pitch_sync_with_trace():
            original_pitch_sync()
            periods.append(synth.T0)

        synth._pitch_sync = _pitch_sync_with_trace

        for f0_hz, params in frame_gen.generate_frames(elements, f0_contour):
            synth.generate_frame(f0_hz, params)

        mean_period = sum(periods) / max(1, len(periods))
        produced_f0 = (4 * synth.sample_rate) / mean_period

        self.assertAlmostEqual(produced_f0, base_f0, delta=base_f0 * 0.1)


if __name__ == "__main__":
    unittest.main()
