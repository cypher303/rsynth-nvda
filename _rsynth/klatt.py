"""
Klatt Formant Synthesizer - Python Port of RSynth opsynth.c

This is a port of the RSynth Klatt synthesizer to pure Python.
Original C code copyright (c) 2002-2004 Nick Ing-Simmons, LGPL licensed.

The Klatt synthesizer generates speech using formant synthesis with:
- Cascade formant filters for voiced sounds
- Parallel formant filters for fricatives
- Glottal pulse waveform for voicing
- Noise source for fricatives and aspiration
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Callable

PI = math.pi

# Natural voice samples from Praat's klatt.cpp - LF model waveform
# These provide a more realistic glottal source than the synthetic pulse
# Based on "A four-parameter model of the glottal flow" Fant et al. 1985
NATURAL_SAMPLES = [
    -310, -400, 530, 356, 224, 89, 23, -10, -58, -16, 461, 599, 536, 701, 770,
    605, 497, 461, 560, 404, 110, 224, 131, 104, -97, 155, 278, -154, -1165,
    -598, 737, 125, -592, 41, 11, -247, -10, 65, 92, 80, -304, 71, 167, -1, 122,
    233, 161, -43, 278, 479, 485, 407, 266, 650, 134, 80, 236, 68, 260, 269, 179,
    53, 140, 275, 293, 296, 104, 257, 152, 311, 182, 263, 245, 125, 314, 140, 44,
    203, 230, -235, -286, 23, 107, 92, -91, 38, 464, 443, 176, 98, -784, -2449,
    -1891, -1045, -1600, -1462, -1384, -1261, -949, -730
]

def _generate_soothing_waveform(samples: int = 80) -> list:
    """
    Create a gentle Rosenberg-style pulse: half-cosine rise, half-cosine fall,
    with a small negative tail to soften closure. Returned values are centered
    around 0 and normalized to roughly match NATURAL_SAMPLES amplitude.
    """
    open_len = int(samples * 0.6)
    close_len = samples - open_len - 5  # leave a few samples for negative tail
    tail_len = samples - open_len - close_len
    wave = []

    # Smooth rise (0 -> 1)
    for i in range(open_len):
        wave.append(0.5 * (1 - math.cos(math.pi * (i / open_len))))

    # Smooth fall (1 -> 0)
    for i in range(close_len):
        wave.append(0.5 * (1 + math.cos(math.pi * (i / close_len))))

    # Small negative tail to mimic glottal closure
    for i in range(tail_len):
        frac = i / max(1, tail_len - 1)
        wave.append(-0.1 * math.sin(math.pi * frac))

    # Normalize to similar scale as NATURAL_SAMPLES (peak around 1200-1500)
    peak = max(abs(v) for v in wave) or 1.0
    scale = 1400.0 / peak
    return [v * scale for v in wave]

SOOTHING_SAMPLES = _generate_soothing_waveform()

# Voice source types
VOICE_IMPULSIVE = 1  # Simple impulse/ramp (original RSynth)
VOICE_NATURAL = 2    # LF model natural samples (from Praat)
VOICE_SOOTHING = 3   # Softened Rosenberg-style pulse
VOICE_CUSTOM = 4     # User-supplied waveform


@dataclass
class Resonator:
    """
    2nd-order IIR resonator filter with coefficient interpolation.

    Implements the difference equation:
        y[n] = a*x[n] + b*y[n-1] + c*y[n-2]

    Interpolation smoothly transitions coefficients to avoid audio artifacts.
    """
    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    p1: float = 0.0  # y[n-1]
    p2: float = 0.0  # y[n-2]
    # Interpolation increments (applied each sample for smooth transitions)
    a_inc: float = 0.0
    b_inc: float = 0.0
    c_inc: float = 0.0

    def process(self, input_sample: float) -> float:
        """Process one sample through the resonator."""
        x = self.a * input_sample + self.b * self.p1 + self.c * self.p2
        self.p2 = self.p1
        self.p1 = x
        return x

    def interpolate(self):
        """Apply one step of coefficient interpolation."""
        self.a += self.a_inc
        self.b += self.b_inc
        self.c += self.c_inc

    def set_target(self, a_new: float, b_new: float, c_new: float, steps: int):
        """Set target coefficients and calculate interpolation increments."""
        if steps > 0:
            self.a_inc = (a_new - self.a) / steps
            self.b_inc = (b_new - self.b) / steps
            self.c_inc = (c_new - self.c) / steps
        else:
            self.a, self.b, self.c = a_new, b_new, c_new
            self.a_inc = self.b_inc = self.c_inc = 0.0

    def reset(self):
        """Reset filter state."""
        self.p1 = 0.0
        self.p2 = 0.0


def set_resonator_coeffs(sample_rate: int, freq: float, bandwidth: float,
                         is_cascade: bool = True) -> tuple:
    """
    Calculate resonator coefficients from frequency and bandwidth.

    Args:
        sample_rate: Sample rate in Hz
        freq: Center frequency in Hz
        bandwidth: Bandwidth in Hz
        is_cascade: True for cascade (pass-through), False for parallel (zero)

    Returns:
        (a, b, c) coefficients for the resonator
    """
    minus_pi_t = -PI / sample_rate
    two_pi_t = -2.0 * minus_pi_t

    # Check if resonator is within Nyquist limit
    if 2 * freq - bandwidth <= sample_rate:
        # Adjust if upper skirt exceeds Nyquist
        if 2 * (freq + bandwidth) > sample_rate:
            low = freq - bandwidth
            freq = (sample_rate / 2 + low) / 2
            bandwidth = freq - low

        r = math.exp(minus_pi_t * bandwidth)
        c = -(r * r)
        b = r * math.cos(two_pi_t * freq) * 2.0
        a = 1.0 - b - c
    else:
        # Beyond Nyquist - make it a no-op
        a = 1.0 if is_cascade else 0.0
        b = 0.0
        c = 0.0

    return a, b, c


def set_antiresonator_coeffs(sample_rate: int, freq: float, bandwidth: float) -> tuple:
    """
    Calculate anti-resonator (zero) coefficients.

    Anti-resonators create notches in the frequency response.
    """
    # First compute ordinary resonator coefficients
    a, b, c = set_resonator_coeffs(sample_rate, freq, bandwidth, is_cascade=True)

    # Convert to antiresonator: a' = 1/a, b' = -b/a, c' = -c/a
    if a != 0:
        a_inv = 1.0 / a
        return a_inv, -b * a_inv, -c * a_inv
    return 1.0, 0.0, 0.0


def db_to_linear(dB: float) -> float:
    """Convert decibels to linear amplitude."""
    # Exact C amptable mapping: 0-87 dB -> linear gain
    amptable = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 13.0,
        14.0, 16.0, 18.0, 20.0, 22.0, 25.0, 28.0, 32.0, 35.0, 40.0,
        45.0, 51.0, 57.0, 64.0, 71.0, 80.0, 90.0, 101.0, 114.0, 128.0,
        142.0, 159.0, 179.0, 202.0, 227.0, 256.0, 284.0, 318.0, 359.0, 405.0,
        455.0, 512.0, 568.0, 638.0, 719.0, 811.0, 911.0, 1024.0, 1137.0, 1276.0,
        1437.0, 1613.0, 1795.0, 2029.0, 2278.0, 2560.0, 2844.0, 3180.0, 3590.0, 4050.0,
        4550.0, 5120.0, 5680.0, 6380.0, 7190.0, 8110.0, 9110.0, 10240.0, 11370.0, 12760.0,
        14370.0, 16130.0, 17950.0, 20290.0, 22780.0, 25600.0, 28440.0, 31800.0
    ]
    if dB <= 0:
        return 0.0
    idx = int(dB)
    if idx >= len(amptable):
        idx = len(amptable) - 1
    return amptable[idx] * 0.001


@dataclass
class Speaker:
    """Voice/speaker characteristics."""
    F0Hz: float = 120.0      # Fundamental frequency
    Gain0: float = 57.0      # Overall gain in dB (matches C defaults)
    F4hz: float = 3900.0     # Fourth formant freq
    B4hz: float = 400.0      # Fourth formant bandwidth
    F5hz: float = 4700.0     # Fifth formant freq
    B5hz: float = 150.0      # Fifth formant bandwidth
    F6hz: float = 4900.0     # Sixth formant freq
    B6hz: float = 150.0      # Sixth formant bandwidth
    FNPhz: float = 270.0     # Nasal pole frequency
    BNhz: float = 500.0      # Nasal bandwidth
    B4phz: float = 500.0     # Parallel 4th formant bw
    B5phz: float = 600.0     # Parallel 5th formant bw
    B6phz: float = 800.0     # Parallel 6th formant bw
    B1phz: float = 80.0      # Parallel 1st formant bw
    # F1/F2/F3 offset and scale (applied to phoneme values)
    F1_offset: float = 0.0   # F1 frequency offset in Hz
    F1_scale: float = 1.0    # F1 frequency multiplier
    F2_offset: float = 0.0   # F2 frequency offset in Hz
    F2_scale: float = 1.0    # F2 frequency multiplier
    F3_offset: float = 0.0   # F3 frequency offset in Hz
    F3_scale: float = 1.0    # F3 frequency multiplier


# Parameter indices (matching rsynth.h enum)
class Param:
    fn = 0   # Nasal zero freq
    f1 = 1   # First formant freq
    f2 = 2   # Second formant freq
    f3 = 3   # Third formant freq
    b1 = 4   # First formant bandwidth
    b2 = 5   # Second formant bandwidth
    b3 = 6   # Third formant bandwidth
    b1 = 4   # First formant bandwidth
    b2 = 5   # Second formant bandwidth
    b3 = 6   # Third formant bandwidth
    an = 7   # Parallel nasal pole amplitude (AN/ANP)
    a1 = 8   # Parallel F1 amplitude
    a2 = 9   # Parallel/serial F2 frication amplitude
    a3 = 10  # Parallel F3 frication amplitude
    a4 = 11  # Parallel F4 frication amplitude
    a5 = 12  # Parallel F5 frication amplitude
    a6 = 13  # Parallel F6 frication amplitude
    ab = 14  # Amp of bypass frication
    av = 15  # Amp of voicing
    avc = 16 # Amp of voice-bar
    asp = 17 # Amp of aspiration
    af = 18  # Amp of frication
    kopen = 19   # Open-phase length override (samples at 4x)
    tlt = 20     # Spectral tilt in dB
    aturb = 21   # Breathiness (Aturb) in dB
    kskew = 22   # Skewness (not yet used)
    b1p = 23     # Parallel F1 bandwidth
    COUNT = 24

PARAM_NAMES = [
    'fn', 'f1', 'f2', 'f3', 'b1', 'b2', 'b3', 'an', 'a1',
    'a2', 'a3', 'a4', 'a5', 'a6', 'ab', 'av', 'avc', 'asp', 'af',
    'kopen', 'tlt', 'aturb', 'kskew', 'b1p',
]


class KlattSynth:
    """
    Klatt formant synthesizer.

    Generates speech by passing a glottal source through a cascade of
    formant resonators, with parallel resonators for frication noise.
    """

    def __init__(self, sample_rate: int = 16000, ms_per_frame: float = 10.0,
                 speaker: Optional[Speaker] = None,
                 voice_source: int = VOICE_NATURAL,
                 custom_waveform: Optional[List[float]] = None,
                 kopen_override: Optional[int] = None,
                 tlt_db: float = 0.0,
                 breathiness_db: float = 0.0,
                 synthesis_model: str = "cascade_parallel",
                 nfcascade: int = 5):
        """
        Initialize the synthesizer.

        Args:
            sample_rate: Output sample rate in Hz
            ms_per_frame: Milliseconds per synthesis frame
            speaker: Speaker characteristics (uses defaults if None)
            voice_source: VOICE_IMPULSIVE (legacy) or VOICE_NATURAL (LF glottal)
            custom_waveform: Optional custom pulse table for VOICE_CUSTOM/VOICE_SOOTHING
            kopen_override: Optional open-phase length (samples at 4x sample rate)
            tlt_db: Spectral tilt in dB (higher values roll off highs)
            breathiness_db: Extra breath noise in dB (Aturb analogue)
            synthesis_model: "cascade_parallel" (default) or "all_parallel"
            nfcascade: number of cascade formants when cascade_parallel (<=8)
        """
        self.sample_rate = sample_rate
        self.samples_per_frame = int((sample_rate * ms_per_frame) / 1000)
        self.speaker = speaker or Speaker()
        self.custom_waveform = custom_waveform
        self.kopen_override = kopen_override  # If None, defaults to T0//3
        self.tlt_db = tlt_db
        self.breathiness_db = breathiness_db
        self.synthesis_model = synthesis_model
        self.nfcascade = max(1, min(8, nfcascade))
        self._effective_nfcascade = self.nfcascade

        # Current frame parameters
        self.params = [0.0] * Param.COUNT

        # Voicing state
        self.nper = 0       # Current position in voicing period (*4)
        self.T0 = 4         # Fundamental period in samples * 4
        self.nopen = 0      # Number of samples in open phase
        self.nmod = 0       # Position to begin noise modulation (for Kskew)
        self.F0Hz = self.speaker.F0Hz

        # Amplitude values (converted from dB)
        self.amp_av = 0.0      # Voice amplitude
        self.amp_bypass = 0.0  # Bypass frication amplitude
        self.amp_asp = 0.0     # Aspiration amplitude
        self.amp_af = 0.0      # Frication amplitude
        self.amp_avc = 0.0     # Voice-bar amplitude
        self.amp_turb = 0.0    # Turbulence amplitude
        self.amp_breth = 0.0   # Breathiness amplitude (Aturb analogue)

        # Attack ramp for voiceless sounds (prevents "h" burst)
        self._noise_ramp_samples = 0  # Counter for ramp-up
        self._noise_ramp_length = 80  # ~5ms at 16kHz (smooth attack)
        self._voiceless_started = False  # Track if we've started voiceless (for cold start)

        # Random number generator state
        self.seed = 5

        # Sample counter
        self.ns = 0

        # Cascade resonators
        self.rgl = Resonator()   # Glottal low-pass
        self.rnz = Resonator()   # Nasal zero (anti-resonator)
        self.rnpc = Resonator()  # Nasal pole
        self.r8c = Resonator()   # F8 cascade (fixed 7500 Hz)
        self.r7c = Resonator()   # F7 cascade (fixed 6500 Hz)
        self.r6c = Resonator()   # F6 cascade
        self.r5c = Resonator()   # F5 cascade
        self.rsc = Resonator()   # Special cascade (3500 Hz)
        self.r4c = Resonator()   # F4 cascade
        self.r3c = Resonator()   # F3 cascade
        self.r2c = Resonator()   # F2 cascade
        self.r1c = Resonator()   # F1 cascade

        # Parallel resonators
        self.rnp_p = Resonator() # Parallel nasal pole
        self.r1p = Resonator()   # F1 parallel
        self.r6p = Resonator()   # F6 parallel
        self.r5p = Resonator()   # F5 parallel
        self.r4p = Resonator()   # F4 parallel
        self.r3p = Resonator()   # F3 parallel
        self.r2p = Resonator()   # F2 parallel

        # Output resonator (low-pass)
        self.rout = Resonator()

        # DC blocking filter state (removes low-frequency hum from resonator accumulation)
        self.dc_x1 = 0.0  # Previous input
        self.dc_y1 = 0.0  # Previous output

        # Smoothing filter coefficient
        self.smooth = 0.5

        # Speed multiplier
        self.speed = 1.0
        # Proportion of voiced energy sent through the parallel nasal/F1 branch
        self.parallel_voice_mix = 1.0 if synthesis_model == "all_parallel" else 0.0

        # Voice naturalness parameters
        # Jitter: random variation in pitch period (1-2% typical for natural voice)
        self.jitter = 0.015  # 1.5% pitch jitter
        # Shimmer: random variation in amplitude (3-5% typical for natural voice)
        self.shimmer = 0.04  # 4% amplitude shimmer

        # Flutter: 3-sine-wave F0 modulation from Klatt & Klatt 1990
        # More natural than simple jitter - creates quasi-random pitch variation
        # Value 0-100, default 20 (matches Praat/eSpeak)
        self.flutter = 20
        self._flutter_t = 0  # Time counter for flutter sines

        # Voice source type: VOICE_IMPULSIVE (legacy) or VOICE_NATURAL (default)
        # Natural uses LF model samples for more realistic glottal source
        self.voice_source = voice_source
        self._sample_pos = 0.0  # Position in natural samples table
        self._tilt_prev = 0.0
        if self.tlt_db > 0:
            self._tilt_alpha = min(0.99, 1 - math.pow(10.0, -self.tlt_db / 20.0))
        else:
            self._tilt_alpha = 0.0

        # Track voicing state for transition detection (resonator reset)
        self._was_voiced = False

    def _set_cascade_resonators(self, interpolate: bool = False):
        """Set up cascade path resonators based on current parameters.

        Args:
            interpolate: If True, use smooth interpolation for F1-F3 resonators
                        to avoid audio artifacts from abrupt coefficient changes.
        """
        sr = self.sample_rate
        ep = self.params
        spk = self.speaker
        steps = self.samples_per_frame if interpolate else 0
        eff_nfcascade = max(1, min(self.nfcascade, 8))
        if eff_nfcascade >= 7 and sr < 16000:
            eff_nfcascade = 6
        self._effective_nfcascade = eff_nfcascade

        # Nasal pole and zero (fixed, no interpolation needed)
        a, b, c = set_resonator_coeffs(sr, spk.FNPhz, spk.BNhz, True)
        self.rnpc.a, self.rnpc.b, self.rnpc.c = a, b, c

        a, b, c = set_antiresonator_coeffs(sr, ep[Param.fn], spk.BNhz)
        self.rnz.a, self.rnz.b, self.rnz.c = a, b, c

        # Special resonator at 3500 Hz (fixed)
        a, b, c = set_resonator_coeffs(sr, 3500, 1800, True)
        self.rsc.a, self.rsc.b, self.rsc.c = a, b, c

        if eff_nfcascade >= 8:
            a, b, c = set_resonator_coeffs(sr, 7500, 600, True)
            self.r8c.a, self.r8c.b, self.r8c.c = a, b, c

        if eff_nfcascade >= 7:
            a, b, c = set_resonator_coeffs(sr, 6500, 500, True)
            self.r7c.a, self.r7c.b, self.r7c.c = a, b, c

        if eff_nfcascade >= 6:
            a, b, c = set_resonator_coeffs(sr, spk.F6hz, spk.B6hz, True)
            self.r6c.a, self.r6c.b, self.r6c.c = a, b, c

        # F5, F4 (from speaker settings, don't change per frame)
        if eff_nfcascade >= 5:
            a, b, c = set_resonator_coeffs(sr, spk.F5hz, spk.B5hz, True)
            self.r5c.a, self.r5c.b, self.r5c.c = a, b, c

        if eff_nfcascade >= 4:
            a, b, c = set_resonator_coeffs(sr, spk.F4hz, spk.B4hz, True)
            self.r4c.a, self.r4c.b, self.r4c.c = a, b, c

        # F3, F2, F1 with speaker offset and scale applied
        # These change per-phoneme, so use interpolation to smooth transitions

        # F3: apply offset and scale, clamp to safe range (1500-3500 Hz)
        f3_adj = ep[Param.f3] * spk.F3_scale + spk.F3_offset
        f3_adj = max(1500, min(3500, f3_adj))
        a, b, c = set_resonator_coeffs(sr, f3_adj, ep[Param.b3], True)
        if interpolate:
            self.r3c.set_target(a, b, c, steps)
        else:
            self.r3c.a, self.r3c.b, self.r3c.c = a, b, c

        # F2: apply offset and scale, clamp to safe range (700-2500 Hz)
        f2_adj = ep[Param.f2] * spk.F2_scale + spk.F2_offset
        f2_adj = max(700, min(2500, f2_adj))
        a, b, c = set_resonator_coeffs(sr, f2_adj, ep[Param.b2], True)
        if interpolate:
            self.r2c.set_target(a, b, c, steps)
        else:
            self.r2c.a, self.r2c.b, self.r2c.c = a, b, c

        # F1: apply offset and scale, clamp to safe range (200-1000 Hz)
        f1_adj = ep[Param.f1] * spk.F1_scale + spk.F1_offset
        f1_adj = max(200, min(1000, f1_adj))
        a, b, c = set_resonator_coeffs(sr, f1_adj, ep[Param.b1], True)
        if interpolate:
            self.r1c.set_target(a, b, c, steps)
        else:
            self.r1c.a, self.r1c.b, self.r1c.c = a, b, c

    def _pitch_sync(self):
        """Update pitch-synchronous parameters with jitter for naturalness."""
        F0Hz = self.F0Hz
        ep = self.params

        if ep[Param.av] > 0 or ep[Param.avc] > 0:
            # Calculate base pitch period
            base_T0 = int((4 * self.sample_rate) / F0Hz)

            # Add jitter (random variation in pitch period)
            # This makes the voice sound more natural/less robotic
            if self.jitter > 0:
                jitter_amount = base_T0 * self.jitter * (random.random() * 2 - 1)
                self.T0 = max(4, int(base_T0 + jitter_amount))
            else:
                self.T0 = base_T0

            av_db = max(0.0, ep[Param.av] - 7.0)  # Match C's -7 dB voicing fudge
            self.amp_av = db_to_linear(av_db)
            self.amp_avc = db_to_linear(ep[Param.avc])
            self.amp_turb = self.amp_avc * 0.05  # Reduced from 0.1 to prevent friction-like artifacts at slow speed
            kopen_param = ep[Param.kopen] if len(ep) > Param.kopen else 0
            if self.kopen_override is not None:
                self.nopen = max(4, min(self.T0, int(self.kopen_override)))
            elif kopen_param > 0:
                self.nopen = max(4, min(self.T0, int(kopen_param)))
            else:
                self.nopen = self.T0 // 3
            # Breathiness (Aturb analogue)
            aturb_param = ep[Param.aturb] if len(ep) > Param.aturb else 0
            breathiness_db = max(self.breathiness_db, aturb_param)
            self.amp_breth = db_to_linear(breathiness_db) * 0.1
            # Skew: reduce noise modulation start when skew > 0 (like C's nmod adjustment)
            kskew = ep[Param.kskew] if len(ep) > Param.kskew else 0
            self.nmod = self.T0 - max(0, min(self.T0, int(kskew)))
        else:
            self.T0 = 4
            self.nopen = self.T0
            self.amp_av = 0.0
            self.amp_avc = 0.0
            self.amp_breth = 0.0
            self.nmod = self.T0
            # Resonator reset is now handled in generate_frame() at frame boundaries
            # to avoid resetting every 4 samples (T0=4 when voiceless)

        if self.T0 != 4 or self.ns == 0:
            # Update glottal low-pass filter
            a, b, c = set_resonator_coeffs(self.sample_rate, 0, 2 * F0Hz, True)
            self.rgl.a, self.rgl.b, self.rgl.c = a, b, c
            # Note: cascade resonators are now set in generate_frame() with interpolation

        # Update spectral tilt smoothing coefficient if a per-frame tilt is provided
        tlt_param = ep[Param.tlt] if len(ep) > Param.tlt else 0
        tilt_db = max(self.tlt_db, tlt_param)
        if tilt_db > 0:
            self._tilt_alpha = min(0.99, 1 - math.pow(10.0, -tilt_db / 20.0))
        else:
            self._tilt_alpha = 0.0

    def _gen_noise(self) -> float:
        """Generate Gaussian-distributed noise."""
        noise = 0.0
        for _ in range(16):
            # Linear congruential generator
            self.seed = (self.seed * 1664525 + 1) & 0xFFFFFFFF
            # Convert to signed 14-bit value
            nrand = ((self.seed << 1) >> 18) - 8192
            noise += nrand
        return noise / 2

    def _gen_voice(self, noise: float) -> float:
        """Generate voice waveform at 4x sample rate."""
        voice = 0.0
        amp = 4096.0

        for _ in range(4):
            if self.nper >= self.T0:
                self.nper = 0
                self._pitch_sync()

            if self.voice_source in (VOICE_NATURAL, VOICE_SOOTHING, VOICE_CUSTOM):
                if self.voice_source == VOICE_NATURAL:
                    table = NATURAL_SAMPLES
                elif self.voice_source == VOICE_SOOTHING:
                    table = SOOTHING_SAMPLES
                else:
                    table = self.custom_waveform or NATURAL_SAMPLES

                # Interpolate through the sample table based on position in period
                num_samples = len(table)
                alpha = self.nper / self.T0 if self.T0 > 0 else 0
                pos = alpha * num_samples
                idx = int(pos)
                frac = pos - idx

                if idx < num_samples - 1:
                    # Linear interpolation between samples
                    voice = table[idx] * (1 - frac) + table[idx + 1] * frac
                elif idx < num_samples:
                    voice = table[idx]
                else:
                    voice = 0

                # Scale to match impulsive source amplitude
                voice *= (amp / 2500.0)
            else:
                # Default: impulsive source (original RSynth)
                # Voice source shape: linear ramp for 1/3, parabola for 2/3
                alpha = self.nper / self.T0 if self.T0 > 0 else 0

                if alpha <= 1.0 / 3:
                    voice = 3 * amp * alpha
                else:
                    voice = amp * ((9 * alpha - 12) * alpha + 3)

            if self.amp_breth > 0:
                voice += self.amp_breth * noise

            if self._tilt_alpha > 0:
                voice = voice * (1 - self._tilt_alpha) + self._tilt_prev * self._tilt_alpha
                self._tilt_prev = voice

            self.nper += 1

        return voice

    def _antiresonator(self, r: Resonator, input_sample: float) -> float:
        """Process through anti-resonator (saves input, not output)."""
        x = r.a * input_sample + r.b * r.p1 + r.c * r.p2
        r.p2 = r.p1
        r.p1 = input_sample
        return x

    def _dc_block(self, sample: float) -> float:
        """
        Remove DC offset using a single-pole high-pass filter.

        This prevents low-frequency hum from accumulating in the resonators.
        Cutoff is approximately 3-5Hz at typical sample rates (matches DECtalk).

        y[n] = x[n] - x[n-1] + R * y[n-1], where R ≈ 0.99
        """
        DC_R = 0.99
        y = sample - self.dc_x1 + DC_R * self.dc_y1
        self.dc_x1 = sample
        self.dc_y1 = y
        return y

    def _setup_frame(self):
        """Set up resonators for current frame parameters."""
        sr = self.sample_rate
        ep = self.params
        spk = self.speaker
        Gain0 = spk.Gain0 - 3

        # Parallel resonators with gain (apply same F2/F3 offset/scale as cascade)
        # F1 uses speaker offset/scale too for the parallel branch
        f1_adj = ep[Param.f1] * spk.F1_scale + spk.F1_offset
        f1_adj = max(200, min(1000, f1_adj))

        f2_adj = ep[Param.f2] * spk.F2_scale + spk.F2_offset
        f2_adj = max(700, min(2500, f2_adj))
        a, b, c = set_resonator_coeffs(sr, f2_adj, ep[Param.b2], False)
        # Parallel branch scales mirror the C pipeline (Holmes amplitudes into
        # Klatt parallel resonators are intentionally attenuated).
        self.r2p.a, self.r2p.b, self.r2p.c = a * db_to_linear(ep[Param.a2]) * 0.15, b, c

        f3_adj = ep[Param.f3] * spk.F3_scale + spk.F3_offset
        f3_adj = max(1500, min(3500, f3_adj))
        a, b, c = set_resonator_coeffs(sr, f3_adj, ep[Param.b3], False)
        self.r3p.a, self.r3p.b, self.r3p.c = a * db_to_linear(ep[Param.a3]) * 0.06, b, c

        # Parallel F1 and nasal pole for voiced path
        b1p_bw = ep[Param.b1p] if len(ep) > Param.b1p else spk.B1phz
        a, b, c = set_resonator_coeffs(sr, f1_adj, b1p_bw, False)
        self.r1p.a, self.r1p.b, self.r1p.c = a * db_to_linear(ep[Param.a1]) * 0.4, b, c

        a, b, c = set_resonator_coeffs(sr, spk.FNPhz, spk.BNhz, False)
        self.rnp_p.a, self.rnp_p.b, self.rnp_p.c = a * db_to_linear(ep[Param.an]) * 0.6, b, c

        a, b, c = set_resonator_coeffs(sr, spk.F4hz, spk.B4phz, False)
        self.r4p.a, self.r4p.b, self.r4p.c = a * db_to_linear(ep[Param.a4]) * 0.04, b, c

        a, b, c = set_resonator_coeffs(sr, spk.F5hz, spk.B5phz, False)
        self.r5p.a, self.r5p.b, self.r5p.c = a * db_to_linear(ep[Param.a5]) * 0.022, b, c

        a, b, c = set_resonator_coeffs(sr, spk.F6hz, spk.B6phz, False)
        self.r6p.a, self.r6p.b, self.r6p.c = a * db_to_linear(ep[Param.a6]) * 0.03, b, c

        # Amplitudes
        self.amp_bypass = db_to_linear(ep[Param.ab]) * 0.05
        self.amp_asp = db_to_linear(ep[Param.asp]) * 0.05
        self.amp_af = db_to_linear(ep[Param.af]) * 0.25

        # Output low-pass filter
        if Gain0 <= 0:
            Gain0 = 57
        a, b, c = set_resonator_coeffs(sr, 0, sr / 2, True)
        self.rout.a, self.rout.b, self.rout.c = a * db_to_linear(Gain0), b, c

    def _filter_sample(self, voice: float, noise: float, raw_voice: float) -> float:
        """Apply cascade and parallel filters to produce output sample."""
        cascade_voice = voice
        if self.synthesis_model != "all_parallel":
            # Cascade path: voice through nasal and formant resonators
            cascade_voice = self.rnpc.process(cascade_voice)
            cascade_voice = self._antiresonator(self.rnz, cascade_voice)
            if self._effective_nfcascade >= 1:
                cascade_voice = self.r1c.process(cascade_voice)
            if self._effective_nfcascade >= 2:
                cascade_voice = self.r2c.process(cascade_voice)
            if self._effective_nfcascade >= 3:
                cascade_voice = self.r3c.process(cascade_voice)
            if self._effective_nfcascade >= 4:
                cascade_voice = self.r4c.process(cascade_voice)
                cascade_voice = self.rsc.process(cascade_voice)
            if self._effective_nfcascade >= 5:
                cascade_voice = self.r5c.process(cascade_voice)
            if self._effective_nfcascade >= 6:
                cascade_voice = self.r6c.process(cascade_voice)
            if self._effective_nfcascade >= 7:
                cascade_voice = self.r7c.process(cascade_voice)
            if self._effective_nfcascade >= 8:
                cascade_voice = self.r8c.process(cascade_voice)

        # Parallel path:
        #   - Voiced energy through parallel nasal/F1
        #   - Frication noise through higher parallel resonators
        parallel_voiced = (
            self.r1p.process(raw_voice) + self.rnp_p.process(raw_voice)
        ) * self.parallel_voice_mix
        parallel_noise = (
            self.r2p.process(noise) +
            self.r3p.process(noise) +
            self.r4p.process(noise) +
            self.r5p.process(noise) +
            self.r6p.process(noise) +
            self.amp_bypass * noise
        )

        if self.synthesis_model == "all_parallel":
            voice = parallel_voiced + parallel_noise
        else:
            voice = cascade_voice + parallel_voiced + parallel_noise

        # Final low-pass and gain
        voice = self.rout.process(voice)

        # Remove DC offset to prevent low-frequency hum
        voice = self._dc_block(voice)

        return voice

    def generate_frame(self, F0Hz: float, params: List[float]) -> List[float]:
        """
        Generate one frame of audio samples.

        Args:
            F0Hz: Fundamental frequency for this frame
            params: List of synthesis parameters (length Param.COUNT)

        Returns:
            List of audio samples for this frame
        """
        self.params = params

        # Check for voicing transition; reset filters only when entering true silence
        is_voiced = params[Param.av] > 0 or params[Param.avc] > 0
        is_silence = (
            params[Param.af] <= 0
            and params[Param.asp] <= 0
            and params[Param.ab] <= 0
            and params[Param.a2] <= 0
            and params[Param.a3] <= 0
            and params[Param.a4] <= 0
            and params[Param.a5] <= 0
            and params[Param.a6] <= 0
        )
        if not is_voiced:
            # Only reset on voiced→silence (or cold-start silence); let filters decay through frication like C
            if is_silence and (self._was_voiced or not self._voiceless_started):
                for r in [self.rnpc, self.rnz, self.r1c, self.r2c, self.r3c,
                          self.r4c, self.r5c, self.r6c, self.r7c, self.r8c, self.rsc, self.rgl,
                          self.r1p, self.rnp_p,
                          self.r2p, self.r3p, self.r4p, self.r5p, self.r6p, self.rout]:
                    r.reset()
                self._noise_ramp_samples = 0  # Restart ramp when we re-enter noise after silence
            self._voiceless_started = True
            self.amp_av = 0.0
            self.amp_avc = 0.0
            self.amp_turb = 0.0
        else:
            self._voiceless_started = False
        self._was_voiced = is_voiced

        # Apply flutter to F0 for natural pitch variation
        self.F0Hz = self._apply_flutter(F0Hz)
        self._setup_frame()

        # Set up cascade resonators with smooth interpolation
        # This prevents abrupt coefficient changes that cause audio artifacts
        self._set_cascade_resonators(interpolate=True)

        samples = [0.0] * self.samples_per_frame

        for i in range(self.samples_per_frame):
            noise = self._gen_noise()

            # Skip voice pulse generation during voiceless sounds
            # This prevents LFO-like pulse leakage into voiceless phonemes
            # Use params directly (amp_av is set inside _gen_voice/_pitch_sync)
            if self.params[Param.av] > 0 or self.params[Param.avc] > 0:
                voice = self._gen_voice(noise)
                lpvoice = self.rgl.process(voice)
            else:
                voice = 0.0
                lpvoice = 0.0

            # Add breathiness during glottal open phase
            if self.nper < self.nopen:
                voice += self.amp_turb * noise

            # Reduce noise in second half of glottal open phase
            if self.nper > self.nmod:
                noise *= 0.5

            # Apply voicing amplitude with shimmer for naturalness
            voice *= self.amp_av
            if self.shimmer > 0 and self.amp_av > 0:
                # Shimmer: random amplitude variation makes voice less robotic
                shimmer_factor = 1.0 + self.shimmer * (random.random() * 2 - 1)
                voice *= shimmer_factor

            # Calculate noise ramp factor (0 to 1) for smooth attack
            if self._noise_ramp_samples < self._noise_ramp_length:
                noise_ramp = self._noise_ramp_samples / self._noise_ramp_length
                self._noise_ramp_samples += 1
            else:
                noise_ramp = 1.0

            # Add aspiration noise (with ramp to prevent "h" burst)
            voice += self.amp_asp * noise_ramp * noise

            # Add voice-bar (low-passed voice)
            voice += self.amp_avc * lpvoice

            # Frication noise (with ramp to prevent "h" burst)
            noise *= self.amp_af * noise_ramp

            # Apply filters
            sample = self._filter_sample(voice, noise, raw_voice=voice)

            # Clip and truncate to int16 like the C clip()
            sample = max(-32767, min(32767, sample))
            samples[i] = int(sample)

            # Apply smooth coefficient interpolation for F1, F2, F3
            # This creates natural formant transitions between phonemes
            self.r1c.interpolate()
            self.r2c.interpolate()
            self.r3c.interpolate()

            self.ns += 1

        return samples

    def _apply_flutter(self, f0_hz: float) -> float:
        """
        Apply 3-sine-wave flutter for natural pitch variation.

        Based on Klatt & Klatt 1990 "Analysis, synthesis and perception of
        voice quality variations among female and male talkers" JASA 87(2).

        Flutter creates quasi-random pitch variation that sounds more natural
        than simple random jitter. Uses three slowly varying sine waves at
        frequencies chosen to avoid simple harmonic relationships.

        Args:
            f0_hz: Base fundamental frequency in Hz

        Returns:
            Modulated F0 with flutter applied

        Notes:
            Flutter is only used when voicing is active (av/avc > 0) because
            voiceless frames skip _gen_voice() entirely and zero the voicing
            amps earlier in synth_frame(). That means any phase mismatch here
            cannot by itself inject voiced energy into fricatives; leakage in
            voiceless consonants stems from how resonators and voicing amps are
            gated/reset rather than from the flutter LFO timing.
        """
        if self.flutter <= 0:
            return f0_hz

        # Port of flutter() from Praat's klatt.cpp lines 165-179
        fla = self.flutter / 50.0          # Flutter amount scaled
        flb = f0_hz / 100.0                # Base F0 scaled
        # Flutter in the original C code increments "time_count" once per frame
        # and uses 2 * PI for the sine waves (nsynth.c lines 320-336).
        # Use the same stepping to keep modulation speed consistent with RSynth.
        flc = math.sin(2 * PI * 12.7 * self._flutter_t)  # Fast sine
        fld = math.sin(2 * PI * 7.1 * self._flutter_t)   # Medium sine
        fle = math.sin(2 * PI * 4.7 * self._flutter_t)   # Slow sine

        # Combine the three sines for quasi-random variation
        delta_f0 = fla * flb * (flc + fld + fle) * 10

        # Increment time counter once per frame and wrap like the original C
        # implementation. The wrap isn't needed to prevent Python overflow, but
        # it keeps the phase math identical to RSynth and avoids huge arguments
        # to sin() that could accumulate floating-point error over long runs.
        self._flutter_t += 1
        if self._flutter_t > 1000:
            self._flutter_t = 0

        return f0_hz + delta_f0

    def reset(self):
        """Reset synthesizer state."""
        self.nper = 0
        self.ns = 0
        self.seed = 5
        self._flutter_t = 0
        self._sample_pos = 0.0
        self._was_voiced = False
        self._noise_ramp_samples = 0
        self._voiceless_started = False
        self._tilt_prev = 0.0
        self.nmod = 0

        # Reset DC blocker state
        self.dc_x1 = 0.0
        self.dc_y1 = 0.0

        # Reset all resonators
        for r in [self.rgl, self.rnz, self.rnpc, self.r8c, self.r7c, self.r6c,
                  self.r5c, self.rsc, self.r4c, self.r3c, self.r2c, self.r1c,
                  self.rnp_p, self.r1p,
                  self.r6p, self.r5p, self.r4p, self.r3p, self.r2p, self.rout]:
            r.reset()
