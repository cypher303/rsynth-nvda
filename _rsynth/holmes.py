"""
Frame generation and parameter interpolation (Holmes model).

Ports the frame interpolation logic from holmes.c, which sits between
phoneme-to-element conversion (phtoelm.c) and the Klatt synthesizer.
"""

from typing import List, Tuple, Optional

from .elements import ELEMENTS, Param


def linear_interpolate(a: float, b: float, t: int, d: int) -> float:
    """Linear interpolation from a to b over duration d at time t."""
    if t <= 0:
        return a
    elif t >= d:
        return b
    else:
        f = t / d
        return a + (b - a) * f


def interpolate_param(start_val: float, start_t: int,
                      end_val: float, end_t: int,
                      mid_val: float, t: int, dur: int) -> float:
    """
    Interpolate a parameter value at time t within an element.

    The parameter transitions from start_val to mid_val over start_t frames,
    stays at mid_val, then transitions to end_val over end_t frames.
    """
    steady = dur - (start_t + end_t)

    if steady >= 0:
        # Value reaches steady state
        if t < start_t:
            return linear_interpolate(start_val, mid_val, t, start_t)
        else:
            t -= start_t
            if t <= steady:
                return mid_val
            else:
                return linear_interpolate(mid_val, end_val, int(t - steady), end_t)
    else:
        # No steady state - blend start and end transitions
        f = 1.0 - (t / dur)
        sp = linear_interpolate(start_val, mid_val, t, start_t)
        ep = linear_interpolate(end_val, mid_val, dur - t, end_t)
        return f * sp + (1.0 - f) * ep


class FrameGenerator:
    """
    Generate synthesis frames from element sequences.

    Handles parameter interpolation between elements according to
    the Holmes model.
    """

    def __init__(self, sample_rate: int = 16000, ms_per_frame: float = 10.0,
                 f0_default: float = 120.0, speed: float = 1.0, smooth: float = 0.5):
        self.sample_rate = sample_rate
        self.ms_per_frame = ms_per_frame
        self.f0_default = f0_default
        self.speed = speed
        self.smooth = smooth

        # Smoothing filter state
        self._filter_state = [0.0] * Param.COUNT

    def reset(self):
        """Reset filter state."""
        self._filter_state = [0.0] * Param.COUNT

    def _smooth_param(self, idx: int, value: float) -> float:
        """Apply smoothing filter to parameter, scaled by speed."""
        # Adjust smooth to maintain proportional smoothing regardless of speed
        # At slow speed (speed > 1), use higher coefficient for same proportional effect
        effective_smooth = self.smooth ** (1.0 / self.speed)
        self._filter_state[idx] = (effective_smooth * value +
                                    (1.0 - effective_smooth) * self._filter_state[idx])
        return self._filter_state[idx]

    def generate_frames(self, elements: List[Tuple[int, int]],
                        f0_contour: Optional[List[float]] = None):
        """
        Generate synthesis parameter frames from element sequence.

        Args:
            elements: List of (element_index, duration) pairs
            f0_contour: F0 contour from phonemes_to_elements(), format:
                        [f0_start, dur_1, f0_1, dur_2, f0_2, ...]
                        Duration of 0 means instant transition (pitch pulse)

        Yields:
            (f0_hz, params) tuples for each frame
        """
        if not elements:
            return

        elem_list = list(ELEMENTS.values())

        # Initialize filter state from first element
        if elements:
            first_elem = elem_list[elements[0][0]]
            for j in range(Param.COUNT):
                self._filter_state[j] = first_elem.params[j].stdy

        # Default F0 contour if none provided
        if f0_contour is None or len(f0_contour) < 1:
            total_dur = sum(dur for _, dur in elements)
            f0_start = self.f0_default * 1.1
            f0_end = self.f0_default * 0.7
            f0_contour = [f0_start, total_dur, f0_end]

        # Track current F0 state
        f0_idx = 0
        f0_current = f0_contour[0] if len(f0_contour) > 0 else self.f0_default
        f0_target = f0_contour[2] if len(f0_contour) > 2 else f0_current
        f0_dur = int(f0_contour[1]) if len(f0_contour) > 1 else 1
        f0_t = 0

        for i, (elem_idx, dur) in enumerate(elements):
            curr_elem = elem_list[elem_idx]
            last_elem = elem_list[elements[i - 1][0]] if i > 0 else curr_elem
            next_elem = elem_list[elements[i + 1][0]] if i + 1 < len(elements) else elem_list[0]

            # Calculate transition slopes
            start_slopes = []
            end_slopes = []

            for j in range(Param.COUNT):
                # Start transition (from last to current)
                if curr_elem.params[j].rk > last_elem.params[j].rk:
                    # Current dominates
                    t = int(curr_elem.params[j].internal_duration * self.speed)
                    afrac = curr_elem.params[j].prop * 0.01
                    v = curr_elem.params[j].stdy * (1.0 - afrac) + afrac * last_elem.params[j].stdy
                else:
                    # Last dominates
                    t = int(last_elem.params[j].ed * self.speed)
                    afrac = last_elem.params[j].prop * 0.01
                    v = last_elem.params[j].stdy * (1.0 - afrac) + afrac * curr_elem.params[j].stdy
                start_slopes.append((v, t))

                # End transition (from current to next)
                if next_elem.params[j].rk > curr_elem.params[j].rk:
                    # Next dominates
                    t = int(next_elem.params[j].ed * self.speed)
                    afrac = next_elem.params[j].prop * 0.01
                    v = next_elem.params[j].stdy * (1.0 - afrac) + afrac * curr_elem.params[j].stdy
                else:
                    # Current dominates
                    t = int(curr_elem.params[j].internal_duration * self.speed)
                    afrac = curr_elem.params[j].prop * 0.01
                    v = curr_elem.params[j].stdy * (1.0 - afrac) + afrac * next_elem.params[j].stdy
                end_slopes.append((v, t))

            # Generate frames for this element
            for t in range(dur):
                params = []
                for j in range(Param.COUNT):
                    if j == Param.aturb:
                        # Breathiness follows the voice-bar amplitude (Holmes: Aturb = tp[avc])
                        val = params[Param.avc] if params else self._filter_state[Param.avc]
                        self._filter_state[j] = val
                        params.append(val)
                        continue
                    if j == Param.b1p:
                        # Parallel B1 bandwidth mirrors B1hz like the C path
                        val = params[Param.b1] if params else self._filter_state[Param.b1]
                        self._filter_state[j] = val
                        params.append(val)
                        continue
                    val = interpolate_param(
                        start_slopes[j][0], start_slopes[j][1],
                        end_slopes[j][0], end_slopes[j][1],
                        curr_elem.params[j].stdy,
                        t, dur
                    )
                    # Use faster smoothing for voicing to prevent bleed but avoid clicks
                    # av=14, avc=15 need quick transitions (~3 frames) not instant
                    if j in (Param.av, Param.avc):
                        # Fast smoothing: 0.85 decays in 3 frames vs 0.5 in 7 frames
                        # Scale with speed for consistent proportional smoothing
                        #
                        # Clamp voicing targets to zero after smoothing so tiny residuals
                        # don't keep the synthesizer in the "voiced" path and leak glottal
                        # energy into voiceless consonants, while still allowing a natural
                        # decay to zero at the end of voiced segments.
                        target = max(0.0, val)
                        effective_fast_smooth = 0.85 ** (1.0 / self.speed)
                        self._filter_state[j] = (
                            effective_fast_smooth * target
                            + (1.0 - effective_fast_smooth) * self._filter_state[j]
                        )
                        if target == 0.0 and self._filter_state[j] < 1e-6:
                            self._filter_state[j] = 0.0
                        params.append(self._filter_state[j])
                    else:
                        params.append(self._smooth_param(j, val))

                # Interpolate F0 using stress-driven contour
                if f0_dur > 0:
                    f0_hz = linear_interpolate(f0_current, f0_target, f0_t, f0_dur)
                else:
                    # Instant transition (pitch pulse)
                    f0_hz = f0_target
                f0_t += 1

                # Advance to next F0 segment when current one completes
                while f0_t >= f0_dur and f0_idx + 2 < len(f0_contour) - 2:
                    f0_t = 0
                    f0_idx += 2
                    f0_current = f0_target
                    f0_dur = int(f0_contour[f0_idx + 1]) if f0_contour[f0_idx + 1] > 0 else 1
                    if f0_idx + 2 < len(f0_contour):
                        f0_target = f0_contour[f0_idx + 2]
                    # Handle instant transitions (duration 0)
                    if f0_dur == 0 or f0_dur == 1:
                        f0_current = f0_target

                yield (f0_hz, params)

            last_elem = curr_elem
