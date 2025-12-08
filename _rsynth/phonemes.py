"""
Phoneme-to-Element Conversion for RSynth

Converts SAMPA phoneme strings to element sequences for synthesis.
Port of phtoelm.c from RSynth.

Original code copyright (c) 1994,2001-2003 Nick Ing-Simmons, LGPL licensed.
"""

from typing import List, Tuple, Optional

from .elements import (
    ELEMENTS,
    ELEMENT_LIST,
    get_element_index,
)


def decline_f0(current_f0: float, elapsed_frames: int, f0_default: float) -> float:
    """
    Exponential F0 decline between stress markers.

    Port of decline_f0() from phtoelm.c lines 131-141.
    Uses the "magic constant" 0.12 for natural pitch decline.

    Args:
        current_f0: Current F0 value in Hz
        elapsed_frames: Number of frames since last stress marker
        f0_default: Default/base F0 frequency

    Returns:
        New F0 value after decline (floored at 70% of default)
    """
    f = current_f0 - 0.12 * elapsed_frames
    # Don't drop below 70% of default F0
    return max(f, 0.7 * f0_default)

# SAMPA phoneme to element sequence mapping
# Format: phoneme -> [element_names]
PHONEME_MAP = {
    # Silences
    '.': ['END'],
    ' ': ['Q'],
    '_': ['Q'],
    '#': ['Q'],

    # Affricates (compounds)
    'tS': ['T', 'CH'],
    'dZ': ['D', 'DY', 'DZ', 'ZH', 'ZH'],

    # Plosives
    'p': ['P', 'PY', 'PZ'],
    'b': ['B', 'BY', 'BZ'],
    't': ['T', 'TY', 'TZ'],
    'd': ['D', 'DY', 'DZ'],
    'k': ['K', 'KY', 'KZ'],
    'g': ['G', 'GY', 'GZ'],
    '?': ['QQ'],

    # Nasals
    'm': ['M'],
    'n': ['N'],
    'N': ['NG'],

    # Trills and flaps
    '4': ['DT'],
    'rr': ['R', 'QQ', 'R'],
    'R': ['RX'],
    '`': ['RX'],

    # Fricatives
    'f': ['F'],
    'v': ['V'],
    'T': ['TH'],
    'D': ['DH'],
    's': ['S'],
    'z': ['Z'],
    'S': ['SH'],
    'Z': ['ZH'],
    'x': ['X'],
    'h': ['H'],

    # Laterals
    'l': ['L'],
    'K': ['HL'],
    '5': ['LL'],

    # Approximants
    'w': ['W'],
    'j': ['Y'],
    'r': ['R'],

    # Diphthongs
    'eI': ['AI', 'I'],
    'aI': ['IE', 'I'],
    'OI': ['OI', 'I'],
    'aU': ['AI', 'OV'],
    '@U': ['OA', 'OV'],
    'I@': ['IA', 'IB'],
    'e@': ['AIR', 'IB'],
    'U@': ['OOR', 'IB'],
    'O@': ['OI', 'IB'],
    'oU': ['o', 'OV'],

    # Close vowels
    'i': ['EE'],
    'y': ['YY'],
    '1': ['EY'],
    '}': ['JU'],
    'M': ['UW'],
    'u': ['UU'],

    # Lax vowels
    'I': ['I'],
    'Y': ['IU'],
    'U': ['OO'],

    # Close-mid vowels
    'e': ['e'],
    'e~': ['eN'],
    '2': ['EU'],
    '@\\': ['Ur'],
    '8': ['UR'],
    '7': ['UE'],
    'o': ['o'],
    'o~': ['oN'],

    # Schwa
    '@': ['A'],

    # Open-mid vowels
    'E': ['EH'],
    '9': ['oe'],
    '9~': ['oeN'],
    '3': ['ER'],
    '3\\': ['Er'],
    'V': ['U'],
    'O': ['AW'],

    '{': ['AA'],
    '6': ['AA'],

    # Open vowels
    'a': ['a'],
    'a~': ['aN'],
    '&': ['OE'],
    'A': ['AR'],
    'Q': ['O'],
}

# Sorted by length (longest first) for proper matching
PHONEME_KEYS = sorted(PHONEME_MAP.keys(), key=lambda x: -len(x))


def phonemes_to_elements(phoneme_string: str, speed: float = 1.0,
                         f0_default: float = 120.0,
                         track_word_boundaries: bool = False,
                         flat_intonation: bool = False) -> Tuple[List[Tuple[int, int]], List[float], Optional[List[int]]]:
    """
    Convert a phoneme string to elements and F0 contour.

    This is a port of phone_to_elm() from phtoelm.c, implementing
    stress-driven F0 contour generation.

    Args:
        phoneme_string: SAMPA phoneme string (e.g., "h@'l@U")
        speed: Duration multiplier
        f0_default: Default fundamental frequency in Hz
        track_word_boundaries: If True, also return word boundary frame indices
        flat_intonation: If True, use constant F0 (robotic monotone)

    Returns:
        Tuple of:
        - List of (element_index, duration) tuples
        - F0 contour as list of floats: [f0_1, dur_1, f0_2, dur_2, ...]
          where dur values represent frame counts for transitions
        - Word boundaries (if track_word_boundaries): List of frame indices where words begin
    """
    result = []
    f0_events = []
    word_boundaries = [0] if track_word_boundaries else None  # First word starts at frame 0
    stress = 0
    i = 0
    t = 0  # Current time in frames
    last_stress_t = 0  # Time of last stress marker
    seen_vowel = False
    at_word_start = True  # Track if we're at start of a new word

    # Start F0: flat mode uses constant pitch, normal mode uses 110% rise
    if flat_intonation:
        current_f0 = f0_default  # Constant pitch for robotic voice
    else:
        current_f0 = f0_default * 1.1  # Natural sentence-initial rise
    f0_events.append(current_f0)

    while i < len(phoneme_string):
        # Check for stress markers
        ch = phoneme_string[i]

        if ch == "'":  # Primary stress
            stress += 1
            # Fall through to handle all stress levels
        if ch == "'":
            stress = 3
        elif ch == ',':  # Secondary stress
            stress = 2
        elif ch == '+':  # Tertiary stress
            stress = 1

        if ch in "'',+":
            # Stress marker found - generate F0 events (skip in flat mode)
            if not flat_intonation:
                # Port of phtoelm.c lines 183-198
                if stress > 3:
                    stress = 3
                seen_vowel = False

                # Calculate F0 decline since last stress marker
                elapsed = t - last_stress_t
                if elapsed > 0:
                    # Add duration for decline phase
                    f0_events.append(elapsed)
                    # Decline the F0
                    current_f0 = decline_f0(current_f0, elapsed, f0_default)
                    f0_events.append(current_f0)

                last_stress_t = t

                # Instant pitch pulse - push F0 up based on stress level
                # The "0" duration means instant transition
                f0_events.append(0)
                pulse_f0 = current_f0 + f0_default * stress * 0.02
                current_f0 = pulse_f0
                f0_events.append(pulse_f0)

            i += 1
            continue

        elif ch == '-':  # Hyphen (ignore)
            i += 1
            continue
        elif ch == ':':  # Length mark
            i += 1
            continue

        # Try to match a phoneme (longest match first)
        matched = False
        for phoneme in PHONEME_KEYS:
            if phoneme_string[i:].startswith(phoneme):
                element_names = PHONEME_MAP[phoneme]

                # Track word boundaries - space phoneme indicates word break
                if track_word_boundaries and phoneme == ' ':
                    at_word_start = True

                for elem_name in element_names:
                    elem = ELEMENTS.get(elem_name)
                    if elem:
                        # Get element index
                        elem_idx = list(ELEMENTS.keys()).index(elem_name)

                        # Calculate duration based on stress
                        # Stressed vowels are longer (StressDur macro from C code)
                        if stress > 0 and elem.ud != elem.du:
                            dur = int((elem.ud + (elem.du - elem.ud) * stress / 3) * speed)
                        else:
                            dur = int(elem.du * speed)

                        result.append((elem_idx, dur))

                        # Record word boundary before incrementing time
                        if track_word_boundaries and at_word_start and phoneme != ' ' and phoneme != '.':
                            if t not in word_boundaries:
                                word_boundaries.append(t)
                            at_word_start = False

                        t += dur  # Track time

                        # Check if this is a vowel (has vwl feature)
                        # Vowels have different ud and du values
                        if elem.ud != elem.du:
                            seen_vowel = True
                        elif seen_vowel:
                            # Reset stress after first consonant following vowel
                            stress = 0

                # Reset stress after phoneme group
                stress = 0
                i += len(phoneme)
                matched = True
                break

        if not matched:
            # Unknown character, skip it
            i += 1

    # Add a trailing pause so the tail decays cleanly even without punctuation.
    if result:
        last_elem = ELEMENT_LIST[result[-1][0]]
        if last_elem.name not in ("Q", "END"):
            tail_idx = get_element_index("Q")
            if tail_idx < 0:
                tail_idx = get_element_index("END")
            if tail_idx >= 0:
                tail_elem = ELEMENT_LIST[tail_idx]
                tail_dur = int(tail_elem.du * speed)
                if tail_dur <= 0:
                    tail_dur = max(1, tail_elem.du)
                result.append((tail_idx, tail_dur))
                t += tail_dur

    # Add final F0 decline to end of utterance (skip in flat mode)
    if not flat_intonation:
        elapsed = t - last_stress_t
        if elapsed > 0:
            f0_events.append(elapsed)
            final_f0 = decline_f0(current_f0, elapsed, f0_default)
            f0_events.append(final_f0)

    if track_word_boundaries:
        return result, f0_events, word_boundaries
    return result, f0_events, None
