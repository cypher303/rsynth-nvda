"""
RSynth GUI Speech Synthesizer

A wxPython GUI application that exposes rsynth's speech synthesis parameters
as sliders, with text input, Play, Render to WAV, and Stop buttons.
"""

import array
import json
import os
import sys
import tempfile
import threading
import wave

# Add rsynth module path (use local _rsynth copy)
rsynth_dir = os.path.dirname(os.path.abspath(__file__))
if rsynth_dir not in sys.path:
    sys.path.insert(0, rsynth_dir)

import wx

# Import from local _rsynth module
from _rsynth import (
    KlattSynth, Speaker, text_to_phonemes,
    phonemes_to_elements, FrameGenerator,
    VOICE_IMPULSIVE, VOICE_NATURAL
)

# Try to import audio playback library
try:
    import simpleaudio as sa
    HAS_SIMPLEAUDIO = True
except ImportError:
    HAS_SIMPLEAUDIO = False
    import winsound


class RSynthEngine:
    """Wrapper for rsynth synthesis with parameter control."""

    def __init__(self):
        self.sample_rate = 22050
        self.ms_per_frame = 10.0

        # Default parameters
        self.defaults = {
            # Basic controls
            'rate': 1.0,
            'pitch': 120.0,
            'gain': 57.0,
            # Voice quality
            'jitter': 1.5,
            'shimmer': 4.0,
            'flutter': 20,
            'voice_source': VOICE_NATURAL,
            'spectral_tilt': 0.0,   # dB
            'breathiness': 0.0,     # dB (Aturb analogue)
            'kopen_override': 0.0,  # samples at 4x rate; 0 = auto
            'flat_intonation': False,
            # Formants
            'F4hz': 3900.0,
            'B4hz': 400.0,
            'F5hz': 4700.0,
            'B5hz': 150.0,
            'F6hz': 4900.0,
            # Note: No B6hz - F6 only has parallel resonator in Klatt model
            # Nasal
            'FNPhz': 270.0,
            'BNhz': 500.0,
            'B1phz': 80.0,
            # Parallel formant bandwidths (frication path)
            'B4phz': 500.0,
            'B5phz': 600.0,
            'B6phz': 800.0,
            # F1/F2/F3 offset and scale (applied to phoneme values)
            'F1_offset': 0.0,
            'F1_scale': 100.0,  # Percentage
            'F2_offset': 0.0,
            'F2_scale': 100.0,
            'F3_offset': 0.0,
            'F3_scale': 100.0,
        }

        # Current parameters
        self.params = self.defaults.copy()

        # Synthesis objects
        self._init_synth()

    def _init_synth(self):
        """Initialize synthesis objects with current parameters."""
        self.speaker = Speaker(
            F0Hz=self.params['pitch'],
            Gain0=self.params['gain'],
            F4hz=self.params['F4hz'],
            B4hz=self.params['B4hz'],
            F5hz=self.params['F5hz'],
            B5hz=self.params['B5hz'],
            F6hz=self.params['F6hz'],
            FNPhz=self.params['FNPhz'],
            BNhz=self.params['BNhz'],
            B4phz=self.params['B4phz'],
            B5phz=self.params['B5phz'],
            B6phz=self.params['B6phz'],
            B1phz=self.params['B1phz'],
            F1_offset=self.params['F1_offset'],
            F1_scale=self.params['F1_scale'] / 100.0,
            F2_offset=self.params['F2_offset'],
            F2_scale=self.params['F2_scale'] / 100.0,
            F3_offset=self.params['F3_offset'],
            F3_scale=self.params['F3_scale'] / 100.0,
        )

        self.synth = KlattSynth(
            sample_rate=self.sample_rate,
            ms_per_frame=self.ms_per_frame,
            speaker=self.speaker,
            voice_source=self.params['voice_source'],
            kopen_override=int(self.params['kopen_override']) if self.params['kopen_override'] > 0 else None,
            tlt_db=self.params['spectral_tilt'],
            breathiness_db=self.params['breathiness']
        )

        self.frame_gen = FrameGenerator(
            sample_rate=self.sample_rate,
            f0_default=self.params['pitch']
        )

    def update_params(self, **kwargs):
        """Update synthesis parameters."""
        self.params.update(kwargs)

        # Update speaker
        self.speaker.F0Hz = self.params['pitch']
        self.speaker.Gain0 = self.params['gain']
        self.speaker.F4hz = self.params['F4hz']
        self.speaker.B4hz = self.params['B4hz']
        self.speaker.F5hz = self.params['F5hz']
        self.speaker.B5hz = self.params['B5hz']
        self.speaker.F6hz = self.params['F6hz']
        self.speaker.FNPhz = self.params['FNPhz']
        self.speaker.BNhz = self.params['BNhz']
        self.speaker.B1phz = self.params['B1phz']
        self.speaker.B4phz = self.params['B4phz']
        self.speaker.B5phz = self.params['B5phz']
        self.speaker.B6phz = self.params['B6phz']

        # Update F1/F2/F3 offset and scale
        self.speaker.F1_offset = self.params['F1_offset']
        self.speaker.F1_scale = self.params['F1_scale'] / 100.0  # Convert from % to multiplier
        self.speaker.F2_offset = self.params['F2_offset']
        self.speaker.F2_scale = self.params['F2_scale'] / 100.0
        self.speaker.F3_offset = self.params['F3_offset']
        self.speaker.F3_scale = self.params['F3_scale'] / 100.0

        # Update synth voice quality
        self.synth.jitter = self.params['jitter'] / 100.0
        self.synth.shimmer = self.params['shimmer'] / 100.0
        self.synth.flutter = int(self.params['flutter'])
        self.synth.voice_source = self.params['voice_source']
        self.synth.kopen_override = int(self.params['kopen_override']) if self.params['kopen_override'] > 0 else None
        self.synth.tlt_db = self.params['spectral_tilt']
        self.synth.breathiness_db = self.params['breathiness']

        # Update frame generator (invert rate so higher value = faster speech)
        self.frame_gen.speed = 1.0 / self.params['rate']
        self.frame_gen.f0_default = self.params['pitch']

    def reset_to_defaults(self):
        """Reset all parameters to defaults."""
        self.params = self.defaults.copy()
        self._init_synth()
        self.update_params()

    def synthesize(self, text):
        """
        Synthesize text to audio samples.

        Args:
            text: Text to synthesize

        Returns:
            bytes: Audio data as int16 samples
        """
        if not text.strip():
            return bytes()

        # Ensure parameters are applied
        self.update_params()

        # Convert text to phonemes
        phonemes = text_to_phonemes(text)

        # Convert to elements with F0 contour (invert rate so higher = faster)
        result = phonemes_to_elements(
            phonemes,
            speed=1.0 / self.params['rate'],
            f0_default=self.params['pitch'],
            flat_intonation=self.params['flat_intonation']
        )
        elements = result[0]
        f0_contour = result[1]
        # result[2] is word boundaries (optional, not used here)

        # Generate audio
        self.synth.reset()
        self.frame_gen.reset()

        audio = array.array('h')  # signed 16-bit

        for f0_hz, params in self.frame_gen.generate_frames(elements, f0_contour):
            frame_samples = self.synth.generate_frame(f0_hz, params)
            for sample in frame_samples:
                # Clip to int16 range
                sample = max(-32767, min(32767, int(sample)))
                audio.append(sample)

        return audio.tobytes()

    def save_wav(self, audio_bytes, filename):
        """Save audio bytes to a WAV file."""
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_bytes)


class SliderPanel(wx.Panel):
    """A labeled slider with value display."""

    def __init__(self, parent, label, min_val, max_val, default_val,
                 multiplier=1.0, unit="", decimal_places=1):
        super().__init__(parent)

        self.multiplier = multiplier
        self.unit = unit
        self.decimal_places = decimal_places

        # Convert to integer range for slider
        self.scale = 10 ** decimal_places
        self.min_int = int(min_val * self.scale)
        self.max_int = int(max_val * self.scale)
        self.default_int = int(default_val * self.scale)

        sizer = wx.BoxSizer(wx.HORIZONTAL)

        # Label
        label_text = wx.StaticText(self, label=label, size=(100, -1))
        sizer.Add(label_text, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)

        # Slider
        self.slider = wx.Slider(
            self, value=self.default_int,
            minValue=self.min_int, maxValue=self.max_int,
            style=wx.SL_HORIZONTAL
        )
        sizer.Add(self.slider, 1, wx.EXPAND | wx.RIGHT, 5)

        # Value display
        self.value_text = wx.StaticText(self, label="", size=(80, -1))
        sizer.Add(self.value_text, 0, wx.ALIGN_CENTER_VERTICAL)

        self.SetSizer(sizer)
        self._update_value_display()

        # Bind events
        self.slider.Bind(wx.EVT_SLIDER, self._on_slider)
        self.slider.Bind(wx.EVT_KEY_DOWN, self._on_key)

    def _on_slider(self, event):
        self._update_value_display()
        event.Skip()

    def _on_key(self, event):
        """Handle key press - D resets to default."""
        if event.GetKeyCode() == ord('D') or event.GetKeyCode() == ord('d'):
            self.reset()
        else:
            event.Skip()

    def _update_value_display(self):
        val = self.get_value()
        if self.decimal_places == 0:
            text = f"{int(val)}{self.unit}"
        else:
            text = f"{val:.{self.decimal_places}f}{self.unit}"
        self.value_text.SetLabel(text)

    def get_value(self):
        """Get the current slider value."""
        return self.slider.GetValue() / self.scale * self.multiplier

    def set_value(self, value):
        """Set the slider value."""
        int_val = int(value / self.multiplier * self.scale)
        int_val = max(self.min_int, min(self.max_int, int_val))
        self.slider.SetValue(int_val)
        self._update_value_display()

    def reset(self):
        """Reset to default value."""
        self.slider.SetValue(self.default_int)
        self._update_value_display()


class RSynthFrame(wx.Frame):
    """Main application window."""

    def __init__(self):
        super().__init__(
            None,
            title="RSynth Speech Synthesizer",
            size=(600, 750)
        )

        self.engine = RSynthEngine()
        self.play_obj = None
        self.is_playing = False
        self.synth_thread = None
        self.preset_dir = ""  # Remember last preset folder

        self._create_ui()
        self._bind_events()

        # Center on screen
        self.Centre()

    def _create_ui(self):
        """Create the user interface."""
        panel = wx.Panel(self)
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        # Text input
        text_box = wx.StaticBox(panel, label="Text to Synthesize")
        text_sizer = wx.StaticBoxSizer(text_box, wx.VERTICAL)

        self.text_ctrl = wx.TextCtrl(
            panel,
            value="Hello world. This is a test of the RSynth speech synthesizer.",
            style=wx.TE_MULTILINE,
            size=(-1, 80)
        )
        text_sizer.Add(self.text_ctrl, 1, wx.EXPAND | wx.ALL, 5)
        main_sizer.Add(text_sizer, 0, wx.EXPAND | wx.ALL, 10)

        # Buttons
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.play_btn = wx.Button(panel, label="Play")
        self.render_btn = wx.Button(panel, label="Render to WAV")
        self.stop_btn = wx.Button(panel, label="Stop")
        self.stop_btn.Enable(False)

        btn_sizer.Add(self.play_btn, 0, wx.RIGHT, 5)
        btn_sizer.Add(self.render_btn, 0, wx.RIGHT, 5)
        btn_sizer.Add(self.stop_btn, 0)

        main_sizer.Add(btn_sizer, 0, wx.ALL | wx.ALIGN_CENTER_HORIZONTAL, 10)

        # Scrolled window for sliders
        scroll = wx.ScrolledWindow(panel, style=wx.VSCROLL)
        scroll.SetScrollRate(0, 20)
        scroll_sizer = wx.BoxSizer(wx.VERTICAL)

        # Basic Controls
        basic_box = wx.StaticBox(scroll, label="Basic Controls")
        basic_sizer = wx.StaticBoxSizer(basic_box, wx.VERTICAL)

        self.sliders = {}

        self.sliders['rate'] = SliderPanel(
            scroll, "Rate:", 0.5, 2.0, 1.0,
            unit="x", decimal_places=2
        )
        basic_sizer.Add(self.sliders['rate'], 0, wx.EXPAND | wx.ALL, 2)

        self.sliders['pitch'] = SliderPanel(
            scroll, "Pitch:", 50, 300, 120,
            unit=" Hz", decimal_places=0
        )
        basic_sizer.Add(self.sliders['pitch'], 0, wx.EXPAND | wx.ALL, 2)

        self.sliders['gain'] = SliderPanel(
            scroll, "Gain:", 0, 70, 57,
            unit=" dB", decimal_places=0
        )
        basic_sizer.Add(self.sliders['gain'], 0, wx.EXPAND | wx.ALL, 2)

        scroll_sizer.Add(basic_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # Voice Quality
        voice_box = wx.StaticBox(scroll, label="Voice Quality")
        voice_sizer = wx.StaticBoxSizer(voice_box, wx.VERTICAL)

        self.sliders['jitter'] = SliderPanel(
            scroll, "Jitter:", 0, 10, 1.5,
            unit="%", decimal_places=1
        )
        voice_sizer.Add(self.sliders['jitter'], 0, wx.EXPAND | wx.ALL, 2)

        self.sliders['shimmer'] = SliderPanel(
            scroll, "Shimmer:", 0, 20, 4.0,
            unit="%", decimal_places=1
        )
        voice_sizer.Add(self.sliders['shimmer'], 0, wx.EXPAND | wx.ALL, 2)

        self.sliders['flutter'] = SliderPanel(
            scroll, "Flutter:", 0, 100, 20,
            unit="", decimal_places=0
        )
        voice_sizer.Add(self.sliders['flutter'], 0, wx.EXPAND | wx.ALL, 2)

        self.sliders['spectral_tilt'] = SliderPanel(
            scroll, "Spectral Tilt:", 0, 24, 0,
            unit=" dB", decimal_places=0
        )
        voice_sizer.Add(self.sliders['spectral_tilt'], 0, wx.EXPAND | wx.ALL, 2)

        self.sliders['breathiness'] = SliderPanel(
            scroll, "Breathiness:", 0, 40, 0,
            unit=" dB", decimal_places=0
        )
        voice_sizer.Add(self.sliders['breathiness'], 0, wx.EXPAND | wx.ALL, 2)

        self.sliders['kopen_override'] = SliderPanel(
            scroll, "Open Phase Override:", 0, 120, 0,
            unit=" samp@4x (0=auto)", decimal_places=0
        )
        voice_sizer.Add(self.sliders['kopen_override'], 0, wx.EXPAND | wx.ALL, 2)

        self.voice_source_radio = wx.RadioBox(
            scroll,
            label="Glottal Source",
            choices=["Natural (smooth)", "Impulsive (bright)"],
            majorDimension=1,
            style=wx.RA_VERTICAL
        )
        self.voice_source_radio.SetSelection(0)
        voice_sizer.Add(self.voice_source_radio, 0, wx.EXPAND | wx.ALL, 5)

        # Flat intonation checkbox for robotic monotone voice
        self.flat_intonation_cb = wx.CheckBox(scroll, label="Flat Intonation (robotic)")
        voice_sizer.Add(self.flat_intonation_cb, 0, wx.EXPAND | wx.ALL, 5)

        scroll_sizer.Add(voice_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # Formant Parameters
        formant_box = wx.StaticBox(scroll, label="Formant Parameters")
        formant_sizer = wx.StaticBoxSizer(formant_box, wx.VERTICAL)

        # F1/F2/F3 offset and scale controls (applied to phoneme values)
        self.sliders['F1_offset'] = SliderPanel(
            scroll, "F1 Offset:", -500, 500, 0,
            unit=" Hz", decimal_places=0
        )
        formant_sizer.Add(self.sliders['F1_offset'], 0, wx.EXPAND | wx.ALL, 2)

        self.sliders['F1_scale'] = SliderPanel(
            scroll, "F1 Scale:", 50, 150, 100,
            unit="%", decimal_places=0
        )
        formant_sizer.Add(self.sliders['F1_scale'], 0, wx.EXPAND | wx.ALL, 2)

        self.sliders['F2_offset'] = SliderPanel(
            scroll, "F2 Offset:", -500, 500, 0,
            unit=" Hz", decimal_places=0
        )
        formant_sizer.Add(self.sliders['F2_offset'], 0, wx.EXPAND | wx.ALL, 2)

        self.sliders['F2_scale'] = SliderPanel(
            scroll, "F2 Scale:", 50, 150, 100,
            unit="%", decimal_places=0
        )
        formant_sizer.Add(self.sliders['F2_scale'], 0, wx.EXPAND | wx.ALL, 2)

        self.sliders['F3_offset'] = SliderPanel(
            scroll, "F3 Offset:", -500, 500, 0,
            unit=" Hz", decimal_places=0
        )
        formant_sizer.Add(self.sliders['F3_offset'], 0, wx.EXPAND | wx.ALL, 2)

        self.sliders['F3_scale'] = SliderPanel(
            scroll, "F3 Scale:", 50, 150, 100,
            unit="%", decimal_places=0
        )
        formant_sizer.Add(self.sliders['F3_scale'], 0, wx.EXPAND | wx.ALL, 2)

        self.sliders['F4hz'] = SliderPanel(
            scroll, "F4 Freq:", 2500, 5000, 3900,
            unit=" Hz", decimal_places=0
        )
        formant_sizer.Add(self.sliders['F4hz'], 0, wx.EXPAND | wx.ALL, 2)

        self.sliders['B4hz'] = SliderPanel(
            scroll, "F4 BW:", 100, 1000, 400,
            unit=" Hz", decimal_places=0
        )
        formant_sizer.Add(self.sliders['B4hz'], 0, wx.EXPAND | wx.ALL, 2)

        self.sliders['F5hz'] = SliderPanel(
            scroll, "F5 Freq:", 3000, 6000, 4700,
            unit=" Hz", decimal_places=0
        )
        formant_sizer.Add(self.sliders['F5hz'], 0, wx.EXPAND | wx.ALL, 2)

        self.sliders['B5hz'] = SliderPanel(
            scroll, "F5 BW:", 50, 600, 150,
            unit=" Hz", decimal_places=0
        )
        formant_sizer.Add(self.sliders['B5hz'], 0, wx.EXPAND | wx.ALL, 2)

        self.sliders['F6hz'] = SliderPanel(
            scroll, "F6 Freq:", 4000, 7000, 4900,
            unit=" Hz", decimal_places=0
        )
        formant_sizer.Add(self.sliders['F6hz'], 0, wx.EXPAND | wx.ALL, 2)

        self.sliders['B6phz'] = SliderPanel(
            scroll, "F6 BW:", 100, 1500, 800,
            unit=" Hz", decimal_places=0
        )
        formant_sizer.Add(self.sliders['B6phz'], 0, wx.EXPAND | wx.ALL, 2)

        # Parallel formant bandwidths for F4/F5 (frication path)
        self.sliders['B4phz'] = SliderPanel(
            scroll, "F4 Par BW:", 100, 1000, 500,
            unit=" Hz", decimal_places=0
        )
        formant_sizer.Add(self.sliders['B4phz'], 0, wx.EXPAND | wx.ALL, 2)

        self.sliders['B5phz'] = SliderPanel(
            scroll, "F5 Par BW:", 100, 1200, 600,
            unit=" Hz", decimal_places=0
        )
        formant_sizer.Add(self.sliders['B5phz'], 0, wx.EXPAND | wx.ALL, 2)

        scroll_sizer.Add(formant_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # Nasal Parameters
        nasal_box = wx.StaticBox(scroll, label="Nasal Parameters")
        nasal_sizer = wx.StaticBoxSizer(nasal_box, wx.VERTICAL)

        self.sliders['FNPhz'] = SliderPanel(
            scroll, "Nasal Freq:", 200, 600, 270,
            unit=" Hz", decimal_places=0
        )
        nasal_sizer.Add(self.sliders['FNPhz'], 0, wx.EXPAND | wx.ALL, 2)

        self.sliders['BNhz'] = SliderPanel(
            scroll, "Nasal BW:", 100, 1000, 500,
            unit=" Hz", decimal_places=0
        )
        nasal_sizer.Add(self.sliders['BNhz'], 0, wx.EXPAND | wx.ALL, 2)

        self.sliders['B1phz'] = SliderPanel(
            scroll, "Parallel F1 BW:", 40, 400, 80,
            unit=" Hz", decimal_places=0
        )
        nasal_sizer.Add(self.sliders['B1phz'], 0, wx.EXPAND | wx.ALL, 2)

        scroll_sizer.Add(nasal_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # Preset buttons
        preset_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.save_preset_btn = wx.Button(scroll, label="Save Preset")
        self.load_preset_btn = wx.Button(scroll, label="Load Preset")
        preset_sizer.Add(self.save_preset_btn, 0, wx.RIGHT, 5)
        preset_sizer.Add(self.load_preset_btn, 0)
        scroll_sizer.Add(preset_sizer, 0, wx.ALL | wx.ALIGN_CENTER_HORIZONTAL, 5)

        # Reset button
        self.reset_btn = wx.Button(scroll, label="Reset to Defaults")
        scroll_sizer.Add(self.reset_btn, 0, wx.ALL | wx.ALIGN_CENTER_HORIZONTAL, 10)

        scroll.SetSizer(scroll_sizer)
        main_sizer.Add(scroll, 1, wx.EXPAND)

        # Status bar
        self.CreateStatusBar()
        self.SetStatusText("Ready")

        panel.SetSizer(main_sizer)

    def _bind_events(self):
        """Bind event handlers."""
        self.play_btn.Bind(wx.EVT_BUTTON, self.on_play)
        self.render_btn.Bind(wx.EVT_BUTTON, self.on_render)
        self.stop_btn.Bind(wx.EVT_BUTTON, self.on_stop)
        self.reset_btn.Bind(wx.EVT_BUTTON, self.on_reset)
        self.save_preset_btn.Bind(wx.EVT_BUTTON, self.on_save_preset)
        self.load_preset_btn.Bind(wx.EVT_BUTTON, self.on_load_preset)
        self.Bind(wx.EVT_CLOSE, self.on_close)

        # Add keyboard shortcuts (Alt+P=Play, Alt+R=Render, Alt+D=Reset, Ctrl+S=Save, Ctrl+O=Open)
        play_id = wx.NewIdRef()
        render_id = wx.NewIdRef()
        reset_id = wx.NewIdRef()
        save_preset_id = wx.NewIdRef()
        load_preset_id = wx.NewIdRef()
        self.Bind(wx.EVT_MENU, self.on_play, id=play_id)
        self.Bind(wx.EVT_MENU, self.on_render, id=render_id)
        self.Bind(wx.EVT_MENU, self.on_reset, id=reset_id)
        self.Bind(wx.EVT_MENU, self.on_save_preset, id=save_preset_id)
        self.Bind(wx.EVT_MENU, self.on_load_preset, id=load_preset_id)
        accel_tbl = wx.AcceleratorTable([
            (wx.ACCEL_ALT, ord('P'), play_id),
            (wx.ACCEL_ALT, ord('R'), render_id),
            (wx.ACCEL_ALT, ord('D'), reset_id),
            (wx.ACCEL_CTRL, ord('S'), save_preset_id),
            (wx.ACCEL_CTRL, ord('O'), load_preset_id),
        ])
        self.SetAcceleratorTable(accel_tbl)

    def _get_params_from_sliders(self):
        """Get current parameter values from all sliders and checkboxes."""
        params = {name: slider.get_value() for name, slider in self.sliders.items()}
        params['flat_intonation'] = self.flat_intonation_cb.GetValue()
        params['voice_source'] = (
            VOICE_NATURAL
            if self.voice_source_radio.GetSelection() == 0
            else VOICE_IMPULSIVE
        )
        if params['kopen_override'] <= 0:
            params['kopen_override'] = 0.0
        return params

    def _synthesize(self, text):
        """Synthesize text with current parameters."""
        params = self._get_params_from_sliders()
        self.engine.update_params(**params)
        return self.engine.synthesize(text)

    def on_play(self, event):
        """Handle Play button click."""
        text = self.text_ctrl.GetValue()
        if not text.strip():
            self.SetStatusText("No text to synthesize")
            return

        self.play_btn.Enable(False)
        self.render_btn.Enable(False)
        self.stop_btn.Enable(True)
        self.SetStatusText("Synthesizing...")

        # Run synthesis in background thread
        def synth_and_play():
            try:
                audio_bytes = self._synthesize(text)

                if not audio_bytes:
                    wx.CallAfter(self._on_play_done, "No audio generated")
                    return

                # Play audio
                if HAS_SIMPLEAUDIO:
                    self.play_obj = sa.play_buffer(
                        audio_bytes,
                        num_channels=1,
                        bytes_per_sample=2,
                        sample_rate=self.engine.sample_rate
                    )
                    self.is_playing = True
                    wx.CallAfter(self.SetStatusText, "Playing...")
                    self.play_obj.wait_done()
                    self.is_playing = False
                    wx.CallAfter(self._on_play_done, "Playback finished")
                else:
                    # Fall back to winsound (blocking, saves to temp file)
                    wx.CallAfter(self.SetStatusText, "Playing...")
                    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                    temp_file.close()
                    self.engine.save_wav(audio_bytes, temp_file.name)
                    winsound.PlaySound(temp_file.name, winsound.SND_FILENAME)
                    os.unlink(temp_file.name)
                    wx.CallAfter(self._on_play_done, "Playback finished")

            except Exception as e:
                wx.CallAfter(self._on_play_done, f"Error: {str(e)}")

        self.synth_thread = threading.Thread(target=synth_and_play, daemon=True)
        self.synth_thread.start()

    def _on_play_done(self, message):
        """Called when playback is complete."""
        self.play_btn.Enable(True)
        self.render_btn.Enable(True)
        self.stop_btn.Enable(False)
        self.SetStatusText(message)

    def on_render(self, event):
        """Handle Render to WAV button click."""
        text = self.text_ctrl.GetValue()
        if not text.strip():
            self.SetStatusText("No text to synthesize")
            return

        # Ask for save location
        with wx.FileDialog(
            self,
            "Save WAV file",
            wildcard="WAV files (*.wav)|*.wav",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT
        ) as dialog:
            if dialog.ShowModal() == wx.ID_CANCEL:
                return

            filename = dialog.GetPath()

        self.SetStatusText("Rendering...")
        self.play_btn.Enable(False)
        self.render_btn.Enable(False)

        def render_thread():
            try:
                audio_bytes = self._synthesize(text)

                if not audio_bytes:
                    wx.CallAfter(self._on_render_done, "No audio generated", None)
                    return

                self.engine.save_wav(audio_bytes, filename)
                duration = len(audio_bytes) / 2 / self.engine.sample_rate
                wx.CallAfter(
                    self._on_render_done,
                    f"Saved {duration:.2f}s to {os.path.basename(filename)}",
                    filename
                )

            except Exception as e:
                wx.CallAfter(self._on_render_done, f"Error: {str(e)}", None)

        threading.Thread(target=render_thread, daemon=True).start()

    def _on_render_done(self, message, filename):
        """Called when rendering is complete."""
        self.play_btn.Enable(True)
        self.render_btn.Enable(True)
        self.SetStatusText(message)

    def on_stop(self, event):
        """Handle Stop button click."""
        if HAS_SIMPLEAUDIO and self.play_obj and self.is_playing:
            self.play_obj.stop()
            self.is_playing = False
        self._on_play_done("Stopped")

    def on_reset(self, event):
        """Handle Reset button click."""
        self.engine.reset_to_defaults()
        for name, slider in self.sliders.items():
            slider.set_value(self.engine.defaults[name])
        self.flat_intonation_cb.SetValue(False)
        if self.engine.defaults['voice_source'] == VOICE_IMPULSIVE:
            self.voice_source_radio.SetSelection(1)
        else:
            self.voice_source_radio.SetSelection(0)
        self.SetStatusText("Parameters reset to defaults")

    def on_save_preset(self, event):
        """Handle Save Preset button click."""
        # Get current parameters
        params = self._get_params_from_sliders()

        # Ask for save location
        with wx.FileDialog(
            self,
            "Save Preset",
            defaultDir=self.preset_dir,
            wildcard="JSON files (*.json)|*.json",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT
        ) as dialog:
            if dialog.ShowModal() == wx.ID_CANCEL:
                return

            filename = dialog.GetPath()

        try:
            with open(filename, 'w') as f:
                json.dump(params, f, indent=2)
            self.preset_dir = os.path.dirname(filename)
            self.SetStatusText(f"Preset saved to {os.path.basename(filename)}")
        except Exception as e:
            wx.MessageBox(f"Error saving preset: {str(e)}", "Error", wx.OK | wx.ICON_ERROR)

    def on_load_preset(self, event):
        """Handle Load Preset button click."""
        # Ask for file location
        with wx.FileDialog(
            self,
            "Load Preset",
            defaultDir=self.preset_dir,
            wildcard="JSON files (*.json)|*.json",
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
        ) as dialog:
            if dialog.ShowModal() == wx.ID_CANCEL:
                return

            filename = dialog.GetPath()

        try:
            with open(filename, 'r') as f:
                params = json.load(f)

            # Apply loaded values to sliders
            for name, value in params.items():
                if name in self.sliders:
                    self.sliders[name].set_value(value)
                elif name == 'flat_intonation':
                    self.flat_intonation_cb.SetValue(bool(value))
                elif name == 'voice_source':
                    if value == VOICE_IMPULSIVE:
                        self.voice_source_radio.SetSelection(1)
                    else:
                        self.voice_source_radio.SetSelection(0)

            # Update engine
            self.engine.update_params(**params)
            self.preset_dir = os.path.dirname(filename)
            self.SetStatusText(f"Preset loaded from {os.path.basename(filename)}")
        except Exception as e:
            wx.MessageBox(f"Error loading preset: {str(e)}", "Error", wx.OK | wx.ICON_ERROR)

    def on_close(self, event):
        """Handle window close."""
        if HAS_SIMPLEAUDIO and self.play_obj and self.is_playing:
            self.play_obj.stop()
        self.Destroy()


class RSynthApp(wx.App):
    """Main application class."""

    def OnInit(self):
        frame = RSynthFrame()
        frame.Show()
        return True


def main():
    """Main entry point."""
    app = RSynthApp()
    app.MainLoop()


if __name__ == '__main__':
    main()
