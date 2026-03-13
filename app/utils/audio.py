import logging

import numpy as np
from scipy.signal import butter, resample_poly, sosfilt, sosfilt_zi

logger = logging.getLogger(__name__)

TELER_SAMPLE_RATE = 8000    # rate we send to Teler (output)
GEMINI_SAMPLE_RATE = 24000  # rate Gemini outputs audio
TELER_INPUT_RATE = 16000    # rate Teler sends to us (input)

_DOWNSAMPLE_FACTOR = GEMINI_SAMPLE_RATE // TELER_SAMPLE_RATE  # 3
_UPSAMPLE_FACTOR_UP = GEMINI_SAMPLE_RATE // 8000              # 3  (16k * 3/2 = 24k)
_UPSAMPLE_FACTOR_DOWN = TELER_INPUT_RATE // 8000              # 2


class AudioResampler:
    """Stateful per-session audio resampler.

    Must be instantiated once per call session so IIR filter state is
    preserved across chunks, eliminating boundary transients.
    """

    def __init__(self):
        # Pre-compute anti-aliasing filter for 24kHz -> 8kHz (factor 3).
        # Cutoff at Nyquist of target prevents aliasing; order-8 gives
        # ~48 dB/octave roll-off with minimal passband ripple.
        self._sos = butter(8, 1.0 / _DOWNSAMPLE_FACTOR, output='sos')
        self._zi = sosfilt_zi(self._sos) * 0.0  # zero initial conditions

    def downsample(self, pcm: bytes) -> bytes:
        """Downsample Gemini's 24kHz PCM to Teler's 8kHz."""
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        filtered, self._zi = sosfilt(self._sos, samples, zi=self._zi)
        return filtered[::_DOWNSAMPLE_FACTOR].astype(np.int16).tobytes()

    def upsample(self, pcm: bytes) -> bytes:
        """Upsample Teler's 16kHz PCM to Gemini's 24kHz."""
        samples = np.frombuffer(pcm, dtype=np.int16)
        return resample_poly(samples, _UPSAMPLE_FACTOR_UP, _UPSAMPLE_FACTOR_DOWN).astype(np.int16).tobytes()
