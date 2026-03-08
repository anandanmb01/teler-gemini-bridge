import logging

import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)

TELER_OUTPUT_RATE = 8000
TELER_INPUT_RATE = 16000


class AudioResampler:
    def downsample(self, audio_data: bytes, source_rate: int) -> bytes:
        """Downsample PCM audio to Teler's required 8kHz."""
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            factor = source_rate // TELER_OUTPUT_RATE
            downsampled = signal.decimate(audio_array, q=factor, n=8, ftype='iir')
            return downsampled.astype(np.int16).tobytes()
        except Exception as e:
            logger.error(f"Error downsampling audio: {e}")
            return audio_data

    def upsample(self, audio_data: bytes, target_rate: int) -> bytes:
        """Upsample PCM audio from Teler's 16kHz to target rate."""
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            factor = target_rate / TELER_INPUT_RATE
            upsampled = signal.resample(audio_array, int(len(audio_array) * factor))
            return upsampled.astype(np.int16).tobytes()
        except Exception as e:
            logger.error(f"Error upsampling audio: {e}")
            return audio_data
