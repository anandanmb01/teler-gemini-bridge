import logging

import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)


class AudioResampler:
    """
    High-quality audio resampling for real-time communication between Gemini (16kHz/24kHz) and Teler (16kHz→8kHz).
    """

    def __init__(self):
        self.gemini_input_sample_rate = 16000  # Gemini expects 16kHz input
        self.gemini_output_sample_rate = 24000  # Gemini outputs 24kHz
        self.teler_input_sample_rate = 16000  # Teler streams to us at 16kHz
        self.teler_output_sample_rate = 8000   # Teler requires 8kHz from us

    def downsample(self, audio_data: bytes, source_rate: int) -> bytes:
        """
        Downsample audio to Teler's required 8kHz.

        Args:
            audio_data: Raw PCM audio data as bytes
            source_rate: Source sample rate

        Returns:
            Downsampled audio data as bytes (pcm_s16le format at 8kHz)
        """
        try:
            # Convert bytes to numpy array (pcm_s16le format)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)

            # Calculate downsampling factor
            factor = source_rate // self.teler_output_sample_rate

            # Use scipy.decimate for high-quality downsampling
            downsampled_array = signal.decimate(
                audio_array, 
                q=factor,
                n=8,
                ftype='iir'
            )

            # Convert back to 16-bit PCM
            downsampled_int16 = downsampled_array.astype(np.int16)

            return downsampled_int16.tobytes()

        except Exception as e:
            logger.error(f"Error downsampling audio: {e}")
            return audio_data

    def upsample(self, audio_data: bytes, target_rate: int) -> bytes:
        """
        Upsample audio from Teler's 16kHz input to target rate.

        Args:
            audio_data: Raw PCM audio data as bytes
            target_rate: Target sample rate

        Returns:
            Upsampled audio data as bytes
        """
        try:
            # Convert bytes to numpy array (pcm_s16le format)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)

            # Calculate upsampling factor
            factor = target_rate / self.teler_input_sample_rate

            # Use scipy.resample for upsampling
            upsampled_array = signal.resample(
                audio_array, 
                int(len(audio_array) * factor)
            )

            # Convert back to 16-bit PCM
            upsampled_int16 = upsampled_array.astype(np.int16)

            return upsampled_int16.tobytes()

        except Exception as e:
            logger.error(f"Error upsampling audio: {e}")
            return audio_data