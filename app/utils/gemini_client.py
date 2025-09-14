import logging

import numpy as np
from scipy import signal
from google import genai
from google.genai import types

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


class GeminiClient:
    """Client for interacting with Gemini Live API"""
    
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash-preview-native-audio-dialog", 
                 system_message: str = "You are a friendly and helpful AI voice assistant on a phone call. Be concise and clear."):
        self.api_key = api_key
        self.model = model
        self.system_message = system_message
        # Initialize client with API key
        self.client = genai.Client(api_key=api_key)
        self.audio_resampler = AudioResampler()
        self.audio_chunk_count = 5  # Number of audio chunks to buffer before sending to Gemini
        
    
    async def send_audio_to_gemini(self, session, audio_data: bytes):
        """
        Send audio data to Gemini Live session.
        
        Args:
            session: Gemini Live session
            audio_data: Raw PCM audio data (16kHz from Teler)
        """
        try:
            # Teler now sends 16kHz audio, which matches Gemini's input requirement
            # No resampling needed for input
            await session.send_realtime_input(
                audio=types.Blob(data=audio_data, mime_type="audio/pcm;rate=16000")
            )
            logger.debug("Sent PCM16 audio to Gemini (16kHz)")
            
        except Exception as e:
            logger.error(f"Error sending audio to Gemini: {e}")
            raise
    
    async def receive_audio_from_gemini(self, session):
        """
        Receive audio data from Gemini Live session.
        
        Args:
            session: Gemini Live session
            
        Yields:
            Audio data chunks from Gemini
        """
        try:
            async for response in session.receive():
                if response.data is not None:
                    # Gemini sends audio data (24kHz output)
                    logger.debug(f"Received audio chunk from Gemini (size: {len(response.data)} bytes)")
                    yield response.data
                
                # Handle other response types if needed
                if response.server_content and response.server_content.model_turn is not None:
                    logger.debug("Model response metadata received")
                    
        except Exception as e:
            logger.error(f"Error receiving audio from Gemini: {e}")
            raise
