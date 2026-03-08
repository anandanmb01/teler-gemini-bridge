import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # Gemini Configuration
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-preview-native-audio-dialog")
    gemini_system_message: str = os.getenv("GEMINI_SYSTEM_MESSAGE", "You are a friendly and helpful AI voice assistant on a phone call. Be concise and clear.")
    gemini_initial_prompt: str = os.getenv("GEMINI_INITIAL_PROMPT", "hello who is this")
    gemini_audio_chunk_count: int = int(os.getenv("GEMINI_AUDIO_CHUNK_COUNT", "5"))

    # Server Configuration
    server_domain: str = os.getenv("SERVER_DOMAIN", "")

    # Teler Configuration
    teler_api_key: str = os.getenv("TELER_API_KEY", "")

    class Config:
        env_file = ".env"
        case_sensitive = False

# Create settings instance
settings = Settings()
