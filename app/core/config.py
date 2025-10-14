import os

from pydantic_settings import BaseSettings
from app.utils.ngrok_utils import get_server_domain


class Settings(BaseSettings):
    """Application settings"""
    
    # Gemini Configuration
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-preview-native-audio-dialog")
    gemini_system_message: str = os.getenv("GEMINI_SYSTEM_MESSAGE", "You are a friendly and helpful AI voice assistant on a phone call. Be concise and clear.")
    gemini_initial_prompt: str = os.getenv("GEMINI_INITIAL_PROMPT", "hello who is this")
    gemini_audio_chunk_count: int = int(os.getenv("GEMINI_AUDIO_CHUNK_COUNT", "5"))
    
    # Server Configuration - dynamically get ngrok URL
    @property
    def server_domain(self) -> str:
        return get_server_domain()
    
    server_host: str = os.getenv("SERVER_HOST", "0.0.0.0")
    server_port: int = int(os.getenv("SERVER_PORT", "8000"))
    
    # Teler Configuration
    teler_api_key: str = os.getenv("TELER_API_KEY", "")
    from_number: str = os.getenv("FROM_NUMBER", "+91123*******")
    to_number: str = os.getenv("TO_NUMBER", "+91456*******")
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Create settings instance
settings = Settings()
