"""Centralized configuration and environment settings for the AI Voice Assistant."""

from dataclasses import dataclass, field
import os
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent


def _load_env_file(env_path: Path) -> None:
    """Loads key-value pairs from .env into os.environ without requiring external packages."""
    if not env_path.exists():
        return

    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
    except ImportError:
        # Fallback pure-Python parser for .env
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    if key and key not in os.environ:
                        os.environ[key] = val
        except Exception:
            pass


# Automatically load .env
_load_env_file(BASE_DIR / ".env")


@dataclass
class AudioConfig:
    sample_rate: int = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
    channels: int = int(os.getenv("AUDIO_CHANNELS", "1"))
    chunk_size: int = int(os.getenv("AUDIO_CHUNK_SIZE", "1024"))
    format_bytes: int = int(os.getenv("AUDIO_FORMAT_BYTES", "2"))  # 16-bit PCM = 2 bytes
    vad_aggressiveness: int = int(os.getenv("VAD_AGGRESSIVENESS", "3"))  # 0 to 3
    silence_timeout_sec: float = float(os.getenv("SILENCE_TIMEOUT_SEC", "1.5"))


@dataclass
class WakewordConfig:
    enabled: bool = os.getenv("WAKEWORD_ENABLED", "True").lower() == "true"
    keywords: list[str] = field(
        default_factory=lambda: [
            w.strip().lower()
            for w in os.getenv("WAKE_WORDS", "jarvis,hey assistant").split(",")
            if w.strip()
        ]
    )
    sensitivity: float = float(os.getenv("WAKEWORD_SENSITIVITY", "0.5"))


@dataclass
class SpeechConfig:
    stt_model_size: str = os.getenv("STT_MODEL_SIZE", "base")
    stt_device: str = os.getenv("STT_DEVICE", "cpu")
    tts_engine: str = os.getenv("TTS_ENGINE", "auto")  # auto, pyttsx3, sapi, or print
    tts_voice: str = os.getenv("TTS_VOICE", "default")
    tts_rate: int = int(os.getenv("TTS_RATE", "180"))


@dataclass
class IntelligenceConfig:
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "gemini-1.5-flash")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    system_prompt: str = os.getenv(
        "SYSTEM_PROMPT",
        "You are an intelligent, friendly, and concise voice assistant. "
        "Keep responses conversational, natural, and brief for spoken dialogue.",
    )
    max_history_turns: int = int(os.getenv("MAX_HISTORY_TURNS", "10"))


@dataclass
class SpeakerConfig:
    verification_threshold: float = float(os.getenv("SPEAKER_VERIFICATION_THRESHOLD", "0.75"))
    embedding_dim: int = int(os.getenv("SPEAKER_EMBEDDING_DIM", "192"))
    storage_dir: Path = BASE_DIR / "data" / "speakers"


@dataclass
class EmotionConfig:
    enabled: bool = os.getenv("EMOTION_DETECTION_ENABLED", "True").lower() == "true"
    confidence_threshold: float = float(os.getenv("EMOTION_CONFIDENCE_THRESHOLD", "0.6"))


@dataclass
class DatabaseConfig:
    db_path: Path = BASE_DIR / os.getenv("DATABASE_PATH", "data/assistant.db")


@dataclass
class Settings:
    audio: AudioConfig = field(default_factory=AudioConfig)
    wakeword: WakewordConfig = field(default_factory=WakewordConfig)
    speech: SpeechConfig = field(default_factory=SpeechConfig)
    intelligence: IntelligenceConfig = field(default_factory=IntelligenceConfig)
    speaker: SpeakerConfig = field(default_factory=SpeakerConfig)
    emotion: EmotionConfig = field(default_factory=EmotionConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    base_dir: Path = BASE_DIR


# Global settings instance
settings = Settings()
