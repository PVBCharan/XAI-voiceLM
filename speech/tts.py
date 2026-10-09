"""Text-to-Speech (TTS) synthesizer for generating spoken voice responses."""

import sys
import threading
from config.settings import settings
from utils.logger import logger


class TextToSpeech:
    """Synthesizes text into spoken audio output supporting pyttsx3 and Windows SAPI."""

    def __init__(
        self,
        engine_type: str = settings.speech.tts_engine,
        rate: int = settings.speech.tts_rate,
        voice: str = settings.speech.tts_voice,
    ):
        self.engine_type = engine_type.lower()
        self.rate = rate
        self.voice = voice
        self._engine = None
        self._sapi_voice = None
        self._lock = threading.Lock()
        self._init_engine()

    def _init_engine(self):
        """Initializes TTS engine with automatic fallback cascade."""
        # 1. Try pyttsx3 if preferred or auto
        if self.engine_type in ("pyttsx3", "auto"):
            try:
                import pyttsx3
                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", self.rate)
                logger.info("pyttsx3 TTS engine initialized.")
                return
            except ImportError:
                pass
            except Exception as e:
                logger.debug(f"pyttsx3 init failed: {e}")

        # 2. Try Windows native SAPI (comtypes or win32com)
        if sys.platform == "win32":
            try:
                import comtypes.client
                self._sapi_voice = comtypes.client.CreateObject("SAPI.SpVoice")
                logger.info("Windows SAPI.SpVoice initialized.")
                return
            except Exception as e:
                logger.debug(f"comtypes SAPI init failed: {e}")

            try:
                import win32com.client
                self._sapi_voice = win32com.client.Dispatch("SAPI.SpVoice")
                logger.info("Windows win32com SAPI.SpVoice initialized.")
                return
            except Exception as e:
                logger.debug(f"win32com SAPI init failed: {e}")

        logger.info("No audio synthesizer available. Spoken responses will display on console.")

    def speak(self, text: str) -> None:
        """Speaks the text synchronously."""
        if not text:
            return

        logger.info(f"Assistant speaking: '{text}'")

        with self._lock:
            # Try pyttsx3
            if self._engine is not None:
                try:
                    self._engine.say(text)
                    self._engine.runAndWait()
                    return
                except Exception as e:
                    logger.error(f"pyttsx3 playback error: {e}")

            # Try Windows SAPI
            if self._sapi_voice is not None:
                try:
                    self._sapi_voice.Speak(text)
                    return
                except Exception as e:
                    logger.error(f"SAPI playback error: {e}")

            # Fallback console display
            print(f"\n[ASSISTANT VOICE]: {text}\n")
