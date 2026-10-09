"""Wake word detection for triggering assistant activation."""

import numpy as np
from config.settings import settings
from utils.logger import logger


class WakeWordDetector:
    """Detects specified wake words (e.g. 'Jarvis', 'Hey Assistant') in streaming audio."""

    def __init__(
        self,
        keywords: list[str] = settings.wakeword.keywords,
        sensitivity: float = settings.wakeword.sensitivity,
        sample_rate: int = settings.audio.sample_rate,
    ):
        self.keywords = [k.lower() for k in keywords]
        self.sensitivity = sensitivity
        self.sample_rate = sample_rate
        self._engine = None
        self._init_engine()

    def _init_engine(self):
        """Initializes wake word engine (e.g. openWakeWord, Porcupine, or acoustic model)."""
        logger.info(f"Wake word detector configured for keywords: {self.keywords}")

    def process_frame(self, audio_frame: np.ndarray) -> str | None:
        """Analyzes an audio frame to check if any wake word was triggered.

        Returns:
            The matched wake word string if detected, otherwise None.
        """
        if audio_frame is None or len(audio_frame) == 0:
            return None

        # Placeholder: Return None during passive listening until speech activation
        return None

    def match_transcript(self, text: str) -> str | None:
        """Helper to match wake word directly from immediate transcribed text."""
        lowered = text.lower().strip()
        for kw in self.keywords:
            if kw in lowered:
                return kw
        return None
