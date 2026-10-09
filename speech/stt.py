"""Speech-to-Text (STT) transcription using Whisper models."""

import numpy as np
from config.settings import settings
from utils.logger import logger


class SpeechToText:
    """Transcribes audio waveform arrays or files into text."""

    def __init__(
        self,
        model_size: str = settings.speech.stt_model_size,
        device: str = settings.speech.stt_device,
    ):
        self.model_size = model_size
        self.device = device
        self._faster_model = None
        self._openai_whisper = None
        self._init_model()

    def _init_model(self):
        """Attempts to load faster-whisper or standard whisper."""
        try:
            from faster_whisper import WhisperModel
            self._faster_model = WhisperModel(self.model_size, device=self.device, compute_type="int8")
            logger.info(f"Loaded faster-whisper model '{self.model_size}' on {self.device}.")
            return
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Error loading faster-whisper: {e}")

        try:
            import whisper
            self._openai_whisper = whisper.load_model(self.model_size, device=self.device)
            logger.info(f"Loaded OpenAI Whisper model '{self.model_size}' on {self.device}.")
            return
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Error loading OpenAI whisper: {e}")

        logger.info("Neither faster-whisper nor whisper is installed. STT running in mock mode.")

    def transcribe(self, audio: np.ndarray, language: str = "en") -> str:
        """Transcribes float32 or int16 numpy audio array to text."""
        if audio is None or len(audio) == 0:
            return ""

        # Flatten multi-channel to 1D mono
        audio_mono = np.squeeze(audio)
        if audio_mono.ndim > 1:
            audio_mono = audio_mono.mean(axis=1)

        # Normalize audio to float32 between -1.0 and 1.0
        if audio_mono.dtype == np.int16:
            audio_float = audio_mono.astype(np.float32) / 32768.0
        elif audio_mono.dtype == np.int32:
            audio_float = audio_mono.astype(np.float32) / 2147483648.0
        else:
            audio_float = audio_mono.astype(np.float32)

        # 1. Faster-Whisper
        if self._faster_model is not None:
            try:
                segments, _ = self._faster_model.transcribe(audio_float, language=language, beam_size=5)
                text = " ".join([segment.text for segment in segments]).strip()
                logger.info(f"STT Transcript: '{text}'")
                return text
            except Exception as e:
                logger.error(f"faster-whisper transcription error: {e}")
                return ""

        # 2. OpenAI Whisper
        if self._openai_whisper is not None:
            try:
                result = self._openai_whisper.transcribe(audio_float, language=language)
                text = result.get("text", "").strip()
                logger.info(f"STT Transcript: '{text}'")
                return text
            except Exception as e:
                logger.error(f"whisper transcription error: {e}")
                return ""

        logger.info("STT: No speech recognition engine available. Run 'pip install faster-whisper'.")
        return ""
