"""Voice Activity Detection (VAD) for detecting speech and silence endpoints."""

import numpy as np
from config.settings import settings
from utils.logger import logger


class VoiceActivityDetector:
    """Detects speech activity using WebRTC VAD with an energy-based fallback."""

    def __init__(
        self,
        sample_rate: int = settings.audio.sample_rate,
        aggressiveness: int = settings.audio.vad_aggressiveness,
        energy_threshold: float = 0.015,
    ):
        self.sample_rate = sample_rate
        self.aggressiveness = aggressiveness
        self.energy_threshold = energy_threshold
        self._vad = None

        try:
            import webrtcvad
            self._vad = webrtcvad.Vad(self.aggressiveness)
            logger.info(f"WebRTC VAD initialized (aggressiveness {self.aggressiveness}).")
        except ImportError:
            logger.warning("webrtcvad is not installed. Using energy-based fallback VAD.")

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Determines if the provided audio chunk contains speech."""
        if audio_chunk is None or len(audio_chunk) == 0:
            return False

        # Flatten to 1D mono
        audio = np.squeeze(audio_chunk)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # 1. If WebRTC VAD is available, process in 30ms slices (480 samples at 16kHz)
        if self._vad is not None and self.sample_rate in (8000, 16000, 32000, 48000):
            frame_size = int(self.sample_rate * 0.03)  # 30ms
            int16_audio = (
                audio.astype(np.int16)
                if audio.dtype == np.int16
                else (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
            )

            speech_frames = 0
            total_frames = 0
            for i in range(0, len(int16_audio) - frame_size + 1, frame_size):
                frame_bytes = int16_audio[i : i + frame_size].tobytes()
                total_frames += 1
                try:
                    if self._vad.is_speech(frame_bytes, self.sample_rate):
                        speech_frames += 1
                except Exception:
                    pass

            if total_frames > 0:
                # If more than 30% of frames are speech, classify chunk as speech
                return (speech_frames / total_frames) >= 0.30

        # 2. Energy fallback (RMS calculation)
        rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
        normalized_rms = rms / 32768.0 if audio.dtype == np.int16 else rms
        return bool(normalized_rms > self.energy_threshold)
