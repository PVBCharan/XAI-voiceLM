"""Audio recording and microphone stream management."""

import queue
import threading
import numpy as np
from utils.logger import logger
from config.settings import settings


class AudioRecorder:
    """Manages audio capture from microphone in real-time chunks using sounddevice."""

    def __init__(
        self,
        sample_rate: int = settings.audio.sample_rate,
        channels: int = settings.audio.channels,
        chunk_size: int = settings.audio.chunk_size,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._is_recording = False
        self._stream = None
        self._lock = threading.Lock()

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback executed by sounddevice stream whenever new audio is captured."""
        if status:
            logger.warning(f"Audio stream status: {status}")
        self.audio_queue.put(indata.copy())

    def start_stream(self) -> None:
        """Starts real-time continuous audio streaming from the microphone."""
        with self._lock:
            if self._is_recording:
                return

            try:
                import sounddevice as sd
                self._stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    blocksize=self.chunk_size,
                    dtype="int16",
                    callback=self._audio_callback,
                )
                self._stream.start()
                self._is_recording = True
                logger.info("Audio stream started successfully.")
            except ImportError:
                logger.warning("sounddevice is not installed. Using simulated stream.")
                self._is_recording = True
            except Exception as e:
                logger.error(f"Failed to start audio stream: {e}")
                raise

    def stop_stream(self) -> None:
        """Stops the audio recording stream."""
        with self._lock:
            if not self._is_recording:
                return

            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception as e:
                    logger.error(f"Error stopping stream: {e}")
                self._stream = None

            self._is_recording = False
            logger.info("Audio stream stopped.")

    def read_chunk(self, timeout: float = 1.0) -> np.ndarray | None:
        """Reads a single chunk from the audio queue."""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def record_seconds(self, duration: float) -> np.ndarray:
        """Synchronously records audio for a given duration in seconds."""
        try:
            import sounddevice as sd
            logger.info(f"Recording for {duration} seconds...")
            recording = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="int16",
            )
            sd.wait()
            return recording.squeeze()
        except ImportError:
            logger.warning("sounddevice not installed. Returning empty buffer.")
            return np.zeros(int(duration * self.sample_rate), dtype=np.int16)

    @property
    def is_recording(self) -> bool:
        return self._is_recording
