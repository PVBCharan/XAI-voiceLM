"""Acoustic feature extraction for speech emotion recognition (SER)."""

from dataclasses import dataclass
import numpy as np


@dataclass
class AcousticFeatures:
    energy_rms: float
    zero_crossing_rate: float
    pitch_estimate: float
    spectral_centroid: float


class FeatureExtractor:
    """Extracts fundamental prosodic and spectral features from raw audio."""

    def extract(self, audio: np.ndarray, sample_rate: int = 16000) -> AcousticFeatures:
        """Extracts energy, zero-crossing rate, spectral centroid, and estimated pitch."""
        if audio is None or len(audio) == 0:
            return AcousticFeatures(0.0, 0.0, 0.0, 0.0)

        audio_float = audio.astype(np.float32)
        if audio.dtype == np.int16:
            audio_float = audio_float / 32768.0

        # Energy RMS
        energy_rms = float(np.sqrt(np.mean(audio_float ** 2)))

        # Zero Crossing Rate
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_float)))) / (2 * len(audio_float))

        # Spectral Centroid
        spectrum = np.abs(np.fft.rfft(audio_float))
        frequencies = np.fft.rfftfreq(len(audio_float), d=1.0 / sample_rate)
        sum_spectrum = np.sum(spectrum)
        spectral_centroid = (
            float(np.sum(frequencies * spectrum) / sum_spectrum)
            if sum_spectrum > 0
            else 0.0
        )

        # Autocorrelation pitch approximation
        autocorr = np.correlate(audio_float, audio_float, mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]
        d = np.diff(autocorr)
        start = np.where(d > 0)[0]
        pitch_estimate = 0.0
        if len(start) > 0:
            peak = np.argmax(autocorr[start[0] :]) + start[0]
            if peak > 0:
                pitch_estimate = float(sample_rate / peak)

        return AcousticFeatures(
            energy_rms=energy_rms,
            zero_crossing_rate=float(zero_crossings),
            pitch_estimate=pitch_estimate,
            spectral_centroid=spectral_centroid,
        )
