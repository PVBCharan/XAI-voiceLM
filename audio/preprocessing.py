"""Audio preprocessing utilities: normalization, filtering, and resampling."""

import numpy as np


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Normalizes audio signal to float32 values between -1.0 and 1.0."""
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        return audio.astype(np.float32) / 2147483648.0

    # If already float, scale to max absolute peak
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        return audio / max_val
    return audio


def float_to_int16(audio: np.ndarray) -> np.ndarray:
    """Converts float32 audio (-1.0 to 1.0) back to int16 PCM."""
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16)


def remove_dc_offset(audio: np.ndarray) -> np.ndarray:
    """Removes DC bias/offset from audio signal."""
    return audio - np.mean(audio)


def apply_noise_gate(audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """Attenuates samples that fall below an amplitude threshold."""
    gated = audio.copy()
    gated[np.abs(gated) < threshold] = 0.0
    return gated
