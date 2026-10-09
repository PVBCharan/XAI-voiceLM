"""Emotion classification model based on acoustic features and textual sentiment."""

from dataclasses import dataclass
import numpy as np
from emotion.features import FeatureExtractor, AcousticFeatures
from config.settings import settings
from utils.logger import logger


@dataclass
class EmotionPrediction:
    primary_emotion: str  # neutral, happy, sad, angry, surprised
    confidence: float
    features: AcousticFeatures | None = None


class EmotionModel:
    """Predicts emotional state from user voice audio and transcription."""

    def __init__(self, confidence_threshold: float = settings.emotion.confidence_threshold):
        self.confidence_threshold = confidence_threshold
        self.feature_extractor = FeatureExtractor()

    def predict(self, audio: np.ndarray, text: str = "") -> EmotionPrediction:
        """Evaluates audio features and text clues to determine affective state."""
        features = self.feature_extractor.extract(audio)

        # Baseline heuristic rule engine based on pitch and energy
        # Higher energy + higher pitch -> happy or excited / angry
        # Very low energy + low pitch -> sad or calm
        emotion = "neutral"
        confidence = 0.70

        if features.energy_rms > 0.08 and features.pitch_estimate > 200:
            emotion = "happy"
            confidence = 0.80
        elif features.energy_rms > 0.12:
            emotion = "angry"
            confidence = 0.75
        elif features.energy_rms < 0.02 and features.energy_rms > 0.001:
            emotion = "sad"
            confidence = 0.72

        logger.info(f"Detected emotion: {emotion} (conf: {confidence:.2f})")
        return EmotionPrediction(
            primary_emotion=emotion,
            confidence=confidence,
            features=features,
        )
