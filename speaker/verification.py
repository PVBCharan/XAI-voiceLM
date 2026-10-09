"""Speaker verification and identification against stored voice biometrics."""

from dataclasses import dataclass
import numpy as np
from config.settings import settings
from speaker.embedding import SpeakerEmbeddingExtractor
from speaker.database import SpeakerDatabase
from utils.logger import logger


@dataclass
class VerificationResult:
    is_verified: bool
    speaker_id: str | None
    speaker_name: str
    similarity_score: float


class SpeakerVerification:
    """Matches an input audio sample against registered speaker embeddings."""

    def __init__(
        self,
        extractor: SpeakerEmbeddingExtractor | None = None,
        db: SpeakerDatabase | None = None,
        threshold: float = settings.speaker.verification_threshold,
    ):
        self.extractor = extractor or SpeakerEmbeddingExtractor()
        self.db = db or SpeakerDatabase()
        self.threshold = threshold

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Computes cosine similarity between two unit-normalized vectors."""
        dot = np.dot(v1, v2)
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm_product == 0:
            return 0.0
        return float(dot / norm_product)

    def identify_speaker(self, audio: np.ndarray, sample_rate: int = 16000) -> VerificationResult:
        """Identifies which registered speaker is talking."""
        query_emb = self.extractor.extract(audio, sample_rate)
        candidates = self.db.get_all_embeddings()

        if not candidates:
            return VerificationResult(
                is_verified=False,
                speaker_id=None,
                speaker_name="Unknown Speaker",
                similarity_score=0.0,
            )

        best_score = -1.0
        best_id = None
        best_name = "Unknown Speaker"

        for spk_id, (name, emb) in candidates.items():
            score = self._cosine_similarity(query_emb, emb)
            if score > best_score:
                best_score = score
                best_id = spk_id
                best_name = name

        is_verified = best_score >= self.threshold
        logger.info(
            f"Speaker identification: {best_name} (score={best_score:.3f}, verified={is_verified})"
        )

        return VerificationResult(
            is_verified=is_verified,
            speaker_id=best_id if is_verified else None,
            speaker_name=best_name if is_verified else "Unknown Speaker",
            similarity_score=max(0.0, best_score),
        )
