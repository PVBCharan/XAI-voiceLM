"""Speaker profile and voiceprint embedding database."""

import json
from pathlib import Path
import numpy as np
from config.settings import settings
from utils.logger import logger


class SpeakerDatabase:
    """Manages enrolled speakers and their stored voiceprint embeddings."""

    def __init__(self, storage_dir: Path = settings.speaker.storage_dir):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.storage_dir / "speakers_index.json"
        self._cache: dict[str, dict] = {}
        self._load_index()

    def _load_index(self):
        """Loads enrolled speakers from disk."""
        if self.index_file.exists():
            try:
                with open(self.index_file, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
                logger.info(f"Loaded {len(self._cache)} registered speakers.")
            except Exception as e:
                logger.error(f"Failed to load speaker index: {e}")
                self._cache = {}

    def _save_index(self):
        """Saves enrolled speakers index to disk."""
        try:
            with open(self.index_file, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save speaker index: {e}")

    def enroll_speaker(self, speaker_id: str, name: str, embedding: np.ndarray) -> bool:
        """Saves a new speaker's voiceprint embedding."""
        emb_file = self.storage_dir / f"{speaker_id}.npy"
        np.save(emb_file, embedding)

        self._cache[speaker_id] = {
            "name": name,
            "embedding_file": str(emb_file.name),
        }
        self._save_index()
        logger.info(f"Speaker '{name}' ({speaker_id}) successfully enrolled.")
        return True

    def get_all_embeddings(self) -> dict[str, tuple[str, np.ndarray]]:
        """Returns mapping of speaker_id -> (name, embedding_vector)."""
        results = {}
        for spk_id, meta in self._cache.items():
            emb_path = self.storage_dir / meta["embedding_file"]
            if emb_path.exists():
                emb = np.load(emb_path)
                results[spk_id] = (meta["name"], emb)
        return results
