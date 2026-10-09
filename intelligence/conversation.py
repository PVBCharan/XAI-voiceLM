"""Conversation context and dialog memory manager."""

from dataclasses import dataclass, field
from datetime import datetime
from config.settings import settings
from utils.logger import logger


@dataclass
class Turn:
    role: str  # 'user' or 'assistant'
    content: str
    speaker_id: str | None = None
    emotion: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class ConversationManager:
    """Maintains active dialog history, multi-turn memory, and context."""

    def __init__(self, max_turns: int = settings.intelligence.max_history_turns):
        self.max_turns = max_turns
        self.history: list[Turn] = []

    def add_user_message(self, message: str, speaker_id: str | None = None, emotion: str | None = None) -> None:
        """Appends a user statement to history."""
        self.history.append(
            Turn(role="user", content=message, speaker_id=speaker_id, emotion=emotion)
        )
        self._prune()

    def add_assistant_message(self, response: str) -> None:
        """Appends an assistant response to history."""
        self.history.append(Turn(role="assistant", content=response))
        self._prune()

    def _prune(self):
        """Keeps history within max_turns boundary."""
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-self.max_turns * 2 :]

    def get_context_list(self) -> list[dict]:
        """Returns dialog formatted for LLM prompts."""
        return [{"role": t.role, "content": t.content} for t in self.history]

    def clear(self):
        """Resets the conversation history."""
        self.history.clear()
        logger.info("Conversation history cleared.")
