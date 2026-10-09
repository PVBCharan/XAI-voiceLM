"""State machine and states for the AI Voice Assistant."""

from enum import Enum, auto
from utils.logger import logger


class AssistantState(Enum):
    IDLE = auto()
    LISTENING = auto()
    PROCESSING = auto()
    SPEAKING = auto()
    ERROR = auto()


class StateManager:
    """Tracks and validates state transitions of the assistant."""

    def __init__(self):
        self._current_state = AssistantState.IDLE

    @property
    def current_state(self) -> AssistantState:
        return self._current_state

    def transition_to(self, new_state: AssistantState) -> None:
        """Transitions to a new state and logs the transition."""
        if self._current_state != new_state:
            logger.info(f"State transition: {self._current_state.name} ➔ {new_state.name}")
            self._current_state = new_state
