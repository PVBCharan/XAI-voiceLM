"""Intent recognition for fast local action routing."""

from dataclasses import dataclass, field
import re
from utils.logger import logger


@dataclass
class IntentResult:
    intent_name: str
    confidence: float
    parameters: dict = field(default_factory=dict)


class IntentRecognizer:
    """Classifies user queries into system commands or general queries."""

    # Built-in flexible pattern matchers for local system tasks
    PATTERNS = {
        "get_time": [
            r"\bwhat('s| is)?\s+(the\s+)?time\b",
            r"\bwhat\s+time\s+is\s+it\b",
            r"\bcurrent\s+time\b",
            r"\btell\s+me\s+(the\s+)?time\b",
        ],
        "get_date": [
            r"\bwhat('s| is)?\s+(the\s+)?date\b",
            r"\bwhat\s+day(\s+is\s+it|\s+is\s+today)?\b",
            r"\btoday('s)?\s+date\b",
        ],
        "set_volume": [
            r"\bvolume\s+(to|at)?\s*(\d+)\b",
            r"\bset\s+volume\s+(to|at)?\s*(\d+)\b",
            r"\b(increase|decrease|raise|lower)\s+volume\b",
        ],
        "stop_assistant": [
            r"\b(stop|exit|quit|bye|goodbye|shutdown|sleep)\b",
        ],
        "open_application": [
            r"\b(open|launch|start)\s+([a-zA-Z0-9_\- ]+)\b",
        ],
    }

    def detect_intent(self, text: str) -> IntentResult:
        """Analyzes text and returns the recognized intent and extracted parameters."""
        cleaned = text.lower().strip()

        for intent, patterns in self.PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, cleaned)
                if match:
                    params = {}
                    if intent == "set_volume":
                        # Extract digits if present
                        digit_match = re.search(r"\d+", cleaned)
                        if digit_match:
                            params["level"] = digit_match.group()
                    elif intent == "open_application":
                        groups = match.groups()
                        if len(groups) >= 2 and groups[1]:
                            params["app_name"] = groups[1].strip()

                    logger.info(f"Matched local intent: {intent} (params={params})")
                    return IntentResult(intent_name=intent, confidence=0.95, parameters=params)

        # Default to general chat query
        return IntentResult(intent_name="general_query", confidence=0.5, parameters={"query": text})
