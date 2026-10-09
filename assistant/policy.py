"""Assistant decision policy: routes between local commands, database memory, and LLM reasoning."""

from dataclasses import dataclass
from intelligence.intent import IntentRecognizer, IntentResult
from assistant.commands import CommandHandler
from intelligence.llm import LLMClient
from intelligence.conversation import ConversationManager
from utils.logger import logger


@dataclass
class PolicyDecision:
    response_text: str
    action_executed: bool
    should_exit: bool = False


class PolicyEngine:
    """Orchestrates how user queries are processed and answered."""

    def __init__(
        self,
        intent_recognizer: IntentRecognizer | None = None,
        command_handler: CommandHandler | None = None,
        llm_client: LLMClient | None = None,
        conversation_manager: ConversationManager | None = None,
    ):
        self.intent_recognizer = intent_recognizer or IntentRecognizer()
        self.command_handler = command_handler or CommandHandler()
        self.llm_client = llm_client or LLMClient()
        self.conversation_manager = conversation_manager or ConversationManager()

    def process(self, user_text: str, speaker_name: str = "User", emotion: str = "neutral") -> PolicyDecision:
        """Evaluates input and returns the generated spoken response."""
        # 1. Detect Intent
        intent_result = self.intent_recognizer.detect_intent(user_text)

        # 2. Check if local command
        if intent_result.intent_name != "general_query":
            command_output = self.command_handler.execute(
                intent_result.intent_name, intent_result.parameters
            )
            if command_output:
                should_exit = intent_result.intent_name == "stop_assistant"
                return PolicyDecision(
                    response_text=command_output,
                    action_executed=True,
                    should_exit=should_exit,
                )

        # 3. Route to LLM for conversational query
        self.conversation_manager.add_user_message(
            user_text, speaker_id=speaker_name, emotion=emotion
        )
        context = self.conversation_manager.get_context_list()

        # Augment prompt with user context
        augmented_prompt = f"[Speaker: {speaker_name}, Emotional state: {emotion}]\n{user_text}"
        response_text = self.llm_client.generate_response(augmented_prompt, context)

        self.conversation_manager.add_assistant_message(response_text)

        return PolicyDecision(
            response_text=response_text,
            action_executed=False,
            should_exit=False,
        )
