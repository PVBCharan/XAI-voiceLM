
"""Large Language Model (LLM) client interface supporting Gemini and OpenAI."""

from config.settings import settings
from utils.logger import logger


class LLMClient:
    """Handles communication with generative AI models (Google Gemini / OpenAI)."""

    def __init__(
        self,
        api_key: str = settings.intelligence.gemini_api_key,
        model_name: str = settings.intelligence.llm_model,
        system_instruction: str = settings.intelligence.system_prompt,
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.system_instruction = system_instruction
        self._backend = None  # 'genai_new', 'google_generativeai', or 'offline'
        self._client = None
        self._init_client()

    def _init_client(self):
        """Initializes Google GenAI or legacy google-generativeai client."""
        if not self.api_key or self.api_key.startswith("your_"):
            logger.warning("GEMINI_API_KEY is not configured in .env. Assistant will run in offline mode.")
            self._backend = "offline"
            return

        # 1. Try modern google-genai SDK
        try:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
            self._backend = "genai_new"
            logger.info(f"Initialized google-genai client with model '{self.model_name}'.")
            return
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"google-genai init skipped: {e}")

        # 2. Try google-generativeai SDK (already installed in Python 3.10 environment)
        try:
            import google.generativeai as genai_legacy
            genai_legacy.configure(api_key=self.api_key)
            self._client = genai_legacy.GenerativeModel(
                model_name=self.model_name,
                system_instruction=self.system_instruction,
            )
            self._backend = "google_generativeai"
            logger.info(f"Initialized google-generativeai client with model '{self.model_name}'.")
            return
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"google-generativeai init failed: {e}")

        self._backend = "offline"
        logger.warning("No generative AI library found. Offline responses will be used.")

    def generate_response(self, prompt: str, conversation_history: list[dict] | None = None) -> str:
        """Generates a text response from the model given the prompt and conversation context."""
        if not prompt:
            return ""

        # Format conversation history
        history_str = ""
        if conversation_history:
            recent_turns = conversation_history[-settings.intelligence.max_history_turns:]
            history_str = "\n".join(
                f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent_turns
            )

        full_prompt = (
            f"Previous conversation:\n{history_str}\n\nUser: {prompt}"
            if history_str
            else prompt
        )

        # 1. google-genai
        if self._backend == "genai_new" and self._client is not None:
            try:
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=full_prompt,
                    config={
                        "system_instruction": self.system_instruction,
                        "temperature": settings.intelligence.temperature,
                    },
                )
                return response.text.strip()
            except Exception as e:
                logger.error(f"google-genai generation error: {e}")

        # 2. google-generativeai
        elif self._backend == "google_generativeai" and self._client is not None:
            try:
                response = self._client.generate_content(full_prompt)
                return response.text.strip()
            except Exception as e:
                logger.error(f"google-generativeai generation error: {e}")

        # Offline fallback response
        return f"I heard your request: '{prompt}'. To enable live AI intelligence, add your GEMINI_API_KEY to .env."
