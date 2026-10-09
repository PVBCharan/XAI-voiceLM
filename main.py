"""AI Voice Assistant - Main Application Entrypoint."""

import sys
import time
import numpy as np

from config.settings import settings
from utils.logger import logger
from audio.recorder import AudioRecorder
from audio.vad import VoiceActivityDetector
from wakeword.detector import WakeWordDetector
from speaker.verification import SpeakerVerification
from speech.stt import SpeechToText
from speech.tts import TextToSpeech
from emotion.emotion_model import EmotionModel
from assistant.state import StateManager, AssistantState
from assistant.policy import PolicyEngine
from database.database import DatabaseManager


class VoiceAssistant:
    """Master orchestrator for the AI Voice Assistant pipeline."""

    def __init__(self):
        logger.info("Initializing AI Voice Assistant components...")
        self.state_mgr = StateManager()
        self.recorder = AudioRecorder()
        self.vad = VoiceActivityDetector()
        self.wakeword = WakeWordDetector()
        self.speaker_verification = SpeakerVerification()
        self.stt = SpeechToText()
        self.tts = TextToSpeech()
        self.emotion_model = EmotionModel()
        self.policy_engine = PolicyEngine()
        self.db = DatabaseManager()
        self._running = False

    def run_cli_interactive(self):
        """Runs an interactive text/simulation mode for testing without a microphone."""
        logger.info("Starting AI Voice Assistant in Interactive Mode...")
        self.tts.speak("Hello! I am ready to assist you.")

        while True:
            try:
                user_input = input("\nYou (type a prompt or 'exit'): ").strip()
                if not user_input:
                    continue

                if user_input.lower() in ("exit", "quit", "bye"):
                    self.tts.speak("Goodbye!")
                    break

                self.state_mgr.transition_to(AssistantState.PROCESSING)

                # Simulate speaker identification and emotion
                speaker_name = "User"
                emotion = "neutral"

                # Decision Policy & Generation
                decision = self.policy_engine.process(
                    user_text=user_input,
                    speaker_name=speaker_name,
                    emotion=emotion,
                )

                # Output via TTS
                self.state_mgr.transition_to(AssistantState.SPEAKING)
                self.tts.speak(decision.response_text)

                # Persist to database
                self.db.log_interaction(
                    speaker_name=speaker_name,
                    user_input=user_input,
                    assistant_response=decision.response_text,
                    detected_emotion=emotion,
                )

                self.state_mgr.transition_to(AssistantState.IDLE)

                if decision.should_exit:
                    break

            except (KeyboardInterrupt, EOFError):
                logger.info("Session terminated by user.")
                break

    def run_voice_loop(self):
        """Starts real-time microphone listening loop."""
        logger.info("Starting real-time voice listening loop...")
        self._running = True
        self.recorder.start_stream()
        self.tts.speak("Voice assistant is listening.")

        try:
            while self._running:
                chunk = self.recorder.read_chunk(timeout=0.5)
                if chunk is None:
                    continue

                # Voice Activity Detection
                if self.vad.is_speech(chunk):
                    self.state_mgr.transition_to(AssistantState.LISTENING)
                    # Accumulate speech frames
                    speech_frames = [chunk]
                    silence_chunks = 0
                    max_silence = int(settings.audio.silence_timeout_sec / 0.1)

                    while silence_chunks < max_silence:
                        next_chunk = self.recorder.read_chunk(timeout=0.2)
                        if next_chunk is None:
                            break
                        speech_frames.append(next_chunk)
                        if self.vad.is_speech(next_chunk):
                            silence_chunks = 0
                        else:
                            silence_chunks += 1

                    # Combine recorded speech
                    audio_data = np.concatenate(speech_frames)
                    self.state_mgr.transition_to(AssistantState.PROCESSING)

                    # 1. Transcribe speech
                    transcript = self.stt.transcribe(audio_data)
                    if not transcript:
                        self.state_mgr.transition_to(AssistantState.IDLE)
                        continue

                    # 2. Check wake word if enabled
                    if settings.wakeword.enabled:
                        matched = self.wakeword.match_transcript(transcript)
                        if not matched:
                            logger.info(f"Ignored non-wake audio: '{transcript}'")
                            self.state_mgr.transition_to(AssistantState.IDLE)
                            continue

                    # 3. Identify speaker
                    speaker_res = self.speaker_verification.identify_speaker(audio_data)

                    # 4. Detect Emotion
                    emotion_res = self.emotion_model.predict(audio_data, transcript)

                    # 5. Process through Policy Engine
                    decision = self.policy_engine.process(
                        user_text=transcript,
                        speaker_name=speaker_res.speaker_name,
                        emotion=emotion_res.primary_emotion,
                    )

                    # 6. Speak response
                    self.state_mgr.transition_to(AssistantState.SPEAKING)
                    self.tts.speak(decision.response_text)

                    # 7. Log interaction
                    self.db.log_interaction(
                        speaker_name=speaker_res.speaker_name,
                        user_input=transcript,
                        assistant_response=decision.response_text,
                        detected_emotion=emotion_res.primary_emotion,
                    )

                    self.state_mgr.transition_to(AssistantState.IDLE)

                    if decision.should_exit:
                        break

        except KeyboardInterrupt:
            logger.info("Voice loop stopped by user.")
        finally:
            self.recorder.stop_stream()
            logger.info("Shutdown complete.")


def main():
    assistant = VoiceAssistant()
    if "--cli" in sys.argv or "--interactive" in sys.argv:
        # Text-only interactive mode (no microphone needed)
        assistant.run_cli_interactive()
    else:
        # Default: real-time microphone voice loop
        try:
            assistant.run_voice_loop()
        except Exception as e:
            logger.error(f"Voice loop error: {e}. Falling back to CLI mode.")
            assistant.run_cli_interactive()


if __name__ == "__main__":
    main()
