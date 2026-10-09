"""AI Voice Assistant - Web Demo Interface.

A beautiful, interactive web-based demo that showcases every module of the
AI Voice Assistant pipeline without requiring microphone hardware or API keys.
Run with:  python web_demo.py
Then open:  http://localhost:5000
"""

import json
import datetime
import sys
import os

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, render_template_string

# --- Import all project modules to prove they load correctly ---
from config.settings import settings
from utils.logger import logger
from audio.recorder import AudioRecorder
from audio.vad import VoiceActivityDetector
from audio.preprocessing import normalize_audio, remove_dc_offset, apply_noise_gate
from wakeword.detector import WakeWordDetector
from speaker.verification import SpeakerVerification
from speaker.embedding import SpeakerEmbeddingExtractor
from speaker.diarization import SpeakerDiarization
from speech.stt import SpeechToText
from speech.tts import TextToSpeech
from emotion.emotion_model import EmotionModel
from emotion.features import FeatureExtractor
from intelligence.intent import IntentRecognizer
from intelligence.llm import LLMClient
from intelligence.conversation import ConversationManager
from assistant.state import StateManager, AssistantState
from assistant.policy import PolicyEngine, PolicyDecision
from assistant.commands import CommandHandler
from database.database import DatabaseManager

# =====================================================================
# Initialize the application
# =====================================================================
app = Flask(__name__)

# Initialize ALL components (same as main.py VoiceAssistant.__init__)
logger.info("=" * 60)
logger.info("  AI Voice Assistant – Web Demo Starting")
logger.info("=" * 60)

state_mgr = StateManager()
recorder = AudioRecorder()
vad = VoiceActivityDetector()
wakeword = WakeWordDetector()
speaker_verification = SpeakerVerification()
stt = SpeechToText()
tts = TextToSpeech()
emotion_model = EmotionModel()
policy_engine = PolicyEngine()
db = DatabaseManager()

logger.info("All modules initialized successfully.")
logger.info(f"Database path: {settings.database.db_path}")
logger.info(f"LLM backend: {policy_engine.llm_client._backend}")
logger.info(f"Wake words: {settings.wakeword.keywords}")


# =====================================================================
# HTML Template - Beautiful Chat UI
# =====================================================================
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Voice Assistant</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0a0a0f;
            --bg-secondary: #12121a;
            --bg-card: #1a1a2e;
            --bg-input: #16162a;
            --accent-primary: #6c63ff;
            --accent-secondary: #4ecdc4;
            --accent-warning: #ff6b6b;
            --accent-success: #2ed573;
            --accent-gold: #ffa502;
            --text-primary: #e8e8f0;
            --text-secondary: #8888a8;
            --text-muted: #555570;
            --border: #2a2a40;
            --shadow-glow: rgba(108, 99, 255, 0.15);
            --user-bubble: linear-gradient(135deg, #6c63ff 0%, #4834d4 100%);
            --assistant-bubble: linear-gradient(135deg, #1a1a2e 0%, #222240 100%);
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            overflow: hidden;
        }

        /* ---- Animated Background ---- */
        .bg-animation {
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            z-index: 0;
            overflow: hidden;
        }
        .bg-animation .orb {
            position: absolute;
            border-radius: 50%;
            filter: blur(80px);
            opacity: 0.12;
            animation: float 20s ease-in-out infinite alternate;
        }
        .bg-animation .orb:nth-child(1) {
            width: 500px; height: 500px;
            background: #6c63ff;
            top: -100px; left: -100px;
            animation-delay: 0s;
        }
        .bg-animation .orb:nth-child(2) {
            width: 400px; height: 400px;
            background: #4ecdc4;
            bottom: -80px; right: -80px;
            animation-delay: -5s;
        }
        .bg-animation .orb:nth-child(3) {
            width: 300px; height: 300px;
            background: #ff6b6b;
            top: 50%; left: 50%;
            transform: translate(-50%, -50%);
            animation-delay: -10s;
        }
        @keyframes float {
            0% { transform: translate(0, 0) scale(1); }
            33% { transform: translate(30px, -40px) scale(1.05); }
            66% { transform: translate(-20px, 20px) scale(0.95); }
            100% { transform: translate(10px, -10px) scale(1.02); }
        }

        /* ---- Main Layout ---- */
        .app-container {
            position: relative;
            z-index: 1;
            display: flex;
            height: 100vh;
        }

        /* ---- Sidebar ---- */
        .sidebar {
            width: 300px;
            background: rgba(18, 18, 26, 0.85);
            backdrop-filter: blur(20px);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            padding: 24px;
        }
        .logo-section {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 32px;
        }
        .logo-icon {
            width: 48px; height: 48px;
            background: var(--user-bubble);
            border-radius: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 22px;
            box-shadow: 0 4px 20px rgba(108, 99, 255, 0.3);
        }
        .logo-text h1 {
            font-size: 18px;
            font-weight: 700;
            letter-spacing: -0.5px;
        }
        .logo-text p {
            font-size: 11px;
            color: var(--text-secondary);
            margin-top: 2px;
        }

        /* Status Panel */
        .status-panel {
            background: var(--bg-card);
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid var(--border);
        }
        .status-panel h3 {
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: var(--text-secondary);
            margin-bottom: 16px;
        }
        .status-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255,255,255,0.04);
        }
        .status-item:last-child { border-bottom: none; }
        .status-label {
            font-size: 13px;
            color: var(--text-secondary);
        }
        .status-value {
            font-size: 13px;
            font-weight: 600;
        }
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 11px;
            font-weight: 600;
        }
        .badge-online {
            background: rgba(46, 213, 115, 0.15);
            color: #2ed573;
        }
        .badge-offline {
            background: rgba(255, 107, 107, 0.15);
            color: #ff6b6b;
        }
        .badge-idle {
            background: rgba(78, 205, 196, 0.15);
            color: #4ecdc4;
        }
        .dot { width: 6px; height: 6px; border-radius: 50%; }
        .dot-green { background: #2ed573; }
        .dot-red { background: #ff6b6b; }
        .dot-cyan { background: #4ecdc4; animation: pulse-dot 2s ease-in-out infinite; }
        @keyframes pulse-dot {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }

        /* Module Status */
        .modules-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            margin-top: 12px;
        }
        .module-chip {
            background: rgba(108, 99, 255, 0.08);
            border: 1px solid rgba(108, 99, 255, 0.15);
            border-radius: 8px;
            padding: 8px 10px;
            font-size: 10px;
            text-align: center;
            color: var(--text-secondary);
            transition: all 0.3s ease;
        }
        .module-chip.active {
            background: rgba(108, 99, 255, 0.18);
            color: var(--accent-primary);
            border-color: rgba(108, 99, 255, 0.3);
        }

        /* ---- Chat Area ---- */
        .chat-area {
            flex: 1;
            display: flex;
            flex-direction: column;
            position: relative;
        }

        /* Header */
        .chat-header {
            padding: 20px 32px;
            background: rgba(18, 18, 26, 0.6);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .chat-header-left h2 {
            font-size: 16px;
            font-weight: 600;
        }
        .chat-header-left p {
            font-size: 12px;
            color: var(--text-secondary);
            margin-top: 2px;
        }
        .header-controls {
            display: flex;
            gap: 8px;
        }
        .control-btn {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 8px 14px;
            color: var(--text-secondary);
            cursor: pointer;
            font-size: 12px;
            font-family: 'Inter', sans-serif;
            transition: all 0.2s ease;
        }
        .control-btn:hover {
            background: rgba(108, 99, 255, 0.15);
            color: var(--accent-primary);
            border-color: rgba(108, 99, 255, 0.3);
        }

        /* Messages */
        .messages-container {
            flex: 1;
            overflow-y: auto;
            padding: 32px;
            scroll-behavior: smooth;
        }
        .messages-container::-webkit-scrollbar { width: 6px; }
        .messages-container::-webkit-scrollbar-track { background: transparent; }
        .messages-container::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

        .welcome-screen {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            text-align: center;
            gap: 16px;
        }
        .welcome-icon {
            width: 80px; height: 80px;
            background: var(--user-bubble);
            border-radius: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 36px;
            box-shadow: 0 8px 40px rgba(108, 99, 255, 0.25);
            animation: breathe 4s ease-in-out infinite;
        }
        @keyframes breathe {
            0%, 100% { transform: scale(1); box-shadow: 0 8px 40px rgba(108, 99, 255, 0.25); }
            50% { transform: scale(1.05); box-shadow: 0 12px 50px rgba(108, 99, 255, 0.35); }
        }
        .welcome-screen h2 {
            font-size: 24px;
            font-weight: 700;
            letter-spacing: -0.5px;
        }
        .welcome-screen p {
            color: var(--text-secondary);
            font-size: 14px;
            max-width: 400px;
            line-height: 1.6;
        }
        .quick-actions {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            justify-content: center;
            margin-top: 12px;
        }
        .quick-btn {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 10px 18px;
            color: var(--text-secondary);
            font-size: 13px;
            font-family: 'Inter', sans-serif;
            cursor: pointer;
            transition: all 0.25s ease;
        }
        .quick-btn:hover {
            background: rgba(108, 99, 255, 0.12);
            border-color: rgba(108, 99, 255, 0.3);
            color: var(--text-primary);
            transform: translateY(-2px);
        }

        /* Message Bubbles */
        .message {
            display: flex;
            gap: 12px;
            margin-bottom: 20px;
            animation: slideIn 0.35s cubic-bezier(0.16, 1, 0.3, 1);
        }
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(12px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .message.user { flex-direction: row-reverse; }
        .msg-avatar {
            width: 36px; height: 36px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
            flex-shrink: 0;
        }
        .user .msg-avatar {
            background: var(--user-bubble);
        }
        .assistant .msg-avatar {
            background: var(--assistant-bubble);
            border: 1px solid var(--border);
        }
        .msg-body {
            max-width: 65%;
        }
        .msg-meta {
            display: flex;
            gap: 8px;
            align-items: center;
            margin-bottom: 6px;
            font-size: 11px;
            color: var(--text-muted);
        }
        .user .msg-meta { flex-direction: row-reverse; }
        .msg-bubble {
            padding: 14px 18px;
            border-radius: 18px;
            font-size: 14px;
            line-height: 1.6;
        }
        .user .msg-bubble {
            background: var(--user-bubble);
            border-bottom-right-radius: 4px;
        }
        .assistant .msg-bubble {
            background: var(--assistant-bubble);
            border: 1px solid var(--border);
            border-bottom-left-radius: 4px;
        }

        /* Pipeline Info */
        .pipeline-info {
            margin-top: 8px;
            background: rgba(108, 99, 255, 0.06);
            border: 1px solid rgba(108, 99, 255, 0.12);
            border-radius: 12px;
            padding: 12px 16px;
            font-size: 11px;
        }
        .pipeline-info .pi-title {
            font-weight: 600;
            color: var(--accent-primary);
            margin-bottom: 8px;
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .pipeline-steps {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        .pipeline-step {
            display: flex;
            align-items: center;
            gap: 8px;
            color: var(--text-secondary);
        }
        .pipeline-step .step-icon { font-size: 12px; }
        .pipeline-step .step-label { font-weight: 500; }
        .pipeline-step .step-value { color: var(--text-muted); margin-left: auto; }

        /* Typing Indicator */
        .typing-indicator {
            display: none;
            gap: 12px;
            margin-bottom: 20px;
            align-items: flex-start;
        }
        .typing-indicator.visible { display: flex; }
        .typing-dots {
            display: flex;
            gap: 4px;
            padding: 16px 20px;
            background: var(--assistant-bubble);
            border: 1px solid var(--border);
            border-radius: 18px;
            border-bottom-left-radius: 4px;
        }
        .typing-dots span {
            width: 8px; height: 8px;
            background: var(--text-secondary);
            border-radius: 50%;
            animation: typing 1.4s ease-in-out infinite;
        }
        .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
        .typing-dots span:nth-child(3) { animation-delay: 0.4s; }
        @keyframes typing {
            0%, 100% { opacity: 0.3; transform: scale(0.8); }
            50% { opacity: 1; transform: scale(1); }
        }

        /* Input Area */
        .input-area {
            padding: 20px 32px;
            background: rgba(18, 18, 26, 0.8);
            backdrop-filter: blur(12px);
            border-top: 1px solid var(--border);
        }
        .input-wrapper {
            display: flex;
            gap: 12px;
            align-items: center;
            background: var(--bg-input);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 6px 6px 6px 20px;
            transition: border-color 0.3s ease, box-shadow 0.3s ease;
        }
        .input-wrapper:focus-within {
            border-color: rgba(108, 99, 255, 0.5);
            box-shadow: 0 0 0 3px rgba(108, 99, 255, 0.1);
        }
        .input-wrapper input {
            flex: 1;
            background: none;
            border: none;
            color: var(--text-primary);
            font-size: 14px;
            font-family: 'Inter', sans-serif;
            outline: none;
        }
        .input-wrapper input::placeholder {
            color: var(--text-muted);
        }
        .send-btn {
            width: 44px; height: 44px;
            background: var(--user-bubble);
            border: none;
            border-radius: 12px;
            color: white;
            font-size: 18px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s ease;
            flex-shrink: 0;
        }
        .send-btn:hover {
            transform: scale(1.05);
            box-shadow: 0 4px 20px rgba(108, 99, 255, 0.3);
        }
        .send-btn:active { transform: scale(0.95); }

        .input-hint {
            text-align: center;
            font-size: 11px;
            color: var(--text-muted);
            margin-top: 8px;
        }

        /* Emotion Badge */
        .emotion-badge {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 11px;
            font-weight: 600;
        }
        .emotion-neutral { background: rgba(136,136,168,0.15); color: #8888a8; }
        .emotion-happy { background: rgba(46,213,115,0.15); color: #2ed573; }
        .emotion-sad { background: rgba(116,185,255,0.15); color: #74b9ff; }
        .emotion-angry { background: rgba(255,107,107,0.15); color: #ff6b6b; }
        .emotion-surprised { background: rgba(255,165,2,0.15); color: #ffa502; }

        /* Responsive */
        @media (max-width: 768px) {
            .sidebar { display: none; }
            .messages-container { padding: 16px; }
            .input-area { padding: 12px 16px; }
            .msg-body { max-width: 85%; }
        }
    </style>
</head>
<body>
    <div class="bg-animation">
        <div class="orb"></div>
        <div class="orb"></div>
        <div class="orb"></div>
    </div>

    <div class="app-container">
        <!-- Sidebar -->
        <aside class="sidebar">
            <div class="logo-section">
                <div class="logo-icon">🤖</div>
                <div class="logo-text">
                    <h1>AI Voice Assistant</h1>
                    <p>Intelligent • Modular • Real-time</p>
                </div>
            </div>

            <!-- System Status -->
            <div class="status-panel">
                <h3>System Status</h3>
                <div class="status-item">
                    <span class="status-label">State</span>
                    <span class="status-badge badge-idle" id="stateDisplay">
                        <span class="dot dot-cyan"></span>
                        IDLE
                    </span>
                </div>
                <div class="status-item">
                    <span class="status-label">LLM</span>
                    <span class="status-value" id="llmStatus" style="color: var(--accent-secondary); font-size:12px;">{{ llm_backend }}</span>
                </div>
                <div class="status-item">
                    <span class="status-label">Wake Word</span>
                    <span class="status-value" style="color: var(--accent-gold); font-size:12px;">{{ wake_words }}</span>
                </div>
                <div class="status-item">
                    <span class="status-label">STT Engine</span>
                    <span class="status-value" style="font-size:12px;">{{ stt_engine }}</span>
                </div>
            </div>

            <!-- Modules -->
            <div class="status-panel">
                <h3>Active Modules</h3>
                <div class="modules-grid">
                    <div class="module-chip active" id="mod-vad">🎙 VAD</div>
                    <div class="module-chip active" id="mod-wakeword">👋 Wake Word</div>
                    <div class="module-chip active" id="mod-stt">📝 STT</div>
                    <div class="module-chip active" id="mod-tts">🔊 TTS</div>
                    <div class="module-chip active" id="mod-emotion">😊 Emotion</div>
                    <div class="module-chip active" id="mod-speaker">👤 Speaker</div>
                    <div class="module-chip active" id="mod-intent">🎯 Intent</div>
                    <div class="module-chip active" id="mod-llm">🧠 LLM</div>
                    <div class="module-chip active" id="mod-policy">⚙️ Policy</div>
                    <div class="module-chip active" id="mod-db">💾 Database</div>
                </div>
            </div>

            <div style="margin-top:auto; text-align:center; font-size:11px; color:var(--text-muted);">
                Python 3.10 • Flask • SQLite<br>
                Built for Demo Prototype
            </div>
        </aside>

        <!-- Chat Area -->
        <main class="chat-area">
            <div class="chat-header">
                <div class="chat-header-left">
                    <h2>💬 Interactive Session</h2>
                    <p>Type commands or questions — all pipeline modules are active</p>
                </div>
                <div class="header-controls">
                    <button class="control-btn" onclick="clearChat()">🗑 Clear</button>
                    <button class="control-btn" onclick="showHistory()">📋 History</button>
                </div>
            </div>

            <div class="messages-container" id="messagesContainer">
                <div class="welcome-screen" id="welcomeScreen">
                    <div class="welcome-icon">🤖</div>
                    <h2>Hello! I'm your AI Assistant</h2>
                    <p>I can answer questions, run local commands, detect emotions, and maintain conversation context. Try one of the suggestions below!</p>
                    <div class="quick-actions">
                        <button class="quick-btn" onclick="sendQuick('What time is it?')">⏰ What time is it?</button>
                        <button class="quick-btn" onclick="sendQuick('What is today\'s date?')">📅 Today's date</button>
                        <button class="quick-btn" onclick="sendQuick('Tell me about yourself')">🤖 About you</button>
                        <button class="quick-btn" onclick="sendQuick('Open notepad')">📝 Open Notepad</button>
                        <button class="quick-btn" onclick="sendQuick('Set volume to 75')">🔊 Set Volume</button>
                        <button class="quick-btn" onclick="sendQuick('How does photosynthesis work?')">🌿 Science Q&A</button>
                    </div>
                </div>

                <!-- Typing indicator -->
                <div class="typing-indicator" id="typingIndicator">
                    <div class="msg-avatar" style="background: var(--assistant-bubble); border: 1px solid var(--border);">🤖</div>
                    <div class="typing-dots">
                        <span></span><span></span><span></span>
                    </div>
                </div>
            </div>

            <div class="input-area">
                <div class="input-wrapper">
                    <input type="text" id="userInput" placeholder="Type a message or command..." 
                           autocomplete="off" autofocus
                           onkeydown="if(event.key==='Enter') sendMessage()">
                    <button class="send-btn" onclick="sendMessage()">➤</button>
                </div>
                <div class="input-hint">
                    Press Enter to send • Try "what time is it", "open calculator", or ask any question
                </div>
            </div>
        </main>
    </div>

    <script>
        const messagesContainer = document.getElementById('messagesContainer');
        const welcomeScreen = document.getElementById('welcomeScreen');
        const typingIndicator = document.getElementById('typingIndicator');
        const userInput = document.getElementById('userInput');
        const stateDisplay = document.getElementById('stateDisplay');
        let messageCount = 0;

        const EMOTION_MAP = {
            'neutral': { icon: '😐', cls: 'emotion-neutral' },
            'happy':   { icon: '😊', cls: 'emotion-happy' },
            'sad':     { icon: '😢', cls: 'emotion-sad' },
            'angry':   { icon: '😠', cls: 'emotion-angry' },
            'surprised': { icon: '😮', cls: 'emotion-surprised' },
        };

        function updateState(state) {
            const colors = {
                'IDLE': { badge: 'badge-idle', dot: 'dot-cyan' },
                'PROCESSING': { badge: 'badge-online', dot: 'dot-green' },
                'SPEAKING': { badge: 'badge-online', dot: 'dot-green' },
            };
            const c = colors[state] || colors['IDLE'];
            stateDisplay.className = `status-badge ${c.badge}`;
            stateDisplay.innerHTML = `<span class="dot ${c.dot}"></span> ${state}`;
        }

        function addMessage(role, text, pipelineData) {
            if (messageCount === 0 && welcomeScreen) {
                welcomeScreen.style.display = 'none';
            }
            messageCount++;

            const msgDiv = document.createElement('div');
            msgDiv.className = `message ${role}`;

            const avatar = role === 'user' ? '👤' : '🤖';
            const name = role === 'user' ? 'You' : 'Assistant';
            const time = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});

            let pipelineHTML = '';
            if (pipelineData && role === 'assistant') {
                const em = EMOTION_MAP[pipelineData.emotion] || EMOTION_MAP['neutral'];
                pipelineHTML = `
                    <div class="pipeline-info">
                        <div class="pi-title">⚡ Pipeline Trace</div>
                        <div class="pipeline-steps">
                            <div class="pipeline-step">
                                <span class="step-icon">🎯</span>
                                <span class="step-label">Intent</span>
                                <span class="step-value">${pipelineData.intent}</span>
                            </div>
                            <div class="pipeline-step">
                                <span class="step-icon">${em.icon}</span>
                                <span class="step-label">Emotion</span>
                                <span class="step-value"><span class="emotion-badge ${em.cls}">${em.icon} ${pipelineData.emotion}</span></span>
                            </div>
                            <div class="pipeline-step">
                                <span class="step-icon">👤</span>
                                <span class="step-label">Speaker</span>
                                <span class="step-value">${pipelineData.speaker}</span>
                            </div>
                            <div class="pipeline-step">
                                <span class="step-icon">📍</span>
                                <span class="step-label">Route</span>
                                <span class="step-value">${pipelineData.route}</span>
                            </div>
                            <div class="pipeline-step">
                                <span class="step-icon">💾</span>
                                <span class="step-label">Database</span>
                                <span class="step-value">Logged ✓</span>
                            </div>
                        </div>
                    </div>
                `;
            }

            msgDiv.innerHTML = `
                <div class="msg-avatar">${avatar}</div>
                <div class="msg-body">
                    <div class="msg-meta">
                        <span style="font-weight:600">${name}</span>
                        <span>${time}</span>
                    </div>
                    <div class="msg-bubble">${text}</div>
                    ${pipelineHTML}
                </div>
            `;

            // Insert before typing indicator
            messagesContainer.insertBefore(msgDiv, typingIndicator);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        async function sendMessage() {
            const text = userInput.value.trim();
            if (!text) return;

            userInput.value = '';
            addMessage('user', text);

            updateState('PROCESSING');
            typingIndicator.classList.add('visible');
            messagesContainer.scrollTop = messagesContainer.scrollHeight;

            // Pulse active modules
            document.querySelectorAll('.module-chip').forEach(el => {
                el.style.animation = 'none';
                void el.offsetHeight;
                el.style.animation = '';
            });

            try {
                const res = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: text })
                });
                const data = await res.json();

                typingIndicator.classList.remove('visible');
                updateState('SPEAKING');

                addMessage('assistant', data.response, data.pipeline);

                setTimeout(() => updateState('IDLE'), 1200);

                if (data.should_exit) {
                    setTimeout(() => {
                        userInput.disabled = true;
                        userInput.placeholder = 'Session ended — refresh to restart';
                    }, 1500);
                }
            } catch (err) {
                typingIndicator.classList.remove('visible');
                updateState('IDLE');
                addMessage('assistant', 'Sorry, an error occurred. Please try again.');
            }
        }

        function sendQuick(text) {
            userInput.value = text;
            sendMessage();
        }

        function clearChat() {
            const msgs = messagesContainer.querySelectorAll('.message');
            msgs.forEach(m => m.remove());
            messageCount = 0;
            welcomeScreen.style.display = 'flex';
            fetch('/api/clear', { method: 'POST' });
        }

        async function showHistory() {
            try {
                const res = await fetch('/api/history');
                const data = await res.json();
                if (data.history && data.history.length > 0) {
                    let text = '📋 <b>Recent Interaction History</b> (from SQLite DB):<br><br>';
                    data.history.forEach((row, i) => {
                        text += `<b>${i+1}.</b> [${row.timestamp}]<br>`;
                        text += `&nbsp;&nbsp;👤 <i>${row.speaker}</i>: ${row.input}<br>`;
                        text += `&nbsp;&nbsp;🤖 ${row.response}<br>`;
                        text += `&nbsp;&nbsp;😊 Emotion: ${row.emotion}<br><br>`;
                    });
                    addMessage('assistant', text, null);
                } else {
                    addMessage('assistant', '📋 No interaction history found yet. Start chatting to build history!', null);
                }
            } catch(e) {
                addMessage('assistant', 'Could not fetch history.', null);
            }
        }

        // Focus input on load
        userInput.focus();
    </script>
</body>
</html>
"""


# =====================================================================
# API Routes
# =====================================================================

@app.route("/")
def index():
    """Serves the main chat UI."""
    stt_status = "faster-whisper" if stt._faster_model else ("whisper" if stt._openai_whisper else "Mock (offline)")
    return render_template_string(
        HTML_TEMPLATE,
        llm_backend=policy_engine.llm_client._backend.replace("_", " ").title(),
        wake_words=", ".join(settings.wakeword.keywords),
        stt_engine=stt_status,
    )


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Processes a user message through the full assistant pipeline."""
    data = request.get_json()
    user_text = data.get("message", "").strip()
    if not user_text:
        return jsonify({"response": "Please say something!", "pipeline": None})

    # --- Full Pipeline Execution ---

    # 1. State: PROCESSING
    state_mgr.transition_to(AssistantState.PROCESSING)

    # 2. Intent Recognition
    intent_result = policy_engine.intent_recognizer.detect_intent(user_text)

    # 3. Simulated speaker + emotion (demo mode, no mic)
    speaker_name = "Demo User"
    emotion = "neutral"

    # 4. Process through Policy Engine (routes to command or LLM)
    decision = policy_engine.process(
        user_text=user_text,
        speaker_name=speaker_name,
        emotion=emotion,
    )

    # 5. State: SPEAKING
    state_mgr.transition_to(AssistantState.SPEAKING)

    # 6. TTS output (console log for demo)
    tts.speak(decision.response_text)

    # 7. Log to database
    db.log_interaction(
        speaker_name=speaker_name,
        user_input=user_text,
        assistant_response=decision.response_text,
        detected_emotion=emotion,
    )

    # 8. Return to IDLE
    state_mgr.transition_to(AssistantState.IDLE)

    # Determine route label
    route = "Local Command" if decision.action_executed else "LLM / Conversational"

    return jsonify({
        "response": decision.response_text,
        "should_exit": decision.should_exit,
        "pipeline": {
            "intent": intent_result.intent_name.replace("_", " ").title(),
            "emotion": emotion,
            "speaker": speaker_name,
            "route": route,
        },
    })


@app.route("/api/clear", methods=["POST"])
def api_clear():
    """Clears conversation context."""
    policy_engine.conversation_manager.clear()
    return jsonify({"status": "cleared"})


@app.route("/api/history")
def api_history():
    """Returns recent interactions from the SQLite database."""
    try:
        import sqlite3
        conn = sqlite3.connect(settings.database.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT timestamp, speaker_name, user_input, assistant_response, detected_emotion "
            "FROM interactions ORDER BY id DESC LIMIT 10"
        )
        rows = cursor.fetchall()
        conn.close()
        history = [
            {
                "timestamp": row["timestamp"],
                "speaker": row["speaker_name"],
                "input": row["user_input"],
                "response": row["assistant_response"],
                "emotion": row["detected_emotion"],
            }
            for row in reversed(rows)
        ]
        return jsonify({"history": history})
    except Exception as e:
        logger.error(f"History fetch error: {e}")
        return jsonify({"history": []})


# =====================================================================
# Entry Point
# =====================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  AI Voice Assistant - Web Demo")
    print("  Open in browser: http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
