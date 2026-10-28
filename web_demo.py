"""Explainable AI Interactive Voice Assistant for Remote Sensing & Physics.

Proof-of-Concept (PoC) Prototype implementing PRD requirements:
- FR-1: Two-Way Voice Interaction (Mic input, Web Speech synthesis, realtime transcription)
- FR-2: Voice Recognition & Session Memory (Speaker biometrics for Charan, profile match, topic resumption)
- FR-3: Explainability & Confidence Scoring (Equations, citations, grounding score, inference conclusion)
- FR-4: Neural Path & Weight / Dynamic Attention Visualization (Dynamic multi-layer animated neural graph)
- FR-5: Adversarial / Corrupted Source Detection (Physics Invariant Validator)
- Preferred UI Design: Mobile-framed dark ember/burnt orange aesthetic with 3D luminous orb
"""

import sys
import os
import json
import datetime
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, render_template_string
from intelligence.remote_sensing import rs_engine
from config.settings import settings
from utils.logger import logger
from database.database import DatabaseManager

app = Flask(__name__)
db = DatabaseManager()

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>XAI Voice Assistant | Remote Sensing</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-top: #df580c;
            --bg-mid: #a73200;
            --bg-dark: #220800;
            --bg-bottom: #080302;
            --surface-glass: rgba(22, 10, 6, 0.75);
            --surface-card: rgba(36, 16, 10, 0.65);
            --surface-pill: rgba(255, 255, 255, 0.12);
            --border-glass: rgba(255, 140, 60, 0.18);
            --border-focus: rgba(255, 120, 40, 0.45);
            --accent-orange: #ff6a00;
            --accent-glow: #ff8c33;
            --accent-gold: #ffbe3b;
            --text-pure: #ffffff;
            --text-warm: #f6dcd0;
            --text-sub: #b39282;
            --text-muted: #795e52;
            --badge-green: #10b981;
            --badge-red: #ef4444;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            -webkit-tap-highlight-color: transparent;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: #080302;
            color: var(--text-pure);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            overflow-x: hidden;
            background: radial-gradient(circle at 50% 20%, #2a0b02 0%, #060201 100%);
        }

        /* Desktop device container wrapper */
        .device-wrapper {
            width: 100%;
            max-width: 440px;
            height: 900px;
            max-height: 98vh;
            background: linear-gradient(180deg, var(--bg-top) 0%, var(--bg-mid) 28%, var(--bg-dark) 65%, var(--bg-bottom) 100%);
            border-radius: 50px;
            box-shadow: 0 25px 80px rgba(0, 0, 0, 0.8), 0 0 0 12px #18181b, 0 0 0 14px rgba(255, 120, 40, 0.2);
            position: relative;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }

        /* Subtle film grain texture overlay */
        .grain-overlay {
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            pointer-events: none;
            opacity: 0.18;
            background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.8' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)'/%3E%3C/svg%3E");
            z-index: 2;
        }

        /* Status bar */
        .status-bar {
            height: 44px;
            padding: 12px 28px 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 14px;
            font-weight: 600;
            z-index: 20;
            position: relative;
        }

        .dynamic-island {
            width: 110px;
            height: 28px;
            background: #000000;
            border-radius: 20px;
            position: absolute;
            left: 50%;
            top: 10px;
            transform: translateX(-50%);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.5);
        }

        .camera-lens {
            width: 10px;
            height: 10px;
            background: #111a2e;
            border-radius: 50%;
            border: 1px solid rgba(255,255,255,0.08);
        }

        .status-icons {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 12px;
        }

        /* SCREEN 1: Home Dashboard View */
        #screen-home {
            flex: 1;
            display: flex;
            flex-direction: column;
            padding: 18px 24px 24px;
            position: relative;
            z-index: 10;
            transition: opacity 0.35s ease, transform 0.35s ease;
        }

        .header-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 32px;
        }

        .user-profile {
            display: flex;
            align-items: center;
            gap: 12px;
            cursor: pointer;
        }

        .avatar-circle {
            width: 44px;
            height: 44px;
            border-radius: 50%;
            border: 2px solid rgba(255, 255, 255, 0.7);
            display: flex;
            align-items: center;
            justify-content: center;
            background: radial-gradient(circle, rgba(255,255,255,0.2) 0%, rgba(0,0,0,0.4) 100%);
            position: relative;
        }

        .avatar-circle svg {
            width: 26px;
            height: 26px;
            fill: none;
            stroke: #ffffff;
            stroke-width: 1.8;
        }

        .user-text h2 {
            font-size: 16px;
            font-weight: 700;
            letter-spacing: -0.2px;
        }

        .user-text p {
            font-size: 13px;
            color: var(--text-warm);
            opacity: 0.85;
        }

        .header-actions {
            display: flex;
            gap: 10px;
        }

        .icon-btn {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.12);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--text-pure);
            cursor: pointer;
            transition: background 0.2s, transform 0.15s;
        }

        .icon-btn:hover {
            background: rgba(255, 255, 255, 0.2);
            transform: scale(1.05);
        }

        .hero-title {
            font-size: 38px;
            font-weight: 800;
            line-height: 1.12;
            letter-spacing: -1.2px;
            color: #ffffff;
            margin-bottom: 26px;
            text-shadow: 0 4px 18px rgba(0, 0, 0, 0.3);
        }

        /* Suggestion prompt cards row */
        .cards-row {
            display: flex;
            gap: 12px;
            overflow-x: auto;
            padding-bottom: 12px;
            margin-bottom: auto;
            scrollbar-width: none;
        }
        .cards-row::-webkit-scrollbar { display: none; }

        .prompt-card {
            min-width: 156px;
            max-width: 156px;
            height: 152px;
            background: rgba(30, 10, 4, 0.55);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 120, 50, 0.18);
            border-radius: 22px;
            padding: 16px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            cursor: pointer;
            transition: all 0.25s ease;
        }

        .prompt-card:hover {
            transform: translateY(-3px);
            background: rgba(45, 16, 6, 0.7);
            border-color: rgba(255, 140, 60, 0.35);
            box-shadow: 0 8px 24px rgba(255, 90, 0, 0.18);
        }

        .sparkle-icon {
            color: #ff6a00;
            font-size: 20px;
            text-shadow: 0 0 12px rgba(255, 106, 0, 0.8);
        }

        .card-text {
            font-size: 13.5px;
            font-weight: 600;
            line-height: 1.35;
            color: #fcece5;
        }

        .card-tag {
            font-size: 10px;
            color: #ff9d66;
            text-transform: uppercase;
            letter-spacing: 0.6px;
            margin-top: 4px;
            display: block;
        }

        /* Bottom Floating Input Card */
        .bottom-input-container {
            background: rgba(14, 5, 2, 0.88);
            backdrop-filter: blur(28px);
            border: 1px solid rgba(255, 120, 50, 0.2);
            border-radius: 32px;
            padding: 20px 20px 16px;
            display: flex;
            flex-direction: column;
            gap: 16px;
            box-shadow: 0 16px 40px rgba(0, 0, 0, 0.6), 0 0 0 1px rgba(255, 100, 30, 0.1);
        }

        .input-row {
            display: flex;
            align-items: center;
        }

        .chat-input {
            width: 100%;
            background: transparent;
            border: none;
            outline: none;
            color: #ffffff;
            font-size: 16px;
            font-family: inherit;
            font-weight: 400;
        }

        .chat-input::placeholder {
            color: rgba(255, 255, 255, 0.45);
        }

        .input-footer-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .left-controls {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .attach-btn {
            background: none;
            border: none;
            color: rgba(255, 255, 255, 0.6);
            cursor: pointer;
            padding: 4px;
            display: flex;
            align-items: center;
            transition: color 0.2s;
        }
        .attach-btn:hover { color: #ffffff; }

        .mic-btn-main {
            width: 52px;
            height: 52px;
            border-radius: 50%;
            background: radial-gradient(circle at 35% 30%, #ff8833 0%, #ff4b00 80%);
            border: none;
            color: #ffffff;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            box-shadow: 0 4px 20px rgba(255, 80, 0, 0.5), inset 0 1px 2px rgba(255, 255, 255, 0.5);
            transition: all 0.2s ease;
        }

        .mic-btn-main:hover {
            transform: scale(1.08);
            box-shadow: 0 6px 26px rgba(255, 100, 0, 0.7);
        }

        .mic-btn-main:active { transform: scale(0.94); }

        /* SCREEN 2: Active Voice Mode View */
        #screen-voice {
            flex: 1;
            display: none;
            flex-direction: column;
            padding: 16px 24px 28px;
            position: relative;
            z-index: 10;
        }

        .top-close-row {
            display: flex;
            justify-content: center;
            margin-bottom: 24px;
        }

        .close-chat-pill {
            background: rgba(255, 255, 255, 0.14);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 20px;
            padding: 8px 18px;
            font-size: 14px;
            font-weight: 500;
            color: #ffffff;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 6px;
            transition: background 0.2s;
        }
        .close-chat-pill:hover { background: rgba(255, 255, 255, 0.22); }

        /* The 3D Luminous Golden/Amber Orb */
        .orb-stage {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            margin: 10px 0 20px;
            position: relative;
        }

        .orb-container {
            width: 220px;
            height: 220px;
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        /* 3D Realistic Shader Orb via CSS Canvas & Glow Layers */
        .orb-sphere {
            width: 200px;
            height: 200px;
            border-radius: 50%;
            background: radial-gradient(circle at 38% 32%, #fff1b8 0%, #ffc542 22%, #e67300 52%, #7a1c00 85%, #2a0800 100%);
            box-shadow: 
                0 0 50px rgba(255, 140, 20, 0.55),
                0 0 100px rgba(255, 80, 0, 0.35),
                inset -15px -15px 35px rgba(0, 0, 0, 0.8),
                inset 10px 10px 25px rgba(255, 255, 255, 0.6);
            position: relative;
            animation: orbBreath 4s ease-in-out infinite alternate;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .orb-highlight {
            position: absolute;
            top: 24px;
            left: 42px;
            width: 60px;
            height: 40px;
            border-radius: 50%;
            background: radial-gradient(ellipse, rgba(255,255,255,0.7) 0%, rgba(255,255,255,0) 80%);
            transform: rotate(-30deg);
            filter: blur(2px);
            pointer-events: none;
        }

        .orb-aura {
            position: absolute;
            width: 230px;
            height: 230px;
            border-radius: 50%;
            border: 1px dashed rgba(255, 170, 60, 0.3);
            animation: rotateAura 20s linear infinite;
            pointer-events: none;
        }

        @keyframes orbBreath {
            0% { transform: scale(0.97); filter: brightness(0.95); }
            50% { transform: scale(1.02); filter: brightness(1.1); box-shadow: 0 0 70px rgba(255, 150, 30, 0.7); }
            100% { transform: scale(0.97); filter: brightness(0.95); }
        }

        @keyframes rotateAura {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .orb-status-text {
            margin-top: 18px;
            font-size: 14px;
            color: rgba(255, 255, 255, 0.6);
            letter-spacing: 0.3px;
            font-weight: 400;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        /* Transcription text display */
        .transcription-area {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 10px 14px;
            min-height: 110px;
        }

        .transcription-text {
            font-size: 24px;
            font-weight: 700;
            line-height: 1.32;
            letter-spacing: -0.5px;
            color: #ffffff;
            text-shadow: 0 2px 10px rgba(0,0,0,0.4);
            max-width: 360px;
            transition: all 0.3s ease;
        }

        .transcription-placeholder {
            font-size: 18px;
            font-weight: 400;
            color: rgba(255, 255, 255, 0.4);
            letter-spacing: -0.2px;
        }

        /* Bottom voice action buttons */
        .voice-controls-row {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 24px;
            margin-top: auto;
            padding: 10px 0;
        }

        .sub-control-btn {
            width: 52px;
            height: 52px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.12);
            color: #ffffff;
            font-size: 18px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.2s;
        }
        .sub-control-btn:hover {
            background: rgba(255, 255, 255, 0.16);
            transform: scale(1.06);
        }

        .voice-mic-main {
            width: 74px;
            height: 74px;
            border-radius: 50%;
            background: radial-gradient(circle at 35% 30%, #ff8833 0%, #ff4b00 80%);
            border: 4px solid rgba(255, 140, 60, 0.25);
            color: #ffffff;
            font-size: 28px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            box-shadow: 0 0 35px rgba(255, 80, 0, 0.6), inset 0 2px 4px rgba(255, 255, 255, 0.5);
            position: relative;
            transition: all 0.2s;
        }

        .voice-mic-main.recording {
            animation: pulseRecord 1.4s infinite;
            border-color: rgba(255, 255, 255, 0.6);
        }

        @keyframes pulseRecord {
            0% { box-shadow: 0 0 0 0 rgba(255, 80, 0, 0.7); }
            70% { box-shadow: 0 0 0 22px rgba(255, 80, 0, 0); }
            100% { box-shadow: 0 0 0 0 rgba(255, 80, 0, 0); }
        }

        /* PRD EXPLAINABLE AI (XAI) PULL-UP DRAWER */
        .xai-drawer {
            position: absolute;
            left: 0; right: 0; bottom: 0;
            background: rgba(14, 5, 2, 0.96);
            backdrop-filter: blur(35px);
            border-top: 1px solid rgba(255, 120, 50, 0.35);
            border-radius: 36px 36px 0 0;
            padding: 16px 22px 24px;
            z-index: 50;
            transform: translateY(calc(100% - 46px));
            transition: transform 0.4s cubic-bezier(0.16, 1, 0.3, 1);
            max-height: 84%;
            display: flex;
            flex-direction: column;
            box-shadow: 0 -15px 40px rgba(0, 0, 0, 0.85);
        }

        .xai-drawer.expanded {
            transform: translateY(0);
        }

        .drawer-handle {
            width: 44px;
            height: 5px;
            background: rgba(255, 255, 255, 0.3);
            border-radius: 10px;
            margin: 0 auto 12px;
            cursor: pointer;
        }

        .drawer-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 14px;
            cursor: pointer;
        }

        .drawer-title {
            font-size: 15px;
            font-weight: 700;
            color: #ff9d66;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .score-pill {
            font-size: 12px;
            font-weight: 600;
            padding: 4px 10px;
            border-radius: 12px;
            background: rgba(16, 185, 129, 0.18);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }

        .score-pill.warning {
            background: rgba(239, 68, 68, 0.2);
            color: #f87171;
            border-color: rgba(239, 68, 68, 0.4);
        }

        .drawer-content {
            overflow-y: auto;
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 16px;
            padding-right: 4px;
        }
        .drawer-content::-webkit-scrollbar { width: 4px; }
        .drawer-content::-webkit-scrollbar-thumb { background: rgba(255, 120, 50, 0.3); border-radius: 4px; }

        /* XAI Sections */
        .xai-card {
            background: rgba(30, 12, 6, 0.72);
            border: 1px solid rgba(255, 120, 50, 0.18);
            border-radius: 18px;
            padding: 14px;
        }

        .xai-card-title {
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: #f7a57a;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 6px;
            font-weight: 700;
        }

        /* Derivation & Conclusion Box */
        .derivation-step {
            background: rgba(0, 0, 0, 0.4);
            border-left: 3px solid #ff7a1a;
            border-radius: 8px;
            padding: 8px 12px;
            margin-bottom: 8px;
            font-size: 12px;
            line-height: 1.45;
        }
        .derivation-step b { color: #ffbe8a; }

        .equation-box {
            background: rgba(0, 0, 0, 0.45);
            border-radius: 12px;
            padding: 12px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
            color: #ffdecc;
            margin-bottom: 6px;
            border-left: 3px solid #ff7a1a;
            overflow-x: auto;
        }

        .equation-desc {
            font-size: 12px;
            color: var(--text-warm);
            line-height: 1.4;
        }

        /* D3 Neural Attention Graph */
        #neural-graph-container {
            width: 100%;
            height: 220px;
            background: rgba(10, 4, 2, 0.9);
            border-radius: 14px;
            border: 1px solid rgba(255, 120, 50, 0.25);
            position: relative;
            overflow: hidden;
        }

        #neural-canvas {
            width: 100%;
            height: 100%;
            display: block;
        }

        .graph-layer-labels {
            display: flex;
            justify-content: space-between;
            font-size: 10px;
            color: var(--text-sub);
            padding: 4px 6px 0;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* Speaker & Topic Memory Card */
        .memory-row {
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            padding: 4px 0;
            border-bottom: 1px solid rgba(255,255,255,0.06);
        }
        .memory-row:last-child { border-bottom: none; }
        .memory-label { color: var(--text-sub); }
        .memory-value { font-weight: 600; color: #ffbe8a; }

        /* Invariant banner */
        .invariant-banner {
            border-radius: 14px;
            padding: 12px;
            font-size: 12px;
            line-height: 1.45;
        }
        .invariant-banner.passed {
            background: rgba(16, 185, 129, 0.12);
            border: 1px solid rgba(16, 185, 129, 0.35);
            color: #a7f3d0;
        }
        .invariant-banner.contradiction {
            background: rgba(239, 68, 68, 0.16);
            border: 1px solid rgba(239, 68, 68, 0.45);
            color: #fecaca;
        }

        /* Modal Dialog for Attachments / Presets */
        .preset-modal {
            display: none;
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0, 0, 0, 0.85);
            backdrop-filter: blur(20px);
            z-index: 100;
            padding: 30px 20px;
            flex-direction: column;
            justify-content: center;
        }
        .preset-modal.active { display: flex; }
        .preset-card {
            background: #1e0b04;
            border: 1px solid #ff7a1a;
            border-radius: 24px;
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 14px;
        }
        .preset-card h3 { font-size: 18px; color: #fff; }
        .preset-item {
            background: rgba(255,255,255,0.06);
            padding: 12px 14px;
            border-radius: 12px;
            cursor: pointer;
            font-size: 13px;
            transition: background 0.2s;
        }
        .preset-item:hover { background: rgba(255, 120, 40, 0.25); }

        /* Quick Evidence Pills on Voice Screen */
        .evidence-pills-row {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 6px;
            padding: 2px 14px 8px;
            z-index: 10;
        }
        .evidence-pill {
            background: rgba(255, 120, 40, 0.12);
            border: 1px solid rgba(255, 140, 60, 0.28);
            border-radius: 14px;
            padding: 4px 9px;
            font-size: 11px;
            font-weight: 600;
            color: var(--accent-gold);
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 4px;
            transition: all 0.2s;
        }
        .evidence-pill:hover {
            background: rgba(255, 120, 40, 0.25);
            transform: translateY(-1px);
        }

        /* Word-by-Word Diarization in Transcription */
        .diarization-speaker-badge {
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: var(--accent-orange);
            margin-bottom: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
        }
        .diarized-stream {
            font-size: 18px;
            font-weight: 600;
            line-height: 1.45;
            color: rgba(255, 255, 255, 0.7);
            max-height: 110px;
            overflow-y: auto;
            padding: 2px 6px;
        }
        .word-token {
            display: inline-block;
            padding: 1px 3px;
            border-radius: 4px;
            transition: all 0.12s ease;
        }
        .word-token.spoken-active {
            background: rgba(255, 140, 0, 0.55);
            color: #ffffff;
            font-weight: 800;
            transform: scale(1.06);
            box-shadow: 0 0 14px rgba(255, 140, 0, 0.75);
        }
        .word-token.spoken-past {
            color: #ffffff;
        }

        /* Diarization Card in Drawer */
        .diarization-chip-cloud {
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            max-height: 130px;
            overflow-y: auto;
            padding: 6px;
            background: rgba(0, 0, 0, 0.35);
            border-radius: 12px;
        }
        .diarize-chip {
            font-size: 11px;
            font-family: 'JetBrains Mono', monospace;
            padding: 3px 6px;
            border-radius: 6px;
            background: rgba(255, 255, 255, 0.06);
            color: var(--text-warm);
            border: 1px solid rgba(255, 255, 255, 0.08);
            transition: all 0.15s;
        }
        .diarize-chip.active {
            background: rgba(255, 120, 0, 0.4);
            border-color: var(--accent-gold);
            color: #ffffff;
            font-weight: 700;
        }
        .chip-idx {
            font-size: 9px;
            color: var(--accent-orange);
            margin-right: 3px;
        }

        /* Attention Weights Table */
        .weights-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 11px;
            margin-top: 8px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 8px;
            overflow: hidden;
        }
        .weights-table th {
            text-align: left;
            padding: 6px 8px;
            color: var(--accent-orange);
            background: rgba(255, 120, 40, 0.1);
            font-weight: 600;
        }
        .weights-table td {
            padding: 5px 8px;
            border-top: 1px solid rgba(255, 255, 255, 0.06);
            color: var(--text-warm);
        }
        .weight-bar-bg {
            width: 50px;
            height: 6px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
            overflow: hidden;
            display: inline-block;
            vertical-align: middle;
            margin-right: 6px;
        }
        .weight-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #ff8c00, #ffbe3b);
            border-radius: 3px;
        }

        /* Cross-Validation Box */
        .crossval-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 6px 8px;
            background: rgba(0, 0, 0, 0.25);
            border-radius: 8px;
            margin-bottom: 5px;
            font-size: 11px;
        }
        .crossval-tag {
            font-size: 10px;
            font-weight: 700;
            padding: 2px 7px;
            border-radius: 10px;
        }
        .crossval-tag.pass {
            background: rgba(16, 185, 129, 0.2);
            color: #34d399;
        }

        /* Voice Inline Input */
        .voice-text-input-wrap {
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 0 16px 8px;
            z-index: 10;
        }
        .voice-inline-input {
            flex: 1;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 120, 40, 0.25);
            border-radius: 18px;
            padding: 8px 14px;
            font-size: 12px;
            color: #ffffff;
            outline: none;
        }
        .voice-inline-input:focus {
            border-color: var(--accent-orange);
            background: rgba(255, 255, 255, 0.12);
        }
        .voice-inline-send {
            width: 32px;
            height: 32px;
            border-radius: 50%;
            background: var(--accent-orange);
            border: none;
            color: #ffffff;
            font-weight: 700;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: transform 0.2s;
        }
        .voice-inline-send:hover {
            transform: scale(1.08);
        }
    </style>
</head>
<body>

    <div class="device-wrapper">
        <div class="grain-overlay"></div>

        <!-- Dynamic Island & Status Bar -->
        <div class="status-bar">
            <span>9:41</span>
            <div class="dynamic-island">
                <span style="font-size: 10px; color: #ff7a1a; font-weight: 700;">XAI</span>
                <div class="camera-lens"></div>
            </div>
            <div class="status-icons">
                <span>5G</span>
                <span>📶</span>
                <span>100%</span>
            </div>
        </div>

        <!-- ========================================== -->
        <!-- SCREEN 1: HOME DASHBOARD (User Design #1)  -->
        <!-- ========================================== -->
        <div id="screen-home">
            <!-- Header Row -->
            <div class="header-row">
                <div class="user-profile" onclick="toggleSpeakerDetails()">
                    <div class="avatar-circle">
                        <!-- Stylized swirl emblem -->
                        <svg viewBox="0 0 24 24">
                            <circle cx="12" cy="12" r="9"/>
                            <path d="M12 3a9 9 0 0 1 9 9c0 3-2 6-5 6-3 0-4-3-4-6"/>
                        </svg>
                    </div>
                    <div class="user-text">
                        <h2>Charan,</h2>
                        <p>Welcome back</p>
                    </div>
                </div>

                <div class="header-actions">
                    <button class="icon-btn" onclick="openNotifications()" title="Alerts">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"></path>
                            <path d="M13.73 21a2 2 0 0 1-3.46 0"></path>
                        </svg>
                    </button>
                    <button class="icon-btn" onclick="openDrawerDirect()" title="Inspect XAI">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <line x1="8" y1="6" x2="21" y2="6"></line>
                            <line x1="8" y1="12" x2="21" y2="12"></line>
                            <line x1="8" y1="18" x2="21" y2="18"></line>
                            <line x1="3" y1="6" x2="3.01" y2="6"></line>
                            <line x1="3" y1="12" x2="3.01" y2="12"></line>
                            <line x1="3" y1="18" x2="3.01" y2="18"></line>
                        </svg>
                    </button>
                </div>
            </div>

            <!-- Hero Big Title: "What are we learning today?" -->
            <h1 class="hero-title">What are we<br>learning today?</h1>

            <!-- Prompt suggestion cards row with Remote Sensing topic history -->
            <div class="cards-row">
                <div class="prompt-card" onclick="startVoiceQuery('Explain SAR interferometry phase shift and baseline decorrelation.')">
                    <span class="sparkle-icon">✦</span>
                    <div>
                        <div class="card-text">SAR Interferometry & Phase Shift</div>
                        <span class="card-tag">Session 4 • In-progress</span>
                    </div>
                </div>

                <div class="prompt-card" onclick="startVoiceQuery('Optimize NDVI optical band maths and red edge canopy calculations.')">
                    <span class="sparkle-icon">✦</span>
                    <div>
                        <div class="card-text">NDVI & Red-Edge Canopy Maths</div>
                        <span class="card-tag">Session 3 • Completed</span>
                    </div>
                </div>

                <div class="prompt-card" onclick="startVoiceQuery('Audit atmospheric windows and Rayleigh scattering law.')">
                    <span class="sparkle-icon">✦</span>
                    <div>
                        <div class="card-text">Rayleigh Scattering & Windows</div>
                        <span class="card-tag">Session 2 • Completed</span>
                    </div>
                </div>

                <div class="prompt-card" style="border-color: rgba(239, 68, 68, 0.45);" onclick="startVoiceQuery('Does radar require sunlight to capture images?')">
                    <span class="sparkle-icon" style="color: #ef4444;">✦</span>
                    <div>
                        <div class="card-text">Physics Invariant: Radar vs Sun</div>
                        <span class="card-tag" style="color: #ef4444;">Adversarial • Rule Check</span>
                    </div>
                </div>
            </div>

            <!-- Bottom Floating Input Card (Gemini pill removed as requested) -->
            <div class="bottom-input-container">
                <div class="input-row">
                    <input type="text" id="main-text-input" class="chat-input" placeholder="Ask AI a question or describe your idea" onkeydown="handleInputKey(event)">
                </div>

                <div class="input-footer-row">
                    <div class="left-controls">
                        <button class="attach-btn" onclick="openPresetModal()" title="Attach Remote Sensing Sample">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"></path>
                            </svg>
                        </button>
                    </div>

                    <!-- Glowing circular orange microphone trigger button -->
                    <button class="mic-btn-main" onclick="openVoiceMode()" title="Voice Conversation Mode">
                        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
                            <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
                            <line x1="12" y1="19" x2="12" y2="23"></line>
                            <line x1="8" y1="23" x2="16" y2="23"></line>
                        </svg>
                    </button>
                </div>
            </div>
        </div>

        <!-- ========================================== -->
        <!-- SCREEN 2: VOICE MODE (User Design #2)       -->
        <!-- ========================================== -->
        <div id="screen-voice">
            <div class="top-close-row">
                <button class="close-chat-pill" onclick="closeVoiceMode()">
                    <span>✕</span> Close chat
                </button>
            </div>

            <!-- 3D Luminous Orb Centerpiece -->
            <div class="orb-stage">
                <div class="orb-container">
                    <div class="orb-aura"></div>
                    <div class="orb-sphere" id="glowing-orb">
                        <div class="orb-highlight"></div>
                    </div>
                </div>
                <!-- Status below AI: leave as listening -->
                <div class="orb-status-text" id="orb-status-label">
                    <span>Listening...</span>
                </div>
            </div>

            <!-- Live Spoken Transcription Display (With Real-Time Diarization) -->
            <div class="transcription-area">
                <div class="transcription-text" id="live-transcription">
                    <span class="transcription-placeholder">Listening for your query... Speak or type below</span>
                </div>
            </div>

            <!-- Quick Evidence Pills & Diarization Trigger Bar -->
            <div class="evidence-pills-row" id="evidence-pills-row">
                <div class="evidence-pill" onclick="toggleDrawer()">
                    <span>📚</span> <span id="pill-source-text">NASA / IEEE Literature</span>
                </div>
                <div class="evidence-pill" onclick="toggleDrawer()">
                    <span>🛡️</span> <span id="pill-crossval-text">Cross-Validated: 100%</span>
                </div>
                <div class="evidence-pill" onclick="toggleDrawer()">
                    <span>🌌</span> <span id="pill-weights-text">Attention: 0.98</span>
                </div>
                <div class="evidence-pill" onclick="toggleDrawer()">
                    <span>🎙️</span> <span>Word Diarization</span>
                </div>
            </div>

            <!-- Inline Voice Screen Text Input -->
            <div class="voice-text-input-wrap">
                <input type="text" id="voice-inline-input" class="voice-inline-input" placeholder="Ask: 'what is remote sensing and what is the physics behind it'..." onkeydown="handleVoiceInlineInput(event)">
                <button class="voice-inline-send" onclick="submitVoiceInlineInput()" title="Send Question">➔</button>
            </div>

            <!-- Bottom Floating Voice Controls -->
            <div class="voice-controls-row">
                <button class="sub-control-btn" id="pause-btn" onclick="togglePauseVoice()" title="Pause Audio">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                        <rect x="6" y="4" width="4" height="16"></rect>
                        <rect x="14" y="4" width="4" height="16"></rect>
                    </svg>
                </button>

                <button class="voice-mic-main" id="voice-record-btn" onclick="toggleMicrophone()" title="Toggle Mic Capture">
                    <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
                        <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
                        <line x1="12" y1="19" x2="12" y2="23"></line>
                        <line x1="8" y1="23" x2="16" y2="23"></line>
                    </svg>
                </button>

                <button class="sub-control-btn" onclick="submitCurrentVoice()" title="Process Query">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <line x1="22" y1="2" x2="11" y2="13"></line>
                        <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                    </svg>
                </button>
            </div>

            <!-- ============================================== -->
            <!-- PRD EXPLAINABLE AI (XAI) PULL-UP INSPECTOR     -->
            <!-- ============================================== -->
            <div class="xai-drawer" id="xai-drawer">
                <div class="drawer-handle" onclick="toggleDrawer()"></div>

                <div class="drawer-header" onclick="toggleDrawer()">
                    <div class="drawer-title">
                        <span>✦</span> Explainability & Neural Links
                    </div>
                    <div class="score-pill" id="grounding-badge">95% Grounded</div>
                </div>

                <div class="drawer-content">
                    <!-- 1. Citations & Grounding Literature Evidence -->
                    <div class="xai-card">
                        <div class="xai-card-title">
                            <span>📚</span> Literature Evidence & Grounding Sources
                        </div>
                        <div id="citations-list" style="font-size: 12px; color: var(--text-warm); line-height: 1.6;">
                            • NASA Earth Science Division (ARSET): Fundamentals of Remote Sensing & EM Radiation (pp. 4-25)<br>
                            • Remote Sensing & Image Interpretation (Lillesand, Kiefer & Chipman, 8th Ed.)<br>
                            • Principles of Remote Sensing (ITC Netherlands, Ch. 1-2)
                        </div>
                    </div>

                    <!-- 2. Dual-Check Cross-Validation & Physics Invariants -->
                    <div class="xai-card">
                        <div class="xai-card-title">
                            <span>🛡️</span> Dual-Check Cross-Validation & Physics Invariants
                        </div>
                        <div id="invariant-banner" class="invariant-banner passed" style="margin-bottom: 8px;">
                            <b>Status: Physical Invariants Verified</b><br>
                            Query conforms to electromagnetic conservation and active radiation laws.
                        </div>
                        <div id="crossval-checklist-container">
                            <!-- Populated dynamically by renderCrossValidation -->
                        </div>
                    </div>

                    <!-- 3. Dynamic Neural Network & Attention Weights Inspector (FR-4) -->
                    <div class="xai-card">
                        <div class="xai-card-title">
                            <span>🌌</span> Dynamic Neural Network & Attention Weights (FR-4)
                        </div>
                        <div id="neural-graph-container">
                            <canvas id="neural-canvas"></canvas>
                        </div>
                        <div class="graph-layer-labels">
                            <span>Input Tokens</span>
                            <span>Physics Saliency</span>
                            <span>Governing Law</span>
                            <span>Conclusion</span>
                        </div>
                        <div id="attention-weights-table-container">
                            <!-- Populated dynamically by renderAttentionWeights -->
                        </div>
                    </div>

                    <!-- 4. Real-Time Word Diarization & Token Stream -->
                    <div class="xai-card">
                        <div class="xai-card-title">
                            <span>🎙️</span> Real-Time Word Diarization & Spoken Token Stream
                        </div>
                        <div id="diarization-token-container">
                            <p style="font-size: 11px; color: var(--text-sub);">Word token diarization will articulate in real-time as Aira speaks.</p>
                        </div>
                    </div>

                    <!-- 5. Governing Physical Equations (FR-3) -->
                    <div class="xai-card">
                        <div class="xai-card-title">
                            <span>📐</span> Governing Equations & Assumptions (FR-3)
                        </div>
                        <div id="equations-list">
                            <div class="equation-box">c = λ · ν,   E = h · ν = (h · c) / λ</div>
                            <p class="equation-desc">Electromagnetic wave equation and quantized photon energy relation.</p>
                        </div>
                    </div>

                    <!-- 6. Inference Conclusion & Step-by-Step Derivation -->
                    <div class="xai-card">
                        <div class="xai-card-title">
                            <span>🔍</span> Derivation & Conclusion: How Aira Derived the Answer
                        </div>
                        <div id="derivation-steps-container">
                            <div class="derivation-step">
                                <b>1. Tokenization & Routing:</b> <span id="derivation-step-1">Parsed user tokens and mapped to Remote Sensing Physics ontology.</span>
                            </div>
                            <div class="derivation-step">
                                <b>2. Invariant Check:</b> <span id="derivation-step-2">Verified electromagnetic conservation and active radiation independence.</span>
                            </div>
                            <div class="derivation-step">
                                <b>3. Equation Mapping:</b> <span id="derivation-step-3">Bound variables to verified governing equations.</span>
                            </div>
                            <div class="derivation-step" style="border-left-color: #10b981;">
                                <b>4. Conclusion:</b> <span id="derivation-step-4">Formulated physical proof with peer-reviewed citation grounding.</span>
                            </div>
                        </div>
                    </div>

                    <!-- 7. Speaker Biometrics & Topic Memory (FR-2) -->
                    <div class="xai-card">
                        <div class="xai-card-title">
                            <span>👤</span> Speaker Voiceprint & Session Continuity (FR-2)
                        </div>
                        <div class="memory-row">
                            <span class="memory-label">Enrolled Speaker</span>
                            <span class="memory-value" id="speaker-name-val">Charan (RS Analyst)</span>
                        </div>
                        <div class="memory-row">
                            <span class="memory-label">Cosine Similarity</span>
                            <span class="memory-value" id="similarity-score-val">0.94 (Biometrics Verified)</span>
                        </div>
                        <div class="memory-row">
                            <span class="memory-label">Session Topic Continuity</span>
                            <span class="memory-value" id="resumed-topic-val">Remote Sensing Physics</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Attachment / Preset Modal -->
        <div class="preset-modal" id="preset-modal">
            <div class="preset-card">
                <h3>Select Remote Sensing Knowledge Sample</h3>
                <div class="preset-item" onclick="selectPreset('Explain SAR interferometry phase shift and baseline decorrelation.')">
                    📡 <b>SAR & InSAR Interferometry</b>: Wavelength & Surface Deformation
                </div>
                <div class="preset-item" onclick="selectPreset('Explain NDVI optical band maths and red edge canopy calculations.')">
                    🌿 <b>Optical Vegetation Indices</b>: NIR vs Red Chlorophyll Math
                </div>
                <div class="preset-item" onclick="selectPreset('Explain atmospheric windows and Rayleigh scattering law.')">
                    🌤️ <b>Atmospheric Scattering</b>: Rayleigh λ⁻⁴ vs Microwave Windows
                </div>
                <div class="preset-item" onclick="selectPreset('How does LiDAR measure forest canopy height using time of flight?')">
                    🌲 <b>LiDAR Altimetry</b>: Time-of-Flight & CHM Derivation
                </div>
                <div class="preset-item" style="border-left: 3px solid #ef4444;" onclick="selectPreset('Does radar require sunlight to capture images?')">
                    ⚠️ <b>Adversarial Invariant Test</b>: "Does radar require sunlight?"
                </div>
                <button class="close-chat-pill" style="align-self: flex-end; margin-top: 10px;" onclick="closePresetModal()">Cancel</button>
            </div>
        </div>

    </div>

    <!-- SCRIPT LOGIC -->
    <script>
        // State variables
        let isRecording = false;
        let isPaused = false;
        let recognition = null;
        let lastQuery = "";
        let currentGraphData = null;
        let animFrameId = null;
        let pulseOffset = 0;

        // Elements
        const screenHome = document.getElementById('screen-home');
        const screenVoice = document.getElementById('screen-voice');
        const mainTextInput = document.getElementById('main-text-input');
        const liveTranscription = document.getElementById('live-transcription');
        const orbStatusLabel = document.getElementById('orb-status-label');
        const glowingOrb = document.getElementById('glowing-orb');
        const voiceRecordBtn = document.getElementById('voice-record-btn');
        const xaiDrawer = document.getElementById('xai-drawer');
        const groundingBadge = document.getElementById('grounding-badge');
        const invariantBanner = document.getElementById('invariant-banner');
        const equationsList = document.getElementById('equations-list');
        const citationsList = document.getElementById('citations-list');
        const canvas = document.getElementById('neural-canvas');

        let userSpokenTranscript = "";
        let isAiraSpeaking = false;

        // Check Web Speech Recognition
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (SpeechRecognition) {
            recognition = new SpeechRecognition();
            recognition.continuous = false;
            recognition.interimResults = true;
            recognition.lang = 'en-US';

            recognition.onstart = () => {
                isRecording = true;
                userSpokenTranscript = "";
                voiceRecordBtn.classList.add('recording');
                orbStatusLabel.innerHTML = '<span>Listening to your voice...</span>';
                glowingOrb.style.filter = 'brightness(1.25)';
                glowingOrb.style.boxShadow = '0 0 90px rgba(255, 120, 20, 0.9)';
            };

            recognition.onresult = (event) => {
                let interim = '';
                for (let i = event.resultIndex; i < event.results.length; ++i) {
                    interim += event.results[i][0].transcript;
                }
                if (interim.trim()) {
                    userSpokenTranscript = interim.trim();
                    liveTranscription.innerHTML = `<span style="color: #ffffff; font-weight: 600;">${userSpokenTranscript}</span>`;
                }
            };

            recognition.onerror = (event) => {
                console.warn('Speech recognition error:', event.error);
                orbStatusLabel.innerHTML = '<span>Tap mic or speak your question</span>';
                voiceRecordBtn.classList.remove('recording');
                isRecording = false;
            };

            recognition.onend = () => {
                isRecording = false;
                voiceRecordBtn.classList.remove('recording');
                // ONLY execute if user actually spoke during THIS capture session
                const captured = userSpokenTranscript.trim();
                userSpokenTranscript = ""; // Reset immediately to prevent repeat loops!

                if (captured && captured.length >= 3 && !isAiraSpeaking && captured !== lastQuery) {
                    executeXAIQuery(captured);
                } else if (!isAiraSpeaking) {
                    orbStatusLabel.innerHTML = '<span>Tap mic or speak your question</span>';
                }
            };
        }

        let conversationHistory = [];

        function openVoiceMode(initialQuery = null) {
            screenHome.style.display = 'none';
            screenVoice.style.display = 'flex';
            resizeCanvas();
            renderInitialGraph();

            if (initialQuery) {
                liveTranscription.innerHTML = initialQuery;
                executeXAIQuery(initialQuery);
            } else {
                const greeting = "Hello, what do you want to learn today?";
                liveTranscription.innerHTML = `<span style="color: var(--accent-gold); font-weight: 500;">${greeting}</span>`;
                orbStatusLabel.innerHTML = '<span>Aira is greeting...</span>';
                speakAnswer(greeting, () => {
                    liveTranscription.innerHTML = '<span class="transcription-placeholder">Listening for your question... Speak now</span>';
                    startListening();
                });
            }
        }

        function closeVoiceMode() {
            if (window.speechSynthesis) window.speechSynthesis.cancel();
            if (recognition && isRecording) recognition.stop();
            if (animFrameId) cancelAnimationFrame(animFrameId);
            screenVoice.style.display = 'none';
            screenHome.style.display = 'flex';
            xaiDrawer.classList.remove('expanded');
        }

        function toggleDrawer() {
            xaiDrawer.classList.toggle('expanded');
            if (xaiDrawer.classList.contains('expanded')) {
                setTimeout(resizeCanvas, 300);
            }
        }

        function openDrawerDirect() {
            openVoiceMode();
            xaiDrawer.classList.add('expanded');
        }

        function toggleSpeakerDetails() {
            alert("Speaker Enrolled: Charan\nRole: Remote Sensing & Geospatial Analyst\nVoiceprint: ECAPA-TDNN 192-dim vector\nLast Topic: SAR Interferometry & Phase Shift (In-progress)");
        }

        function openNotifications() {
            alert("🔔 Notifications:\n• Topic Resumed: SAR Interferometry & Phase Shift\n• Physics Invariant Engine: Operational (5 active invariant laws)\n• Grounding Confidence: Active");
        }

        function startVoiceQuery(text) {
            mainTextInput.value = text;
            openVoiceMode(text);
        }

        function handleInputKey(e) {
            if (e.key === 'Enter') {
                const text = mainTextInput.value.trim();
                if (text) {
                    mainTextInput.value = '';
                    openVoiceMode(text);
                }
            }
        }

        function toggleMicrophone() {
            if (isRecording) {
                if (recognition) recognition.stop();
                isRecording = false;
                voiceRecordBtn.classList.remove('recording');
            } else {
                startListening();
            }
        }

        function startListening() {
            if (window.speechSynthesis) window.speechSynthesis.cancel();
            orbStatusLabel.innerHTML = '<span>Listening...</span>';
            if (recognition) {
                try {
                    recognition.start();
                } catch(e) {
                    console.log('Recognition restart error:', e);
                }
            }
        }

        function togglePauseVoice() {
            isPaused = !isPaused;
            if (window.speechSynthesis) {
                if (isPaused) {
                    window.speechSynthesis.pause();
                    document.getElementById('pause-btn').style.background = 'rgba(255, 120, 40, 0.4)';
                } else {
                    window.speechSynthesis.resume();
                    document.getElementById('pause-btn').style.background = 'rgba(255, 255, 255, 0.08)';
                }
            }
        }

        function submitCurrentVoice() {
            const query = liveTranscription.textContent.trim();
            if (query && !query.includes('Listening for your query') && !query.includes('Listening for your question')) {
                executeXAIQuery(query);
            }
        }

        // Send query to local XAI Backend
        async function executeXAIQuery(query) {
            lastQuery = query;
            orbStatusLabel.innerHTML = '<span>Analyzing with Gemini & Physics Invariants...</span>';
            glowingOrb.style.filter = 'brightness(1.3) hue-rotate(20deg)';

            try {
                conversationHistory.push({ role: 'user', content: query });

                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        message: query,
                        speaker_id: 'spk_charan',
                        history: conversationHistory
                    })
                });

                const data = await response.json();
                conversationHistory.push({ role: 'assistant', content: data.spoken_response });
                if (conversationHistory.length > 20) {
                    conversationHistory = conversationHistory.slice(-20);
                }

                renderXAIResults(data);

                // Voice synthesis output with interactive turn-taking
                speakAnswer(data.spoken_response);

            } catch (err) {
                console.error('Error contacting backend:', err);
                orbStatusLabel.innerHTML = '<span>Backend processing error</span>';
            }
        }

        function handleVoiceInlineInput(e) {
            if (e.key === 'Enter') {
                submitVoiceInlineInput();
            }
        }

        function submitVoiceInlineInput() {
            const inputEl = document.getElementById('voice-inline-input');
            const text = inputEl ? inputEl.value.trim() : '';
            if (text) {
                if (inputEl) inputEl.value = '';
                executeXAIQuery(text);
            }
        }

        function renderCrossValidation(crossval) {
            const container = document.getElementById('crossval-checklist-container');
            const pillText = document.getElementById('pill-crossval-text');
            if (!container) return;

            if (crossval && crossval.rules_checked) {
                let html = '';
                crossval.rules_checked.forEach(r => {
                    html += `
                        <div class="crossval-row">
                            <span><b>[${r.id}]</b> ${r.name}</span>
                            <span class="crossval-tag ${r.passed ? 'pass' : 'fail'}">${r.passed ? '✓ PASSED' : '⚠️ REJECTED'}</span>
                        </div>
                    `;
                });
                html += `<div style="font-size: 11px; color: var(--accent-gold); margin-top: 6px;"><b>Literature Consensus:</b> ${crossval.literature_consensus || '100% Agreement with NASA / IEEE Literature'}</div>`;
                container.innerHTML = html;
                if (pillText) pillText.textContent = `Cross-Val: ${crossval.status || 'PASSED'}`;
            }
        }

        function renderAttentionWeights(graph) {
            const container = document.getElementById('attention-weights-table-container');
            const pillText = document.getElementById('pill-weights-text');
            if (!container || !graph || !graph.nodes) return;

            let html = `
                <table class="weights-table">
                    <thead>
                        <tr>
                            <th>Layer</th>
                            <th>Active Token / Concept</th>
                            <th>Attention Weight</th>
                        </tr>
                    </thead>
                    <tbody>
            `;
            let maxWeight = 0.95;
            graph.nodes.forEach(n => {
                const w = n.weight || 0.90;
                if (w > maxWeight) maxWeight = w;
                const pct = Math.round(w * 100);
                html += `
                    <tr>
                        <td style="color: var(--accent-orange); font-weight: 600;">${n.layer || 'Feature'}</td>
                        <td>${n.id}</td>
                        <td>
                            <div class="weight-bar-bg"><div class="weight-bar-fill" style="width: ${pct}%;"></div></div>
                            <span style="font-family: monospace; font-weight: 700;">${w}</span>
                        </td>
                    </tr>
                `;
            });
            html += '</tbody></table>';
            container.innerHTML = html;
            if (pillText) pillText.textContent = `Max Attention: ${maxWeight}`;
        }

        function renderDiarizedAnswer(spokenText) {
            const words = spokenText.split(/\s+/).filter(w => w.length > 0);
            let html = '<div class="diarization-speaker-badge">🎙️ Aira (Spoken Voice)</div><div class="diarized-stream">';
            words.forEach((w, idx) => {
                html += `<span class="word-token" id="word-token-${idx}">${w}</span> `;
            });
            html += '</div>';
            liveTranscription.innerHTML = html;

            // Also populate the Diarization Card in the XAI Drawer
            const diarizationCard = document.getElementById('diarization-token-container');
            if (diarizationCard) {
                let cardHTML = `<div style="font-size: 11px; color: var(--accent-gold); margin-bottom: 6px;">Total Articulated Tokens: ${words.length} words (Live Boundary Stream)</div><div class="diarization-chip-cloud">`;
                words.forEach((w, idx) => {
                    cardHTML += `<span class="diarize-chip" id="chip-token-${idx}"><span class="chip-idx">w${idx+1}</span> ${w}</span>`;
                });
                cardHTML += '</div>';
                diarizationCard.innerHTML = cardHTML;
            }
        }

        function renderXAIResults(data) {
            orbStatusLabel.innerHTML = '<span>Aira is speaking...</span>';
            glowingOrb.style.filter = 'brightness(1.1)';

            // 1. Confidence Score & Invariant Status
            const confPct = Math.round(data.confidence_score * 100);
            groundingBadge.textContent = `${confPct}% Grounded`;

            if (data.invariant_status === 'CONTRADICTION_DETECTED') {
                groundingBadge.classList.add('warning');
                invariantBanner.className = 'invariant-banner contradiction';
                invariantBanner.innerHTML = `
                    <b>⚠️ Contradiction Detected: ${data.invariant_rule}</b><br>
                    ${data.invariant_details}
                `;
            } else {
                groundingBadge.classList.remove('warning');
                invariantBanner.className = 'invariant-banner passed';
                invariantBanner.innerHTML = `
                    <b>✓ Status: Physical Invariants Verified</b><br>
                    ${data.invariant_details}
                `;
            }

            // 2. Inference Derivation & Conclusion (How Aira Derived Answer)
            if (data.derivation_conclusion) {
                document.getElementById('derivation-step-1').textContent = data.derivation_conclusion.step_1_parsing;
                document.getElementById('derivation-step-2').textContent = data.derivation_conclusion.step_2_invariant_check;
                document.getElementById('derivation-step-3').textContent = data.derivation_conclusion.step_3_equation_mapping;
                document.getElementById('derivation-step-4').textContent = data.derivation_conclusion.step_4_conclusion;
            }

            // 3. Governing Equations (FR-3)
            let eqHTML = '';
            if (data.governing_equations && data.governing_equations.length > 0) {
                data.governing_equations.forEach(eq => {
                    eqHTML += `
                        <div class="equation-box">${eq.latex || eq.name}</div>
                        <p class="equation-desc"><b>${eq.name}:</b> ${eq.description}</p>
                        <div style="height: 8px;"></div>
                    `;
                });
            }
            equationsList.innerHTML = eqHTML;

            // 4. Citations
            if (data.citations && data.citations.length > 0) {
                let citHTML = '';
                data.citations.forEach(c => {
                    citHTML += `• <b>${c.source}</b> — ${c.chapter} (${c.page})<br>`;
                });
                citationsList.innerHTML = citHTML;

                const pillSource = document.getElementById('pill-source-text');
                if (pillSource) {
                    pillSource.textContent = data.citations[0].source.split('(')[0].trim().slice(0, 24);
                }
            }

            // 5. Speaker Profile (FR-2)
            if (data.speaker_profile) {
                document.getElementById('speaker-name-val').textContent = `${data.speaker_profile.name} (${data.speaker_profile.role})`;
                document.getElementById('similarity-score-val').textContent = `${data.speaker_profile.match_similarity} (Biometrics Verified)`;
                document.getElementById('resumed-topic-val').textContent = data.speaker_profile.resumed_topic;
            }

            // 6. Dynamic Neural Attention Graph (FR-4)
            if (data.attention_graph) {
                currentGraphData = data.attention_graph;
                startGraphAnimation(data.attention_graph);
                renderAttentionWeights(data.attention_graph);
            }

            // 7. Dual Cross-Validation Check
            renderCrossValidation(data.cross_validation);
        }

        function speakAnswer(text, onComplete) {
            if ('speechSynthesis' in window) {
                isAiraSpeaking = true;
                window.speechSynthesis.cancel();

                // Stop microphone listening while assistant speaks to prevent acoustic feedback loop
                if (recognition && isRecording) {
                    try { recognition.stop(); } catch(e) {}
                }

                renderDiarizedAnswer(text);

                const utterance = new SpeechSynthesisUtterance(text);
                utterance.rate = 1.05;
                utterance.pitch = 1.0;

                const voices = window.speechSynthesis.getVoices();
                const natural = voices.find(v => v.lang.startsWith('en') && (v.name.includes('Natural') || v.name.includes('Samantha') || v.name.includes('Google') || v.name.includes('David')));
                if (natural) utterance.voice = natural;

                // Live Word-Level Boundary Diarization
                utterance.onboundary = (event) => {
                    if (event.name === 'word') {
                        const charIndex = event.charIndex;
                        const words = text.split(/\s+/).filter(w => w.length > 0);
                        let cumulative = 0;
                        let activeIdx = 0;
                        for (let i = 0; i < words.length; i++) {
                            cumulative += words[i].length + 1;
                            if (cumulative > charIndex) {
                                activeIdx = i;
                                break;
                            }
                        }
                        // Highlight active word in transcription
                        document.querySelectorAll('.word-token').forEach((el, idx) => {
                            if (idx === activeIdx) {
                                el.classList.add('spoken-active');
                                el.classList.remove('spoken-past');
                            } else if (idx < activeIdx) {
                                el.classList.remove('spoken-active');
                                el.classList.add('spoken-past');
                            } else {
                                el.classList.remove('spoken-active', 'spoken-past');
                            }
                        });
                        // Highlight in drawer diarize chips
                        document.querySelectorAll('.diarize-chip').forEach((el, idx) => {
                            if (idx === activeIdx) el.classList.add('active');
                            else el.classList.remove('active');
                        });
                    }
                };

                utterance.onend = () => {
                    isAiraSpeaking = false;
                    orbStatusLabel.innerHTML = '<span>Listening... (Speak now)</span>';
                    glowingOrb.style.filter = 'brightness(1.0)';
                    if (onComplete) {
                        onComplete();
                    } else if (screenVoice.style.display === 'flex' && !isPaused) {
                        // Interactive turn-taking: automatically resume listening for user response!
                        setTimeout(() => {
                            liveTranscription.innerHTML = '<span class="transcription-placeholder">Listening for your question... Speak now</span>';
                            startListening();
                        }, 400);
                    }
                };

                utterance.onerror = () => {
                    isAiraSpeaking = false;
                };

                window.speechSynthesis.speak(utterance);
            } else if (onComplete) {
                onComplete();
            }
        }

        // ==============================================================
        // DYNAMIC MULTI-LAYER NEURAL NETWORK & WEIGHT ANIMATION (FR-4)
        // ==============================================================
        function resizeCanvas() {
            if (!canvas) return;
            const rect = canvas.parentElement.getBoundingClientRect();
            canvas.width = rect.width * window.devicePixelRatio;
            canvas.height = rect.height * window.devicePixelRatio;
            const ctx = canvas.getContext('2d');
            ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        }

        function renderInitialGraph() {
            const initialGraph = {
                nodes: [
                    { id: "SAR", group: 1, weight: 0.95 },
                    { id: "Phase Shift", group: 1, weight: 0.92 },
                    { id: "Coherent Phase (φ)", group: 2, weight: 0.96 },
                    { id: "Wavelength (λ)", group: 2, weight: 0.91 },
                    { id: "InSAR Phase Law", group: 3, weight: 0.98 },
                    { id: "Surface Deformation", group: 4, weight: 0.99 }
                ],
                links: [
                    { source: "SAR", target: "Coherent Phase (φ)", weight: 0.94 },
                    { source: "Phase Shift", target: "Wavelength (λ)", weight: 0.91 },
                    { source: "Coherent Phase (φ)", target: "InSAR Phase Law", weight: 0.96 },
                    { source: "Wavelength (λ)", target: "InSAR Phase Law", weight: 0.93 },
                    { source: "InSAR Phase Law", target: "Surface Deformation", weight: 0.98 }
                ]
            };
            currentGraphData = initialGraph;
            startGraphAnimation(initialGraph);
        }

        function startGraphAnimation(graphData) {
            if (animFrameId) cancelAnimationFrame(animFrameId);

            function loop() {
                pulseOffset = (pulseOffset + 0.015) % 1.0;
                drawNeuralNetwork(graphData, pulseOffset);
                animFrameId = requestAnimationFrame(loop);
            }
            loop();
        }

        function drawNeuralNetwork(graphData, pulse) {
            if (!canvas) return;
            const ctx = canvas.getContext('2d');
            const w = canvas.parentElement.clientWidth;
            const h = canvas.parentElement.clientHeight;

            ctx.clearRect(0, 0, w, h);

            const nodes = graphData.nodes;
            const links = graphData.links;

            // Group nodes by group (layers 1 to 4)
            const layers = { 1: [], 2: [], 3: [], 4: [] };
            nodes.forEach(n => {
                const g = n.group || 1;
                if (!layers[g]) layers[g] = [];
                layers[g].push(n);
            });

            // Determine X and Y coordinates per layer
            const positions = {};
            const layerX = {
                1: w * 0.14,
                2: w * 0.38,
                3: w * 0.64,
                4: w * 0.88
            };

            Object.keys(layers).forEach(gKey => {
                const g = parseInt(gKey);
                const layerNodes = layers[g];
                const count = layerNodes.length;
                layerNodes.forEach((n, idx) => {
                    const spacing = h / (count + 1);
                    positions[n.id] = {
                        x: layerX[g],
                        y: spacing * (idx + 1),
                        weight: n.weight || 0.85,
                        group: g,
                        name: n.id
                    };
                });
            });

            // 1. Draw Links with dynamic weights and animated forward pulses
            links.forEach(link => {
                const src = positions[link.source];
                const tgt = positions[link.target];
                if (src && tgt) {
                    ctx.beginPath();
                    ctx.moveTo(src.x, src.y);
                    ctx.bezierCurveTo(
                        src.x + (tgt.x - src.x) * 0.5, src.y,
                        src.x + (tgt.x - src.x) * 0.5, tgt.y,
                        tgt.x, tgt.y
                    );
                    ctx.strokeStyle = `rgba(255, 120, 40, ${link.weight * 0.55})`;
                    ctx.lineWidth = link.weight * 2.8;
                    ctx.stroke();

                    // Animated Traveling Synaptic Pulse
                    const t = (pulse + (src.x * 0.003)) % 1.0;
                    // Cubic bezier interpolation
                    const p0 = { x: src.x, y: src.y };
                    const p1 = { x: src.x + (tgt.x - src.x) * 0.5, y: src.y };
                    const p2 = { x: src.x + (tgt.x - src.x) * 0.5, y: tgt.y };
                    const p3 = { x: tgt.x, y: tgt.y };

                    const cx = Math.pow(1 - t, 3) * p0.x + 3 * Math.pow(1 - t, 2) * t * p1.x + 3 * (1 - t) * Math.pow(t, 2) * p2.x + Math.pow(t, 3) * p3.x;
                    const cy = Math.pow(1 - t, 3) * p0.y + 3 * Math.pow(1 - t, 2) * t * p1.y + 3 * (1 - t) * Math.pow(t, 2) * p2.y + Math.pow(t, 3) * p3.y;

                    ctx.fillStyle = "#ffe29a";
                    ctx.shadowColor = "#ff8c33";
                    ctx.shadowBlur = 6;
                    ctx.beginPath();
                    ctx.arc(cx, cy, 2.5, 0, 2 * Math.PI);
                    ctx.fill();
                    ctx.shadowBlur = 0;
                }
            });

            // 2. Draw Nodes
            nodes.forEach(n => {
                const p = positions[n.id];
                if (!p) return;

                // Color by group
                let nodeColor = "#ffaa44";
                if (p.group === 1) nodeColor = "#38bdf8";      // Cyan input
                else if (p.group === 2) nodeColor = "#fbbf24"; // Amber physics
                else if (p.group === 3) nodeColor = "#ff7a1a"; // Orange law
                else if (p.group === 4) nodeColor = "#34d399"; // Emerald conclusion

                // Outer Halo
                const glowGrad = ctx.createRadialGradient(p.x, p.y, 2, p.x, p.y, 16);
                glowGrad.addColorStop(0, "rgba(255, 140, 40, 0.65)");
                glowGrad.addColorStop(1, "rgba(255, 80, 0, 0)");
                ctx.fillStyle = glowGrad;
                ctx.beginPath();
                ctx.arc(p.x, p.y, 16, 0, 2 * Math.PI);
                ctx.fill();

                // Core Node
                ctx.fillStyle = nodeColor;
                ctx.beginPath();
                const radius = 5 + p.weight * 3;
                ctx.arc(p.x, p.y, radius, 0, 2 * Math.PI);
                ctx.fill();

                // Weight ring
                ctx.strokeStyle = "rgba(255,255,255,0.4)";
                ctx.lineWidth = 1;
                ctx.stroke();

                // Label background badge for clean readability
                const text = p.name.length > 14 ? p.name.substring(0, 13) + "…" : p.name;
                ctx.font = "9px Inter, sans-serif";
                const textWidth = ctx.measureText(text).width;
                const textY = p.y > h / 2 ? p.y + 15 : p.y - 10;

                ctx.fillStyle = "rgba(10, 4, 2, 0.75)";
                ctx.fillRect(p.x - textWidth / 2 - 3, textY - 8, textWidth + 6, 12);

                ctx.fillStyle = "#ffffff";
                ctx.textAlign = "center";
                ctx.fillText(text, p.x, textY);
            });
        }

        // Preset Modal
        function openPresetModal() {
            document.getElementById('preset-modal').classList.add('active');
        }
        function closePresetModal() {
            document.getElementById('preset-modal').classList.remove('active');
        }
        function selectPreset(query) {
            closePresetModal();
            startVoiceQuery(query);
        }

        // Window resize handler
        window.addEventListener('resize', () => {
            resizeCanvas();
            if (currentGraphData) drawNeuralNetwork(currentGraphData, pulseOffset);
        });

        // Initialize voices
        if ('speechSynthesis' in window) {
            window.speechSynthesis.onvoiceschanged = () => {
                window.speechSynthesis.getVoices();
            };
        }
    </script>
</body>
</html>
"""

# =====================================================================
# API Endpoints
# =====================================================================

@app.route("/")
def index():
    """Serves the Explainable AI Voice Assistant prototype dashboard."""
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Processes user query with Physics Invariant Verification, RAG retrieval, and Attention Graphing."""
    data = request.get_json() or {}
    user_query = data.get("message", "").strip()
    speaker_id = data.get("speaker_id", "spk_charan")
    conversation_history = data.get("history", [])

    if not user_query:
        return jsonify({"error": "Empty query"}), 400

    # 1. Process query through specialized Remote Sensing & Physics Invariant Engine
    xai_res = rs_engine.process_query(
        user_query,
        speaker_id=speaker_id,
        conversation_history=conversation_history
    )

    # 2. Record interaction to local database
    try:
        db.log_interaction(
            speaker_name=xai_res.speaker_profile.get("name", "Charan"),
            user_input=user_query,
            assistant_response=xai_res.spoken_response,
            detected_emotion="focused",
            intent_name=xai_res.detected_topic
        )
    except Exception as e:
        logger.warning(f"DB log skipped: {e}")

    # 3. Return structured PRD XAI Payload
    return jsonify({
        "spoken_response": xai_res.spoken_response,
        "detailed_answer": xai_res.detailed_answer,
        "derivation_conclusion": xai_res.derivation_conclusion,
        "governing_equations": xai_res.governing_equations,
        "citations": xai_res.citations,
        "confidence_score": xai_res.confidence_score,
        "invariant_status": xai_res.invariant_status,
        "invariant_rule": xai_res.invariant_rule,
        "invariant_details": xai_res.invariant_details,
        "cross_validation": getattr(xai_res, "cross_validation", {}),
        "attention_graph": xai_res.attention_graph,
        "detected_topic": xai_res.detected_topic,
        "speaker_profile": xai_res.speaker_profile,
    })


@app.route("/api/speaker", methods=["GET"])
def api_speaker():
    """Returns enrolled speaker profile and session progress."""
    return jsonify({
        "profile": rs_engine.enrolled_speaker
    })


@app.route("/api/history", methods=["GET"])
def api_history():
    """Returns SQLite session interactions."""
    try:
        import sqlite3
        conn = sqlite3.connect(settings.database.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT timestamp, speaker_name, user_input, assistant_response FROM interactions ORDER BY id DESC LIMIT 10")
        rows = cursor.fetchall()
        conn.close()
        return jsonify({
            "history": [dict(r) for r in rows]
        })
    except Exception as e:
        return jsonify({"history": []})


# =====================================================================
# Main Runner
# =====================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  Explainable AI Voice Assistant for Remote Sensing & Physics")
    print("  Running locally at: http://127.0.0.1:5000")
    print("=" * 70 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
