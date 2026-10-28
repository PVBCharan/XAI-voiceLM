# Product Requirement Document (PRD): Rapid Prototype
## Explainable AI Interactive Voice Assistant for Remote Sensing & Physics

---

## 1. Executive Summary & Clarification of Plan

The objective is to build a lean, functional **Proof-of-Concept (PoC) Prototype** demonstrating a two-way conversational voice assistant specialized in **Remote Sensing and its governing physics** (electromagnetic radiation, radiative transfer, radar/SAR, optical band maths, atmospheric windows).

### Critical Real-World Engineering Adjustments:
Before coding, we must address two foundational points from your specifications:

1. **Clarification on "Adjusted Weights" vs. "Inference Attention" (Point 4):**
   * Pre-trained LLMs do **not** adjust their stored neural network weights during live inference (that only happens during offline training or backpropagation fine-tuning). 
   * What the model *does* adjust dynamically during generation is **Self-Attention Activations / Token Saliency / Routing Weights**. 
   * **Prototype Solution:** We will provide an interactive visualization representing **Layer-Wise Attention Weights / Token Influence Graphs** (e.g., using `BertViz` or a D3.js force-directed neural graph) to show the user which scientific terms and knowledge tokens the neural link prioritized to formulate the answer.
2. **"What if the Resource Content is False?" (Point 5):**
   * If a RAG document contains false information, standard LLMs will blindly reproduce it (*Garbage In, Garbage Out*). 
   * **Prototype Solution:** We implement an **Automated Cross-Validation & Fact-Consistency Verification Loop** using a dual-check mechanism:
     1. Retrieve passages from independent sources (e.g., Primary Source vs. Consensus Physics Rules).
     2. An auxiliary Critic/Verifier Prompt inspects the candidate answer against universal physics invariants (e.g., Conservation of Energy, Planck's Law, speed of light $c = \lambda \nu$, Rayleigh $\lambda^{-4}$ scattering).
     3. If a conflict or low consensus is found, the assistant outputs a **Confidence Penalty** and actively states the discrepancy rather than hallucinating compliance.

---

## 2. Core Functional Requirements for Prototype

| ID | Feature | Prototype Implementation Strategy |
| :--- | :--- | :--- |
| **FR-1** | **Two-Way Voice Interaction** | Browser mic capture $\to$ WebSocket $\to$ VAD (Voice Activity Detection) $\to$ Whisper ASR $\to$ LLM $\to$ Fast streaming TTS. |
| **FR-2** | **Voice Recognition & Session Memory** | Extract speaker embeddings via **SpeechBrain (ECAPA-TDNN)**; cosine similarity matches returning speaker ID; SQLite/JSON stores and resumes unfinished topics. |
| **FR-3** | **Explainability & Confidence Scoring** | Structured JSON output containing: (a) Spoken answer, (b) Governing physical equations and assumptions, (c) Source document citations, (d) Grounding confidence score ($0.0 - 1.0$). |
| **FR-4** | **Neural Path & Weight Visualization** | A visual web panel rendering an animated neural network node graph depicting active token connections and attention weight distribution. |
| **FR-5** | **Adversarial / Corrupted Source Detection** | Physics Invariant Validator: Evaluates candidate answers against fundamental remote sensing laws to detect erroneous/corrupted source material. |

---

## 3. Existing Open-Source Repositories to Leverage

You do **not** need to write this from scratch. You can assemble this prototype using battle-tested open-source modules:

1. **Voice RAG Pipeline Base:**
   * **`woodstocksoftware/voice-rag-assistant`** or **`samay-jain/Voice_Assistant_RAG_System_using_LangChain_and_Streamlit`**: Excellent reference pipelines combining local Whisper, ChromaDB, and TTS.
2. **Speaker Identification & Voiceprints:**
   * **`iamunik/Voice-Identification-Web-app`** (or directly **`speechbrain/spkrec-ecapa-voxceleb`** on Hugging Face): Takes 3 seconds of reference audio, produces a 192-dimensional vector embedding, and performs cosine similarity identification ($>0.75$ match = recognized speaker).
3. **Neural Link & Attention Visualizer:**
   * **`jessevig/bertviz`** or frontend **`d3-force` / `vis.js`**: Renders dynamic attention weights and token-to-token connections.
4. **VAD & Audio Chunking:**
   * **`snakers4/silero-vad`**: Ultra-fast, lightweight voice activity detector to know exactly when the user starts and stops talking.

---

## 4. System Architecture

```text
       ┌────────────────────────────────────────────────────────┐
       │             Frontend (HTML5 / AudioWorklet)            │
       │   - Mic Streaming                - TTS Audio Player    │
       │   - Session Resume Notification  - Neural Graph (D3)   │
       └──────────────────────────┬─────────────────────────────┘
                                  │ Bidirectional WebSocket
                                  ▼
       ┌────────────────────────────────────────────────────────┐
       │                   FastAPI Backend                      │
       ├────────────────────────────────────────────────────────┤
       │ 1. Audio Ingestion & Silero VAD                        │
       │ 2. Speaker ID Engine (SpeechBrain ECAPA-TDNN)          │
       │    └─► Check local SQLite: Match Voice -> Load History │
       │ 3. Speech-to-Text (faster-whisper)                     │
       │ 4. Remote Sensing Knowledge Retrieval (ChromaDB RAG)  │
       │ 5. Physics Invariant & Cross-Verification Layer        │
       │ 6. Response Generator (Gemini Flash / Local Llama-3)   │
       │ 7. Attention Saliency Extractor                        │
       │ 8. Text-to-Speech (Piper TTS or Edge-TTS)              │
       └────────────────────────────────────────────────────────┘
```

---

## 5. Prototype Data Flow & Implementation Details

### Step 1: Voice Enrollment & Matching (Feature 2)
1. On initial connection, the user speaks for 3–5 seconds.
2. `speechbrain.pretrained.EncoderClassifier` computes embedding vector $E_{user}$.
3. Compare $E_{user}$ against enrolled vectors in `profiles.json` using Cosine Similarity:
   $$\text{Sim}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}$$
4. If $\text{Sim} \ge 0.75$, load existing profile: `{"user_id": "Speaker_A", "last_topic": "SAR Synthetic Aperture Radar Phase Shift", "progress": "In-progress"}`.
5. The assistant immediately initiates: *"Welcome back! Last time we were discussing SAR phase shifts. Would you like to continue or explore something new?"*

### Step 2: Knowledge Ingestion & Dual RAG (Features 1 & 5)
* Curate 3–5 sample PDFs in `data/`:
  * *Basics of Remote Sensing & EM Radiation* (Planck's Law, Wien's Displacement Law, Stefan-Boltzmann Law).
  * *Microwave Remote Sensing* (Active vs. Passive, Radar backscatter, Bragg scattering).
  * *Optical Vegetation Indices* (NDVI formula: $\frac{\text{NIR} - \text{RED}}{\text{NIR} + \text{RED}}$).
* Store chunks inside **ChromaDB** with metadata (`chunk_id`, `topic`, `source`).

### Step 3: Self-Correction & Cross-Validation Engine (Feature 5)
When a query arrives (e.g., *"Does radar require sunlight to capture images?"*):
1. **Passage Retrieval:** Retrieve Top-3 relevant chunks from ChromaDB.
2. **Invariant Cross-Checking:** The engine executes a verification prompt:
   ```markdown
   Context: {retrieved_chunks}
   Query: {user_query}
   Validate: Does this context violate physical laws (e.g., active vs. passive emission, EM conservation)?
   Output format: JSON { "is_valid": bool, "confidence": float, "flags": [] }
   ```
3. If an adversary injected a false document claiming *"Radar requires sunlight because it is an optical sensor"*, the verifier detects the contradiction with active sensing principles, rejects the corrupted chunk, tags the answer as **"Contradiction Detected in Source"**, and reports:
   > *"Warning: Source document contradicts fundamental physics. Active microwave sensors supply their own radiation source and do not depend on reflected solar illumination."*

### Step 4: Explainability & Attention Visualization (Features 3 & 4)
The API returns a multi-part payload:
1. **`spoken_response`**: Short, conversational, human-like voice response sent to TTS.
2. **`xai_grounding`**:
   * *Governing Equation:* e.g., $\lambda = \frac{c}{f}$ or $\text{NDVI} = \frac{\rho_{nir} - \rho_{red}}{\rho_{nir} + \rho_{red}}$
   * *Retrieval Confidence:* Calculated cosine distance from ChromaDB ($0.88$).
   * *Citations:* Chapter, Page, and Paragraph excerpt.
3. **`neural_graph_data`**: A structured node/edge array representing the token attention activations:
   ```json
   {
     "nodes": [{"id": "Active Microwave", "group": 1}, {"id": "Radar Antenna", "group": 2}, {"id": "Wavelength", "group": 2}],
     "links": [{"source": "Active Microwave", "target": "Radar Antenna", "weight": 0.92}]
   }
   ```
   Rendered in the UI as an interactive neural constellation that highlights the path traversed to answer the question.

---

## 6. Minimal Tech Stack to Run in VS Code

| Layer | Recommended Library / Tool | Why for Prototype |
| :--- | :--- | :--- |
| **IDE / Runner** | VS Code + Terminal | Native debugging, virtual environments. |
| **Backend API** | `FastAPI` + `Uvicorn` | Asynchronous WebSocket streaming. |
| **Audio Capture** | HTML5 `navigator.mediaDevices` | Zero-install web client. |
| **VAD** | `silero-vad` | Fast speech segmentation in Python. |
| **ASR** | `faster-whisper` (base or small) | Fast local inference without external API latency. |
| **Speaker ID** | `speechbrain` | Pre-trained ECAPA-TDNN in 5 lines of code. |
| **RAG Store** | `chromadb` | Zero-configuration local disk vector store. |
| **LLM Reasoning** | Gemini 2.5 Flash (via API) or Ollama (Llama-3.1-8B) | High speed, reliable JSON schema compliance. |
| **TTS** | `edge-tts` (free, fast) or `piper-tts` | Near-zero latency human-like voice. |
| **UI** | Single HTML/JS Dashboard | Renders mic button, audio playback, citations, and D3 neural graph. |

---

## 7. Step-by-Step Prototype Implementation Roadmap (Quick Demo)

```text
Day 1: Setup & Data Ingestion
 └── Create Python virtual env in VS Code.
 └── Ingest 2-3 Remote Sensing PDF chapters into local ChromaDB.
 └── Test retrieval queries on EM radiation physics.

Day 2: Speaker ID & Session Memory
 └── Integrate SpeechBrain ECAPA-TDNN.
 └── Store speaker voiceprints in profiles.json.
 └── Verify user re-identification and session topic resumption.

Day 3: Explainable RAG & Physics Validation
 └── Create the dual-prompt verification loop (Answer + Physics Critic).
 └── Test adversarial/false source handling.
 └── Format structured XAI JSON output (equations, citations, confidence).

Day 4: Audio Streaming & Voice Loop
 └── Build FastAPI WebSocket endpoint.
 └── Connect faster-whisper ASR and edge-tts streaming.
 └── Add barge-in / speech interruption handling via Silero VAD.

Day 5: Dashboard & Neural Link Visualization
 └── Build simple web frontend with microphone button and audio output.
 └── Add D3.js force-directed graph to render the attention/neural link visualizer.
 └── Run end-to-end dry-run demo for stakeholders.
```