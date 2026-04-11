<div align="center">

# 🎙️ Local Voice-Controlled AI Agent
A fully offline, privacy-first voice agent that listens to spoken commands, understands your intent, and takes action — all running locally on your machine with no cloud APIs.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-green)
![Faster-Whisper](https://img.shields.io/badge/Transcription-Faster--Whisper-FFD700)
![Phi-3](https://img.shields.io/badge/LLM-Phi--3-orange)
![Ollama](https://img.shields.io/badge/Runtime-Ollama-lightgrey)
![LangChain](https://img.shields.io/badge/Framework-LangChain-blueviolet)

</div>

---

## Overview

The **Local Voice-Controlled AI Agent** accepts audio input (microphone or file upload), transcribes it using a local Whisper model, classifies the intent using a locally-run Phi-3 LLM, and routes the command to the appropriate tool — creating files, generating code, summarising text, or answering questions.

Every component runs on your own hardware. No data is sent to any external service.

---

## Features

- **Fully local and offline** — no OpenAI, no cloud APIs, no telemetry, everything on your own hardware.
- **Two audio input modes** — live microphone recording or `.wav`/`.mp3` file upload.
- **Automatic intent classification** — routes commands into `create_file`, `write_code`, `summarize_text`, or `general_chat`.
- **Sandboxed file operations** — all generated files are strictly locked to the `output/` directory; path-traversal attempts are hard-rejected.
- **Context-aware code generation** — reads the tail of an existing file before appending new code, ensuring continuity.
- **LLM output sanitisation** — guards against chatty model responses that could corrupt filenames or file paths.
- **Interactive Streamlit UI** — clean, browser-based interface with a real-time processing pipeline display.

---

## Architecture

![System Flow](https://github.com/atharva-ankad/voice_ai_agent/blob/main/assets/system_flow.png)

The system is composed of three sequential stages orchestrated by `app.py`:

**Stage 1 — Transcription (`modules/audio.py`)**
Faster-Whisper loads the `base` Whisper model and converts the raw audio bytes into a text transcript. The model runs entirely on CPU using `int8` quantisation for broad hardware compatibility.

**Stage 2 — Intent Detection (`modules/intent.py`)**
The transcript is passed to a LangChain chain backed by a local `phi3` model running via Ollama. A strict classification prompt with `temperature=0` forces the model to output exactly one of the four supported intent labels. A regex fallback catches any cases where the model adds extra words despite the instruction.

**Stage 3 — Tool Execution (`modules/tools.py`)**
A master router (`execute_action`) dispatches to the correct tool function based on the detected intent:

| Intent | Tool | What it does |
|---|---|---|
| `create_file` | `create_file_tool` | Extracts a filename from the transcript and creates an empty file in `output/` |
| `write_code` | `write_code_tool` | Generates code via Phi-3 and writes or appends it to a file in `output/` |
| `summarize_text` | `summarize_text_tool` | Returns a concise summary of the spoken text |
| `general_chat` | `general_chat_tool` | Passes the transcript to Phi-3 for a free-form response |

---

## Repository Structure

```
voice_ai_agent/
│
├── app.py                 # Streamlit frontend and pipeline entry point
├── requirements.txt       # Python package dependencies
├── .gitignore
│
├── modules/
│   ├── __init__.py
│   ├── audio.py           # Faster-Whisper transcription
│   ├── intent.py          # LangChain + Phi-3 intent classification
│   └── tools.py           # Sandboxed tool execution engine
│
└── output/                # Auto-created at runtime; stores all agent outputs
```

---

## Requirements

### Software

- **Python** 3.10 or higher
- **Ollama** — required to run Phi-3 locally (see installation below)
- **A modern web browser** — for the Streamlit interface

### Python Packages

```
streamlit>=1.36.0
faster-whisper
langchain
langchain-community
```

### Hardware

| Component | Minimum |
|---|---|
| CPU | Any modern multi-core CPU |
| RAM | 8 GB recommended (Phi-3 + Whisper both load into RAM) |
| GPU | Not required — the system runs fully on CPU by default |
| Disk | ~2–3 GB free for model weights downloaded on first run |

---

## Installation

### Step 1 — Install Ollama

Ollama is the local model runtime. Download and install it from the official site:

**[https://ollama.com/download](https://ollama.com/download)**

After installation, pull the Phi-3 model (this is a one-time download of ~2 GB):

```bash
ollama pull phi3
```

Verify Ollama is running before launching the agent:

```bash
ollama list
```

You should see `phi3` in the list.

---

### Step 2 — Clone the Repository

```bash
git clone https://github.com/atharva-ankad/voice_ai_agent
cd voice_ai_agent
```

---

### Step 3 — Create a Virtual Environment (Recommended)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
```

---

### Step 4 — Install Python Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `faster-whisper` will automatically download the Whisper `base` model weights (~150 MB) on first run.

---

## Usage

Make sure Ollama is running in the background (it starts automatically on Windows after installation), then launch the Streamlit app:

```bash
streamlit run app.py
```

Your browser will open automatically at `http://localhost:8501`.

### Using the Interface

1. Choose an input method — **Microphone** or **Upload Audio File**
2. Record your command or upload a `.wav` / `.mp3` file
3. Click **Process Command**
4. The UI displays four results side by side:
   - Transcribed text
   - Detected intent
   - Action taken
   - Final output

---

## Example Commands

The following voice commands have been tested and produce the expected behaviour:

| What you say | Detected intent | Result |
|---|---|---|
| *"Create a new file called notes.txt"* | `create_file` | Empty `output/notes.txt` created |
| *"Create a file called data.csv"* | `create_file` | Empty `output/data.csv` created |
| *"Write a Python function to sort a list"* | `write_code` | Code written to `output/code_snippet.py` |
| *"Write a Python script called scraper.py that fetches a webpage"* | `write_code` | Code written to `output/scraper.py` |
| *"Summarise this: machine learning is a branch of AI that learns from data"* | `summarize_text` | Concise summary returned |
| *"What is the difference between a list and a tuple in Python?"* | `general_chat` | Free-form answer from Phi-3 |

---

## Hardware Notes and Workarounds

### CPU-only Execution

Both Faster-Whisper and Phi-3 (via Ollama) run on CPU by default in this project. This was chosen for maximum compatibility across machines without requiring a configured GPU or CUDA installation.

**Expected performance on CPU:**
- Whisper transcription: 5–15 seconds depending on audio length
- Phi-3 intent detection + tool response: 10–30 seconds

### Optional GPU Acceleration

If you have a CUDA-compatible GPU and your CUDA drivers are properly installed, you can enable GPU acceleration for the Whisper transcription step by editing `modules/audio.py`:

```python
# Default (CPU)
model = WhisperModel("base", device="cpu", compute_type="int8")

# GPU (requires CUDA)
model = WhisperModel("base", device="cuda", compute_type="float16")
```

Ollama will automatically use your GPU for Phi-3 if CUDA is detected.

### Context Window Cap

Phi-3 is a small model with a limited context window. When appending code to an existing file, only the last 800 characters of the file are passed as context to keep the prompt within the model's limits.

---

## The `output/` Directory

All files created or generated by the agent are saved to the `output/` folder at the project root. This directory is automatically created if it does not exist.

**Important:** Generated files accumulate in `output/` across sessions and are not automatically deleted. Periodically clear this folder manually to avoid clutter:

If your `.gitignore` does not already exclude `output/` contents, add the following to avoid committing generated files:

```
output/*
!output/.gitkeep
```

---

## Known Limitations

- **Four intents only** — commands outside the four supported categories fall back to `general_chat`.
- **English only** — Faster-Whisper's `base` model performs best with English audio; accuracy on other languages will vary.
- **No conversation memory** — each command is processed independently; the agent has no context of previous interactions.
- **Phi-3 response quality** — as a small local model, Phi-3 may produce inconsistent or lower-quality outputs compared to larger cloud-hosted models, particularly for complex code generation requests.
- **Windows tested** — the project has been developed and tested on Windows; behaviour on macOS or Linux may differ, particularly around microphone access in the browser.

---

## Future Improvements

- Add GPU acceleration toggle directly in the Streamlit UI
- Expand the intent set (e.g., `search_web`, `run_terminal_command`, `open_application`)
- Add conversation memory so the agent can handle multi-turn interactions
- Support additional Whisper model sizes (`small`, `medium`) selectable from the UI
- Add a file browser within the UI to view and download files from `output/`
- Implement audio pre-processing (noise reduction) to improve transcription accuracy on low-quality input
