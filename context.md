# 🤖 AI Context: The QualiVault Project

**Project Name:** QualiVault (formerly Sovereign Pipeline)
**Goal:** A GDPR-compliant, local-first Python tool for analyzing qualitative interviews. It processes audio on the researcher's laptop (Mac Silicon or CUDA) and outputs an Obsidian Vault.

**Architecture:**
1.  **Package Structure:**
    * `src/qualivault/`: Main Python package.
    * `core.py`: Handling Whisper (Hugging Face), Pyannote (Diarization), and LoRA loading.
    * `cli.py`: The `qualivault --init` command to setup user workspaces.
    * `pipeline.py`: Resumable convert/analyze/transcribe orchestration (state in `.qualivault/run_state.json`).
    * `templates/`: Notebooks 00–05 and `config.yml`, copied into each project by `--init`.

2.  **Key Technical Decisions:**
    * **Backends:** `transformers` (default; supports LoRA adapters via `peft`, works on Mac Silicon), plus optional `whisperx` (CUDA) and `faster_whisper` (CPU). `backend: auto` picks WhisperX on CUDA when installed.
    * **VAD:** We use `silero-vad` or `no_speech_threshold` to prevent hallucination loops.
    * **Diarization:** We rely on "X/Y Stereo Split" (Left=Student, Right=Farmer) first, then Pyannote for fine-tuning.
    * **Obsidian Output:** We append structured metadata (Inline Fields `Key:: Value`) to the bottom of Markdown files for Dataview queries.

3.  **Environment Strategy:**
    * `environment_mac.yml`: Uses `pip` to install `torch` (MPS support).
    * `environment_cuda.yml`: Uses `pip` with `--index-url` for NVIDIA CUDA 12.8 (torch 2.8, required by pyannote.audio>=4).
    * Internal env name for all (incl. `environment_cpu.yml`): `qualivault`.
