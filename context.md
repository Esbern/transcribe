# 🤖 AI Context: The QualiVault Project

**Project Name:** QualiVault (formerly Sovereign Pipeline)
**Goal:** A GDPR-compliant, local-first Python tool for analyzing qualitative interviews. It processes audio on the researcher's laptop (Mac Silicon or CUDA) and outputs an Obsidian Vault.

**Architecture:**
1.  **Package Structure:**
    * `src/qualivault/`: Main Python package.
    * `core.py`: Handling Whisper (Hugging Face), Pyannote (Diarization), and LoRA loading.
    * `cli.py`: The `qualivault --init` command to setup user workspaces.
    * `notebooks/`: Template Jupyter notebooks (Import, Analysis).

2.  **Key Technical Decisions:**
    * **No WhisperX:** We use standard `transformers` + `peft` to support custom LoRA adapters on Mac Silicon.
    * **VAD:** We use `silero-vad` or `no_speech_threshold` to prevent hallucination loops.
    * **Diarization:** We rely on "X/Y Stereo Split" (Left=Student, Right=Farmer) first, then Pyannote for fine-tuning.
    * **Obsidian Output:** We append structured metadata (Inline Fields `Key:: Value`) to the bottom of Markdown files for Dataview queries.

3.  **Environment Strategy:**
    * `environment_mac.yml`: Uses `pip` to install `torch` (MPS support).
    * `environment_cuda.yml`: Uses `pip` with `--index-url` for NVIDIA CUDA 12.1.
    * Internal env name for both: `qualivault`.

**Current Task:**
We are finalizing the `src/qualivault/` package structure and ensuring the `cli.py` correctly copies notebooks to the user's folder.