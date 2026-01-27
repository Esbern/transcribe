
# 🔐 QualiVault


**QualiVault** is a local-first, GDPR-compliant tool for transcribing and analyzing qualitative interviews. It processes audio on your local machine (supporting Mac Silicon and NVIDIA GPUs) and outputs an Obsidian-ready Markdown vault.

## 🚀 Installation

### 1. Prerequisites
Ensure you have `conda` or `micromamba` installed.

### 2. Clone the Repository
```bash
git clone <your-repo-url>
cd qualivault
```

### 3. Create the Environment
Pick the environment that matches your hardware and transcription backend preference.

**Option A: Mac (Apple Silicon, MPS) — Transformers Whisper (default):**
```bash
micromamba create -f environment_mac.yml
micromamba activate qualivault
```

**Option B: Windows/Linux (NVIDIA CUDA) — Transformers Whisper (default):**
```bash
micromamba create -f environment_cuda.yml
micromamba activate qualivault
```

**Option C: Windows/Linux (NVIDIA CUDA) — WhisperX (fastest):**
```bash
micromamba create -f environment_cuda.yml
micromamba activate qualivault
pip install whisperx
```

**Option D: CPU-only (any OS) — faster-whisper (lightweight):**
```bash
micromamba create -f environment_cpu.yml
micromamba activate qualivault
```

### 4. Install PyTorch and Dependencies
PyTorch is installed via the environment files. WhisperX is optional and installed with pip.

**Optional (any platform):**
```bash
pip install -U label-studio
```

**Optional (CUDA WhisperX):**
```bash
pip install whisperx
```

### 5. Install the Package
```bash
pip install -e .
```

### 6. Install Ollama (Optional - For Transcript Validation)
To validate transcripts and detect hallucinations using local LLMs:

**Install Ollama:**
- **Windows/Linux:** Download from [ollama.ai](https://ollama.ai)
- **Mac:** `brew install ollama`

**Pull a Model:**

Choose based on your needs:

| Model | Size | Language Support | Best For |
|-------|------|------------------|----------|
| `llama2` | 7B | English + multilingual | General validation, fast |
| `mistral` | 7B | English + multilingual | Better reasoning, medium speed |
| `llama3` | 8B | English + multilingual | Most accurate, slower |
| `jobautomation/OpenEuroLLM-Danish:latest` | 7B | **Danish-optimized** | **Danish interviews (recommended)** |

**How to Choose:**
- **For Danish interviews:** Use `jobautomation/OpenEuroLLM-Danish:latest` - specifically trained for Danish
- **For multilingual projects:** Use `llama3` or `mistral`
- **For speed:** Use `llama2` (faster but less accurate)
- **Check available models:** `ollama list`

**Installation:**
```bash
# For Danish interviews (recommended)
ollama pull jobautomation/OpenEuroLLM-Danish:latest

# OR for general use
ollama pull llama2
ollama pull mistral
```

**Start Ollama:**
```bash
ollama serve
```

Then configure the model name in notebook `04_Validate_Transcripts.ipynb`.

### 7. Run Label Studio
To launch Label Studio for annotation:
```bash
label-studio
```
## 🛠️ Project Workflow

QualiVault uses a **Project-based** structure to ensure data isolation. All projects are created inside the `projects/` directory.

**Data Privacy & Git:**
Each project gets its own Git repository. The system automatically configures `.gitignore` to exclude heavy and sensitive audio files, allowing you to version control your *transcripts* and *analysis* safely.

### 1. Initialize a New Project
Run the initialization command with your desired project name. This creates the folder structure, configuration, and notebooks.

```bash
python -m qualivault.cli --init my_interview_study
```

This creates:
*   `projects/my_interview_study/config.yml`
*   `projects/my_interview_study/notebooks/`
*   `projects/my_interview_study/.gitignore` (excludes audio files)

### 2. Configure
Open `projects/my_interview_study/config.yml`.

**Key Settings:**
*   **`hf_token`**: Your Hugging Face token (required for Pyannote diarization).
*   **`transcription`**:
    *   `backend`: `auto` (recommended), `whisperx` (CUDA), or `transformers`.
    *   CPU-only fallback: `faster_whisper`
    *   `model_id`: Whisper model identifier.
    *   `language`: Target language (e.g., "da").
    *   `topic_prompt`: Keywords to guide the AI (jargon, names).
    *   `batch_size`: Reduce to `1` if you run out of memory.
    *   `repetition_penalty`: Increase (e.g., `1.1`) if the model loops.
*   **`speaker_map`**: Rename "SPEAKER_00" to "Interviewer" automatically.
*   **`paths`**: Set `org_audio_folder` to where your raw audio lives.

### 3. Run Notebooks
Launch Jupyter Lab directly in your project folder:

```bash
python -m qualivault.cli --notebook my_interview_study