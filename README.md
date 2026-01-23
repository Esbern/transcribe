
# 🔐 QualiVault


**QualiVault** is a local-first, GDPR-compliant tool for transcribing and analyzing qualitative interviews. It processes audio on your local machine (supporting Mac Silicon and NVIDIA GPUs) and outputs an Obsidian-ready Markdown vault.

to create the enviorment run 
## 🚀 Installation

micromamba create -f .\environment.yml
due to cpmpatability problmens with wisperx wisperx and pytorch must both be installed using pip
### 1. Prerequisites
Ensure you have `conda` or `micromamba` installed.

### 2. Clone the Repository
```bash
git clone <your-repo-url>
cd qualivault
```

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
### 3. Create the Environment
**For Mac (Apple Silicon):**
```bash
micromamba create -f environment_mac.yml
micromamba activate qualivault
```

pip install whisperx
**For Windows/Linux (NVIDIA CUDA):**
```bash
micromamba create -f environment_cuda.yml
micromamba activate qualivault
```

pip install -U label-studio
### 4. Install the Package
```bash
pip install -e .
```

to run label studio run the following from command line
label-studio
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