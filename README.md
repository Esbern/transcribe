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
**For Mac (Apple Silicon):**
```bash
micromamba create -f environment_mac.yml
micromamba activate qualivault
```

**For Windows/Linux (NVIDIA CUDA):**
```bash
micromamba create -f environment_cuda.yml
micromamba activate qualivault
```

### 4. Install the Package
```bash
pip install -e .
```

## 🛠️ Project Workflow

QualiVault uses a **Project-based** structure to ensure data isolation. Your interview data should live in a folder **outside** the main software directory.

**Data Privacy & Git:**
The system is designed so you can use Git for version control of your *transcripts* and *analysis*, while keeping heavy and sensitive *audio files* out of the cloud.

### 1. Create a Project
Create a folder anywhere on your computer (e.g., in your Documents folder).

```bash
mkdir my_interview_study
cd my_interview_study
```

### 2. Initialize the Workspace
Run the initialization command. This sets up the configuration, notebooks, and a **separate git repository** for your data.

```bash
python -m qualivault.cli --init
```
*This automatically creates a `.gitignore` that excludes audio files (`.wav`, `.mp3`, `audio_input/`) to ensure they are never committed to git.*

### 3. Configure
1.  Open `.env` and paste your Hugging Face token.
2.  Open `scanner.yml` and check the paths. By default, it expects audio in `./audio_input`.

### 4. Run
Launch Jupyter Lab to start processing:
```bash
jupyter lab