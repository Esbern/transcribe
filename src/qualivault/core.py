import os
import logging
import torch
import yaml
import re
import subprocess
from pathlib import Path
from huggingface_hub import login
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Configure logging for clear user feedback
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("QualiVault")

def setup_environment():
    """
    Initializes the QualiVault environment.
    1. Loads config.yml to find HF_TOKEN
    2. Authenticates with Hugging Face (if token is present)
    3. Detects and returns the best available hardware device (MPS/CUDA/CPU)
    """
    # 1. Try to load config.yml (check current and parent dir)
    hf_token = None
    config_path = "config.yml" if os.path.exists("config.yml") else "../config.yml"
    
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                hf_token = config.get("hf_token")
        except Exception as e:
            logger.warning(f"⚠️ Could not read config.yml: {e}")

    # 2. Hugging Face Authentication
    if hf_token:
        try:
            login(token=hf_token)
            logger.info("✅ Authenticated with Hugging Face.")
        except Exception as e:
            logger.error(f"❌ Authentication failed: {e}")
    else:
        logger.info("ℹ️ No HF_TOKEN found in config.yml. Proceeding (models must be cached or public).")

    # 3. Device Detection
    device = get_device()
    logger.info(f"🚀 Hardware Acceleration: {device.upper()}")
    
    return device

def get_device():
    """Returns 'cuda', 'mps' (Mac Silicon), or 'cpu'."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    else:
        return "cpu"

def init_git_project(project_root="."):
    """Initializes git and creates .gitignore in the project root."""
    project_root = Path(project_root).resolve()
    git_dir = project_root / ".git"
    
    if git_dir.exists():
        logger.info("ℹ️  Git is already initialized.")
        return

    logger.info(f"🛡️  Initializing Git in: {project_root}")
    
    # Create .gitignore
    gitignore_content = """
.ipynb_checkpoints/
__pycache__/
*.wav
*.mp3
*.m4a
*.flac
audio_input/
"""
    with open(project_root / ".gitignore", "w") as f:
        f.write(gitignore_content.strip())
        
    try:
        subprocess.run(["git", "init"], cwd=project_root, check=True)
        logger.info("✅  Git initialized successfully.")
    except Exception as e:
        logger.error(f"❌  Git init failed: {e}")

def scan_audio_folder(base_path, folder_re, file_re):
    results = {}
    base = Path(base_path)
    
    if not base.exists():
        logger.error(f"❌ Error: Path does not exist: {base}")
        return {}

    for root, dirs, files in os.walk(base):
        # Check if current folder matches the ID pattern
        folder_match = re.search(folder_re, Path(root).name)
        if folder_match:
            try:
                i_id = folder_match.group('id')
            except IndexError:
                i_id = Path(root).name
            
            if i_id not in results: results[i_id] = []
            
            # Find matching files inside
            for file in files:
                if re.match(file_re, file, re.IGNORECASE):
                    results[i_id].append(os.path.join(root, file))
    return results

class Transcriber:
    def __init__(self, device=None, model_id="openai/whisper-large-v3"):
        """
        Initializes the Whisper model.
        """
        self.device = device if device else get_device()
        
        # Mac MPS does not support float16 for all ops yet, so we use float32 for safety on Mac
        # CUDA uses float16 for speed
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        logger.info(f"⏳ Loading Model: {model_id} on {self.device.upper()}...")
        
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, 
            torch_dtype=self.torch_dtype, 
            low_cpu_mem_usage=True, 
            use_safetensors=True
        )
        self.model.to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
        logger.info("✅ Model Loaded.")

    def transcribe(self, audio_path, language="da"):
        logger.info(f"🎙️ Transcribing: {audio_path}")
        # generate_kwargs allows us to force the language (e.g., Danish)
        result = self.pipe(str(audio_path), generate_kwargs={"language": language})
        return result