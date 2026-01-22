import os
import logging
import torch
import gc
import yaml
import re
import subprocess
from pathlib import Path
import warnings
from huggingface_hub import login
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pyannote.audio import Pipeline
import librosa
import soundfile as sf
import pandas as pd
import numpy as np

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
    # 0. Memory Safety for MPS (Prevents premature OOM on Mac)
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

    # 1. Try to load config.yml (check current and parent dir)
    hf_token = None
    config_path = "config.yml" if os.path.exists("config.yml") else "../config.yml"
    
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                hf_token = config.get("hf_token")
                
                # Suppress warnings if configured
                if config.get("suppress_warnings", True):
                    warnings.filterwarnings("ignore")
                
                # Handle Models Cache
                models_dir = config.get("paths", {}).get("models_folder")
                if models_dir:
                    # Resolve relative to config file location
                    config_dir = Path(config_path).parent
                    models_path = (config_dir / models_dir).resolve()
                    models_path.mkdir(parents=True, exist_ok=True)
                    os.environ["HF_HOME"] = str(models_path)
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

def process_recipe(recipe_path, flac_dir, output_dir, config=None, hf_token=None, cache_dir=None):
    """
    Main loop to process the recipe file.
    """
    recipe_path = Path(recipe_path)
    flac_dir = Path(flac_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not recipe_path.exists():
        logger.error(f"❌ Recipe file not found: {recipe_path}")
        return

    with open(recipe_path, 'r') as f:
        recipe = yaml.safe_load(f)

    # Initialize Models
    device = get_device()
    transcriber = Transcriber(device=device, cache_dir=cache_dir)
    
    # Prepare Transcription Settings
    language = "da"
    prompt_ids = None
    if config and "transcription" in config:
        language = config["transcription"].get("language", "da")
        topic_prompt = config["transcription"].get("topic_prompt")
        if topic_prompt:
            prompt_ids = transcriber.processor.get_prompt_ids(topic_prompt)
            # Fix: Ensure prompt_ids is a Tensor (not numpy) and on the correct device
            if isinstance(prompt_ids, (np.ndarray, list)):
                prompt_ids = torch.tensor(prompt_ids)
            if isinstance(prompt_ids, torch.Tensor):
                prompt_ids = prompt_ids.to(device)

    diarization_pipeline = None
    if hf_token:
        try:
            logger.info("⏳ Loading Diarization Pipeline...")
            diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token,
                cache_dir=cache_dir
            )
            diarization_pipeline.to(torch.device(device))
            logger.info("✅ Diarization Pipeline Loaded.")
        except Exception as e:
            logger.error(f"❌ Failed to load Pyannote: {e}")
            return

    updated = False
    for item in recipe:
        if item.get('status') == 'pending':
            try:
                csv_path = process_item(item, flac_dir, output_dir, transcriber, diarization_pipeline, language, prompt_ids)
                if csv_path:
                    item['status'] = 'completed'
                    item['transcript_path'] = str(csv_path)
                    updated = True
            except Exception as e:
                logger.error(f"❌ Failed processing {item['output_name']}: {e}")
                item['status'] = 'failed'
                item['error_msg'] = str(e)
                updated = True
        else:
            logger.info(f"⏭️  Skipping {item['output_name']} (Status: {item.get('status')})")

    if updated:
        with open(recipe_path, 'w') as f:
            yaml.dump(recipe, f, sort_keys=False)
        logger.info("✅ Recipe updated.")

def process_item(item, flac_dir, output_dir, transcriber, diarization_pipeline, language="da", prompt_ids=None):
    filename = item['output_name']
    audio_path = flac_dir / filename
    
    if not audio_path.exists():
        logger.error(f"❌ File not found: {audio_path}")
        return None

    logger.info(f"🚀 Processing: {filename}")
    
    # Load Audio
    y, sr = librosa.load(audio_path, sr=16000, mono=False)
    
    # Prepare Mono for Whisper/Diarization
    if y.ndim > 1:
        y_mono = librosa.to_mono(y)
    else:
        y_mono = y
        
    temp_wav = output_dir / "temp_process.wav"
    sf.write(temp_wav, y_mono, sr)
    
    # Free memory before heavy lifting
    del y, y_mono
    gc.collect()

    # 1. Diarization
    logger.info("   🧠 Running Diarization...")
    diarization = diarization_pipeline(str(temp_wav), min_speakers=2, max_speakers=2)

    # 2. Transcription
    logger.info("   ✍️  Transcribing...")
    transcript = transcriber.transcribe(temp_wav, language=language, prompt_ids=prompt_ids)

    # 3. Merge & Identify
    logger.info("   🔗 Merging text into paragraphs...")
    final_segments = []
    current_speaker = None
    current_text_buffer = []
    current_start = 0.0
    # Check separation stats from recipe
    sep_db = float(item.get('separation_db', 0))
    is_hybrid = sep_db > 5.0 # Threshold

    for chunk in transcript['chunks']:
        start, end = chunk['timestamp']
        text = chunk['text']
        
        # Find best speaker match from diarization
        best_speaker = "UNKNOWN"
        max_overlap = 0
        
        for turn, _, speaker in diarization.itertracks(yield_label=True): 
            overlap = max(0, min(end, turn.end) - max(start, turn.start))
            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = speaker
        
        # Hybrid Logic: If separation is good, check which channel is louder during this segment
        # Note: We loaded mono for processing, so we can't check stereo energy here easily without reloading.
        # For now, we rely on the diarization speaker ID mapping if we implemented it, 
        # or we assume the Diarization step handled it. 
        # (To keep memory low, we skip reloading stereo here as requested by the memory optimization goal).
        
        # Merge Logic
        if best_speaker == current_speaker:
            current_text_buffer.append(text)
        else:
            if current_speaker is not None:
                final_segments.append({
                    "Start": round(current_start, 2),
                    "End": round(start, 2),
                    "Speaker": current_speaker,
                    "Text": " ".join(current_text_buffer)
                })
            current_speaker = best_speaker
            current_text_buffer = [text]
            current_start = start

    # Append final segment
    if current_speaker is not None:
        final_segments.append({
            "Start": round(current_start, 2),
            "End": round(end, 2),
            "Speaker": current_speaker,
            "Text": " ".join(current_text_buffer)
        })

    df = pd.DataFrame(final_segments)
    csv_name = output_dir / filename.replace('.flac', '.csv')
    df.to_csv(csv_name, index=False)
    logger.info(f"   🎉 Saved: {csv_name}")
    
    # 🧹 DEEP CLEAN MEMORY
    del diarization
    del transcript
    if temp_wav.exists(): temp_wav.unlink()
    
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()
    
    return csv_name

class Transcriber:
    def __init__(self, device=None, model_id="openai/whisper-large-v3", cache_dir=None):
        """
        Initializes the Whisper model.
        """
        self.device = device if device else get_device()
        
        # Mac MPS does not support float16 for all ops yet, so we use float32 for safety on Mac
        # CUDA uses float16 for speed. MPS can use float16 for memory saving if supported.
        self.torch_dtype = torch.float16 if self.device in ["cuda", "mps"] else torch.float32
        
        logger.info(f"⏳ Loading Model: {model_id} on {self.device.upper()}...")
        
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, 
            torch_dtype=self.torch_dtype, 
            low_cpu_mem_usage=True, 
            use_safetensors=True,
            cache_dir=cache_dir
        )
        self.model.to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            stride_length_s=(5, 5),
            batch_size=16,
            return_timestamps=True,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
        logger.info("✅ Model Loaded.")

    def transcribe(self, audio_path, language="da", prompt_ids=None):
        logger.info(f"🎙️ Transcribing: {audio_path}")
        generate_kwargs = {"language": language, "task": "transcribe"}
        if prompt_ids is not None:
            generate_kwargs["prompt_ids"] = prompt_ids
            
        result = self.pipe(str(audio_path), generate_kwargs=generate_kwargs)
        return result

def to_markdown(csv_path, output_dir):
    """Converts a CSV transcript to Obsidian-ready Markdown."""
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not csv_path.exists():
        logger.error(f"❌ CSV not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path).fillna('')
    
    # Create Markdown Content
    md_lines = []
    md_lines.append(f"# Transcript: {csv_path.stem}\n")
    
    for _, row in df.iterrows():
        start = row.get('Start', 0)
        speaker = row.get('Speaker', 'Unknown')
        text = row.get('Text', '')
        
        # Format timestamp (MM:SS)
        time_str = f"{int(start//60):02d}:{int(start%60):02d}"
        
        md_lines.append(f"**[{time_str}] {speaker}**:")
        md_lines.append(f"{text}\n")
        
    # Add Metadata (Inline Fields for Dataview)
    md_lines.append("\n---\n")
    md_lines.append("## Metadata")
    md_lines.append(f"**Source**:: [[{csv_path.name}]]")
    md_lines.append(f"**Date**:: {pd.Timestamp.now().strftime('%Y-%m-%d')}")
    md_lines.append("**Status**:: #transcript/verified")
    
    md_content = "\n".join(md_lines)
    
    md_filename = output_dir / f"{csv_path.stem}.md"
    with open(md_filename, "w", encoding="utf-8") as f:
        f.write(md_content)
        
    logger.info(f"   📄 Created Markdown: {md_filename.name}")
    return md_filename

def export_recipe(recipe_path, output_dir):
    """Exports all completed transcripts in the recipe to Markdown."""
    recipe_path = Path(recipe_path)
    output_dir = Path(output_dir)
    obsidian_dir = output_dir / "Obsidian_Vault"
    
    if not recipe_path.exists():
        logger.error(f"❌ Recipe not found: {recipe_path}")
        return

    with open(recipe_path, 'r') as f:
        recipe = yaml.safe_load(f)
        
    logger.info(f"📂 Exporting to: {obsidian_dir}")
    
    count = 0
    for item in recipe:
        if item.get('status') == 'completed' and 'transcript_path' in item:
            csv_path = Path(item['transcript_path'])
            if csv_path.exists():
                to_markdown(csv_path, obsidian_dir)
                count += 1
    
    logger.info(f"✅ Exported {count} transcripts to Markdown.")