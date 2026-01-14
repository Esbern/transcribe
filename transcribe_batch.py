import os
import json
import yaml
import torch
import functools
import gc
from datetime import datetime
import whisperx
from whisperx.diarize import DiarizationPipeline
import numpy as np

# Memory tuning defaults
BATCH_SIZE = 2   # Lower to reduce GPU memory
CHUNK_SIZE = 30  # Seconds per transcription chunk

# --- SECURITY FIX ---
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"
_original_load = torch.load
@functools.wraps(_original_load)
def unsafe_load_wrapper(*args, **kwargs):
    if 'weights_only' in kwargs: 
        del kwargs['weights_only']
    return _original_load(*args, **kwargs, weights_only=False)
torch.load = unsafe_load_wrapper
# -----------------------


class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types in JSON serialization"""
    def default(self, obj):
        if isinstance(obj, np.integer): 
            return int(obj)
        if isinstance(obj, np.floating): 
            return float(obj)
        if isinstance(obj, np.ndarray): 
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def load_recipe(recipe_path):
    """Load the processing recipe YAML file"""
    with open(recipe_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_recipe(recipe_path, data):
    """Save the processing recipe back to YAML"""
    with open(recipe_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def transcribe_audio(audio_files, model, align_model, metadata, diarize_model, device):
    """Transcribe audio files and return result with speaker diarization"""
    # If multiple files, concatenate audio
    audio_list = []
    for audio_file in audio_files:
        try:
            audio = whisperx.load_audio(audio_file)
            audio_list.append(audio)
        except Exception as e:
            print(f"  ⚠ Warning: Failed to load {audio_file}: {e}")
            continue
    
    if not audio_list:
        raise ValueError("No audio files could be loaded")
    
    # Concatenate audio if multiple files
    if len(audio_list) > 1:
        audio = np.concatenate(audio_list)
    else:
        audio = audio_list[0]
    
    # Transcribe with smaller batch and chunking to reduce GPU memory
    result = model.transcribe(audio, batch_size=BATCH_SIZE, chunk_size=CHUNK_SIZE)
    
    # Align
    result = whisperx.align(result["segments"], align_model, metadata, audio, device)
    
    # Diarize (2 speakers)
    diarize_segments = diarize_model(audio, min_speakers=2, max_speakers=2)
    result = whisperx.assign_word_speakers(diarize_segments, result)
    
    return result


def main():
    # Configuration
    RECIPE_PATH = "processing_recipe.yaml"
    OUTPUT_DIR = "transcriptions"
    # Use the FLACs produced by process_audio.py (configured in scanner.yml)
    with open("scanner.yml", "r", encoding="utf-8") as f:
        scanner_cfg = yaml.safe_load(f)
    FLAC_BASE = scanner_cfg["paths"]["output_base_folder"]
    DEVICE = "cuda"
    COMPUTE_TYPE = "float16"  # If OOM persists, try "int8" or lower BATCH_SIZE further
    MODEL_NAME = "large-v3"
    MODEL_DIR = "e:/ai_models/whisper_large_v3"
    DIARIZE_TOKEN = "XXXX"  # Update with your HF token if needed
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Agriculture-specific prompt
    agri_prompt = "Landbrug, omdrift, gødning, efterafgrøder, biomasse, landmand, høst, hektar, bedrift."
    asr_options = {"initial_prompt": agri_prompt}
    
    # Load recipe
    print("Loading processing recipe...")
    recipe = load_recipe(RECIPE_PATH)
    
    # Load models (once)
    print(f"Loading {MODEL_NAME} model on {DEVICE}...")
    model = whisperx.load_model(
        MODEL_NAME,
        DEVICE,
        compute_type=COMPUTE_TYPE,
        download_root=MODEL_DIR,
        language="da",
        asr_options=asr_options
    )
    
    print("Loading alignment model...")
    align_model, metadata = whisperx.load_align_model(language_code="da", device=DEVICE)
    
    print("Loading diarization model...")
    diarize_model = DiarizationPipeline(use_auth_token=DIARIZE_TOKEN, device=DEVICE)
    
    # Process each entry
    processed_count = 0
    skipped_count = 0
    
    for entry in recipe:
        interview_id = entry['id']
        status = entry.get('status', 'pending')
        output_name = entry['output_name']
        flac_path = os.path.join(FLAC_BASE, output_name)
        files = [flac_path]
        
        # Skip if not marked as flac or already transcribed
        if status != 'flac':
            print(f"[{interview_id}] Status is '{status}', skipping...")
            skipped_count += 1
            continue

        if not os.path.exists(flac_path):
            print(f"[{interview_id}] FLAC not found at {flac_path}, skipping and marking error")
            entry['status'] = 'error'
            entry['error'] = f"Missing FLAC: {flac_path}"
            skipped_count += 1
            continue
        
        if not files:
            print(f"[{interview_id}] No files specified, skipping...")
            skipped_count += 1
            continue
        
        try:
            print(f"\n[{interview_id}] Processing: {output_name}")
            print(f"  Files: {files}")
            
            # Transcribe
            print("  Transcribing...")
            result = transcribe_audio(files, model, align_model, metadata, diarize_model, DEVICE)
            
            # Save JSON
            json_path = os.path.join(OUTPUT_DIR, f"{interview_id}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result["segments"], f, cls=NumpyEncoder, indent=2, ensure_ascii=False)
            
            print(f"  ✅ Saved JSON: {json_path}")
            
            # Update recipe
            entry['status'] = 'transcribed'
            entry['processed_at'] = datetime.now().isoformat()
            processed_count += 1
            
            # Save progress incrementally to be crash-resilient
            save_recipe(RECIPE_PATH, recipe)
            
            # Free GPU memory between items
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"  ❌ Error processing {interview_id}: {e}")
            entry['status'] = 'error'
            entry['error'] = str(e)
            
            # Save error status immediately
            save_recipe(RECIPE_PATH, recipe)
            
            torch.cuda.empty_cache()
            gc.collect()
    
    # Final save (in case any other updates were missed)
    print("\nSaving final recipe state...")
    save_recipe(RECIPE_PATH, recipe)
    
    # Summary
    print("\n" + "=" * 50)
    print(f"✅ Processed: {processed_count}")
    print(f"⏭️  Skipped: {skipped_count}")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()
