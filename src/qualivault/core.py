# -*- coding: utf-8 -*-

import os
import logging
import torch
import gc
import yaml
import re
import subprocess
from pathlib import Path
import warnings
from typing import Optional
from huggingface_hub import login
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pyannote.audio import Pipeline
from pyannote.core import Segment
import librosa
import soundfile as sf
import pandas as pd
import numpy as np

try:
    from IPython.display import clear_output
except ImportError:
    def clear_output(wait=False):
        pass

# Configure logging for clear user feedback
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("QualiVault")


def _safe_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "on"}:
            return True
        if v in {"0", "false", "no", "off"}:
            return False
    return default


def _flush_gpu_memory():
    """Flush GPU memory caches for all available backends (CUDA and MPS)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _split_chunks_by_diarization(chunks, diarization=None, precomputed_turns=None):
    """Split ASR chunks on diarization turn boundaries using proportional word allocation."""
    if not chunks:
        return chunks

    if precomputed_turns is not None:
        diar_turns = [(float(t[0]), float(t[1])) for t in precomputed_turns]
    elif diarization is not None:
        diar_turns = []
        for turn, _, _speaker in diarization.itertracks(yield_label=True):
            diar_turns.append((float(turn.start), float(turn.end)))
    else:
        return chunks

    if not diar_turns:
        return chunks

    diar_turns.sort(key=lambda x: x[0])
    split_chunks = []

    for chunk in chunks:
        ts = chunk.get("timestamp")
        if not isinstance(ts, (tuple, list)) or len(ts) != 2:
            split_chunks.append(chunk)
            continue

        c_start, c_end = ts
        if c_start is None or c_end is None:
            split_chunks.append(chunk)
            continue

        c_start = float(c_start)
        c_end = float(c_end)
        if c_end <= c_start:
            split_chunks.append(chunk)
            continue

        text = str(chunk.get("text", "")).strip()
        words = text.split()
        if not words:
            split_chunks.append(chunk)
            continue

        overlaps = []
        for d_start, d_end in diar_turns:
            ov_start = max(c_start, d_start)
            ov_end = min(c_end, d_end)
            if ov_end > ov_start:
                overlaps.append((ov_start, ov_end))

        if len(overlaps) <= 1:
            split_chunks.append(chunk)
            continue

        durations = [max(1e-6, e - s) for s, e in overlaps]
        total = sum(durations)
        raw_counts = [len(words) * (d / total) for d in durations]
        counts = [max(1, int(x)) for x in raw_counts]

        # Balance counts so assigned word totals match exactly.
        delta = len(words) - sum(counts)
        if delta > 0:
            remainders = sorted(
                [(i, raw_counts[i] - int(raw_counts[i])) for i in range(len(raw_counts))],
                key=lambda t: t[1],
                reverse=True,
            )
            idx = 0
            while delta > 0:
                counts[remainders[idx % len(remainders)][0]] += 1
                delta -= 1
                idx += 1
        elif delta < 0:
            overs = sorted(
                [(i, counts[i] - raw_counts[i]) for i in range(len(raw_counts))],
                key=lambda t: t[1],
                reverse=True,
            )
            idx = 0
            made_progress = True
            while delta < 0 and overs and made_progress:
                made_progress = False
                for i in range(len(overs)):
                    target = overs[(idx + i) % len(overs)][0]
                    if counts[target] > 1:
                        counts[target] -= 1
                        delta += 1
                        idx = (idx + i + 1) % len(overs)
                        made_progress = True
                        break

        pos = 0
        for (s, e), n_words in zip(overlaps, counts):
            n = max(1, int(n_words))
            if pos >= len(words):
                break
            if pos + n > len(words):
                n = len(words) - pos
            seg_text = " ".join(words[pos:pos + n]).strip()
            pos += n
            if not seg_text:
                continue
            split_chunks.append({"timestamp": (float(s), float(e)), "text": seg_text})

    return split_chunks


def _chunk_overlaps_speech(start_s: float, end_s: float, speech_segments) -> bool:
    if end_s <= start_s:
        return False
    for seg in speech_segments:
        s0 = float(seg.start)
        s1 = float(seg.end)
        if min(end_s, s1) > max(start_s, s0):
            return True
    return False

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
    Expects multi-stage status fields on each item:
      - convert_status: pending|converted|error
      - analysis_status: pending|analyzed|error
      - transcribe_status: pending|transcribed|error
      - status: summary (mirrors the latest stage)
      - last_good_status: last successful stage label
    Items are transcribed only when conversion and analysis have succeeded.
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
    transcription_config = config.get("transcription", {}) if config else {}
    transcriber = Transcriber(device=device, cache_dir=cache_dir, config=transcription_config)
    
    # Prepare Transcription Settings
    language = "da"
    prompt_ids = None
    language = transcription_config.get("language", "da")
    topic_prompt = transcription_config.get("topic_prompt")
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
        # Track original values to see if we mutate
        orig = {
            'status': item.get('status'),
            'last_good_status': item.get('last_good_status'),
            'convert_status': item.get('convert_status'),
            'analysis_status': item.get('analysis_status'),
            'transcribe_status': item.get('transcribe_status'),
        }

        # Ensure new multi-stage fields exist for backward compatibility
        item.setdefault('status', 'pending')
        item.setdefault('last_good_status', item.get('last_good_status', 'none'))

        # Heuristic defaults
        flac_exists = (flac_dir / item.get('output_name', '')).exists()
        has_analysis = 'separation_status' in item or 'speakers_left' in item or 'separation_db' in item
        has_transcript = 'transcript_path' in item and Path(item['transcript_path']).exists()

        item.setdefault('convert_status', 'converted' if flac_exists else item.get('status', 'pending'))
        item.setdefault('analysis_status', 'analyzed' if has_analysis else item.get('status', 'pending'))
        item.setdefault('transcribe_status', 'transcribed' if has_transcript else 'pending')

        # Normalize legacy values
        if item.get('convert_status') in ['completed', 'done']:
            item['convert_status'] = 'converted'
        if item.get('analysis_status') in ['completed', 'done']:
            item['analysis_status'] = 'analyzed'
        if item.get('transcribe_status') in ['completed', 'done']:
            item['transcribe_status'] = 'transcribed'

        # Set last_good_status to highest completed stage if unknown
        if item.get('last_good_status', 'none') == 'none':
            if item.get('transcribe_status') == 'transcribed':
                item['last_good_status'] = 'transcribed'
            elif item.get('analysis_status') == 'analyzed':
                item['last_good_status'] = 'analyzed'
            elif item.get('convert_status') == 'converted':
                item['last_good_status'] = 'converted'

        # If normalization changed anything, persist later
        if any([
            item.get('status') != orig['status'],
            item.get('last_good_status') != orig['last_good_status'],
            item.get('convert_status') != orig['convert_status'],
            item.get('analysis_status') != orig['analysis_status'],
            item.get('transcribe_status') != orig['transcribe_status'],
        ]):
            updated = True

        convert_status = item.get('convert_status')
        analysis_status = item.get('analysis_status')
        transcribe_status = item.get('transcribe_status')

        # Skip already transcribed
        if transcribe_status == 'transcribed':
            logger.info(f"⏭️  Skipping {item['output_name']} (transcribed)")
            continue

        # Skip until previous stages are good
        if convert_status != 'converted':
            logger.info(f"⏭️  Waiting for conversion: {item['output_name']} (convert_status={convert_status})")
            continue
        if analysis_status != 'analyzed':
            logger.info(f"⏭️  Waiting for analysis: {item['output_name']} (analysis_status={analysis_status})")
            continue

        # Skip errors unless manually reset to pending
        if transcribe_status == 'error':
            logger.info(f"⏭️  Previous transcription error: {item['output_name']} (reset transcribe_status to 'pending' to retry)")
            continue

        # Ready to transcribe
        clear_output(wait=True)
        try:
            csv_path = process_item(item, flac_dir, output_dir, transcriber, diarization_pipeline, language, prompt_ids)
            if csv_path:
                item['transcribe_status'] = 'transcribed'
                item['status'] = 'transcribed'
                item['last_good_status'] = 'transcribed'
                item['transcript_path'] = str(csv_path)
                updated = True
                with open(recipe_path, 'w') as f:
                    yaml.dump(recipe, f, sort_keys=False)
        except Exception as e:
            logger.error(f"❌ Failed processing {item['output_name']}: {e}")
            item['transcribe_status'] = 'error'
            item['status'] = 'error'
            item['error_msg'] = str(e)
            updated = True
            with open(recipe_path, 'w') as f:
                yaml.dump(recipe, f, sort_keys=False)
            continue

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
    
    # Prepare mono waveform for Whisper and keep stereo when available for optional diarization.
    if y.ndim > 1:
        y_stereo = y
        y_mono = librosa.to_mono(y)
    else:
        y_stereo = None
        y_mono = y

    # Keep original stereo/mono array out of memory as soon as mono is prepared.
    del y
    gc.collect()

    # Read metadata from recipe
    spk_left = int(item.get('speakers_left', 0))
    spk_right = int(item.get('speakers_right', 0))
    sep_db = float(item.get('separation_db', 0))
    transcription_config = getattr(transcriber, "config", {}) or {}
    diarization_config = transcription_config.get("diarization", {})
    segment_config = transcription_config.get("segmentation", {})
    preprocessing_config = transcription_config.get("preprocessing", {})
    stereo_config = transcription_config.get("stereo_hint", {})
    adaptive_config = transcription_config.get("adaptive_tuning", {})
    use_tuning_decode_path = _safe_bool(segment_config.get("use_tuning_decode_path", False), False)
    tuning_left_context_s = max(0.0, _safe_float(segment_config.get("tuning_left_context_s"), 35.0))
    tuning_right_context_s = max(0.0, _safe_float(segment_config.get("tuning_right_context_s"), 14.0))

    declared_speakers = _safe_int(item.get("speakers"))
    separation_status = str(item.get("separation_status", "")).strip().lower()

    merge_gap_seconds = max(0.0, _safe_float(segment_config.get("merge_gap_seconds"), 1.0))
    final_merge_gap_seconds = max(0.0, _safe_float(segment_config.get("final_merge_gap_seconds"), 0.75))
    max_segment_duration_seconds = max(1.0, _safe_float(segment_config.get("max_segment_duration_seconds"), 45.0))
    split_by_speaker_turns = _safe_bool(diarization_config.get("split_transcript_by_speaker_turns", True), True)

    min_speakers = max(1, _safe_int(diarization_config.get("min_speakers")) or 2)
    configured_max_speakers = _safe_int(diarization_config.get("max_speakers"))
    speakers_policy = str(diarization_config.get("speakers_policy", "strict")).strip().lower()
    speaker_count_source = str(diarization_config.get("speaker_count_source", "config")).strip().lower()
    use_stereo_when_available = _safe_bool(diarization_config.get("use_stereo_when_available", True), True)

    # Optional preprocessing before diarization/transcription.
    trim_silence = _safe_bool(preprocessing_config.get("trim_silence", True), True)
    trim_top_db = max(10.0, _safe_float(preprocessing_config.get("trim_silence_top_db"), 30.0))
    normalize_audio = _safe_bool(preprocessing_config.get("normalize_audio", True), True)
    normalize_target_peak = max(0.01, _safe_float(preprocessing_config.get("normalize_target_peak"), 0.95))
    denoise_strength = max(0.0, min(1.0, _safe_float(preprocessing_config.get("denoise_strength"), 0.0)))

    if use_tuning_decode_path:
        # Match tuning notebook preprocessing path for ASR quality parity.
        trim_silence = False
        normalize_audio = False
        logger.info("   🧪 Tuning decode path enabled: trim_silence=False, normalize_audio=False")

    if trim_silence:
        try:
            y_mono, trim_idx = librosa.effects.trim(y_mono, top_db=trim_top_db)
            if y_stereo is not None and trim_idx is not None:
                start_i, end_i = int(trim_idx[0]), int(trim_idx[1])
                y_stereo = y_stereo[:, start_i:end_i]
        except Exception:
            pass

    if denoise_strength > 0.0:
        try:
            import noisereduce as nr
            y_mono = nr.reduce_noise(y=y_mono, sr=sr, prop_decrease=denoise_strength, stationary=True)
        except Exception as e:
            logger.warning(f"   ⚠️ Preprocessing denoise skipped: {e}")

    if normalize_audio and len(y_mono) > 0:
        peak = float(np.max(np.abs(y_mono)))
        if peak > 0:
            gain = normalize_target_peak / peak
            y_mono = np.clip(y_mono * gain, -1.0, 1.0)
            if y_stereo is not None:
                y_stereo = np.clip(y_stereo * gain, -1.0, 1.0)

    if len(y_mono) == 0:
        logger.warning("   ⚠️ Preprocessing produced empty audio; falling back to untrimmed waveform.")
        y_mono, _ = librosa.load(audio_path, sr=16000, mono=True)
        if y_stereo is not None:
            try:
                y_stereo, _ = librosa.load(audio_path, sr=16000, mono=False)
            except Exception:
                y_stereo = None

    # Adaptive tuning for low-quality files.
    adaptive_enabled = _safe_bool(adaptive_config.get("enabled", True), True)
    if adaptive_enabled and separation_status == "poor":
        merge_gap_seconds = min(merge_gap_seconds, max(0.0, _safe_float(adaptive_config.get("poor_audio_merge_gap_seconds"), 0.25)))
        final_merge_gap_seconds = min(final_merge_gap_seconds, max(0.0, _safe_float(adaptive_config.get("poor_audio_final_merge_gap_seconds"), 0.25)))
        max_segment_duration_seconds = min(max_segment_duration_seconds, max(1.0, _safe_float(adaptive_config.get("poor_audio_max_segment_duration_seconds"), 15.0)))

    temp_wav = output_dir / "temp_process.wav"
    sf.write(temp_wav, y_mono, sr)

    use_stereo_for_diarization = bool(
        use_stereo_when_available
        and y_stereo is not None
        and np.ndim(y_stereo) > 1
        and y_stereo.shape[0] >= 2
    )

    if use_stereo_for_diarization:
        temp_diar_wav = output_dir / "temp_process_diar.wav"
        sf.write(temp_diar_wav, y_stereo.T, sr)
        diarization_input = temp_diar_wav
        logger.info("   🎚️ Diarization input: stereo (preprocessed)")
    else:
        temp_diar_wav = temp_wav
        diarization_input = temp_wav
        logger.info("   🎚️ Diarization input: mono (preprocessed)")

    del y_mono
    gc.collect()

    # 1. Diarization
    logger.info("   🧠 Running Diarization...")
    # Use per-interview speaker count when available.
    min_spk = min_speakers
    inferred_max_speakers = max(min_spk, spk_left + spk_right) if (spk_left + spk_right) > 0 else None
    if configured_max_speakers is not None:
        max_spk = max(min_spk, configured_max_speakers)
    else:
        max_spk = inferred_max_speakers

    diarization_kwargs = {}
    # Speaker-count source modes:
    # - recipe: use item['speakers'] as exact num_speakers when available
    # - config: prefer config min/max bounds (default)
    # - auto: use legacy policy-based behavior
    if speaker_count_source == "recipe" and declared_speakers and declared_speakers > 0:
        diarization_kwargs["num_speakers"] = int(max(1, declared_speakers))
    elif speaker_count_source == "config":
        diarization_kwargs["min_speakers"] = min_spk
        if configured_max_speakers is not None:
            diarization_kwargs["max_speakers"] = max(min_spk, configured_max_speakers)
        elif max_spk is not None:
            if adaptive_enabled and separation_status == "poor":
                max_spk = min(max_spk, min_spk + 1)
            diarization_kwargs["max_speakers"] = max_spk
    elif declared_speakers and declared_speakers > 0 and speakers_policy == "strict":
        diarization_kwargs["num_speakers"] = declared_speakers
    elif declared_speakers and declared_speakers > 0 and speakers_policy == "bounded":
        diarization_kwargs["min_speakers"] = max(1, declared_speakers)
        diarization_kwargs["max_speakers"] = max(declared_speakers, declared_speakers + 1)
    else:
        if adaptive_enabled and separation_status == "poor" and max_spk is not None:
            max_spk = min(max_spk, min_spk + 1)
        diarization_kwargs["min_speakers"] = min_spk
        if max_spk is not None:
            diarization_kwargs["max_speakers"] = max_spk

    logger.info(f"   🧾 Diarization kwargs: {diarization_kwargs}")

    diarization = diarization_pipeline(str(diarization_input), **diarization_kwargs)
    
    # Memory cleanup after Diarization to prepare for Whisper
    _flush_gpu_memory()

    # 2. Transcription (with diarization-guided filtering and speaker assignment)
    logger.info("   ✍️  Transcribing...")
    
    # Get speech timeline from diarization (merges overlapping speech)
    speech_timeline = diarization.get_timeline().support()
    
    # Merge short silence gaps before chunking for Whisper.
    merged_segments = []
    if len(speech_timeline) > 0:
        sorted_segs = sorted(speech_timeline, key=lambda s: s.start)
        current_seg = sorted_segs[0]
        
        for next_seg in sorted_segs[1:]:
            if next_seg.start - current_seg.end < merge_gap_seconds:
                current_seg = Segment(current_seg.start, next_seg.end)
            else:
                merged_segments.append(current_seg)
                current_seg = next_seg
        merged_segments.append(current_seg)
    
    transcribe_mode = str(segment_config.get("transcribe_mode", "full_audio_windowed")).strip().lower()
    if transcribe_mode not in {"full_audio", "full_audio_windowed", "speech_segments"}:
        transcribe_mode = "full_audio_windowed"

    logger.info(f"   🧭 Transcribe mode: {transcribe_mode}")

    transcript_chunks = []
    if transcribe_mode == "speech_segments":
        for seg in merged_segments:
            # Read Audio Chunk (Memory Efficient)
            start_frame = int(seg.start * 16000)
            frames = int(seg.duration * 16000)
            y_chunk, _ = sf.read(str(temp_wav), start=start_frame, frames=frames)

            # Transcribe Chunk
            res = transcriber.transcribe(y_chunk, language=language, prompt_ids=prompt_ids)

            # Adjust timestamps to be absolute
            if res and 'chunks' in res:
                for chunk in res['chunks']:
                    start_rel, end_rel = chunk['timestamp']
                    # Handle missing timestamps (Whisper sometimes fails to predict end)
                    if start_rel is None:
                        start_rel = 0.0
                    if end_rel is None:
                        end_rel = seg.duration

                    chunk['timestamp'] = (start_rel + seg.start, end_rel + seg.start)
                    transcript_chunks.append(chunk)
    elif transcribe_mode == "full_audio":
        # Match tuning notebook behavior: decode a continuous waveform for better context,
        # then keep only chunks that overlap diarized speech.
        res = transcriber.transcribe(str(temp_wav), language=language, prompt_ids=prompt_ids)
        if res and 'chunks' in res:
            for chunk in res['chunks']:
                start_abs, end_abs = chunk['timestamp']
                if start_abs is None:
                    start_abs = 0.0
                if end_abs is None:
                    end_abs = float(start_abs)
                start_abs = float(start_abs)
                end_abs = float(end_abs)
                if not _chunk_overlaps_speech(start_abs, end_abs, speech_timeline):
                    continue
                chunk['timestamp'] = (start_abs, end_abs)
                transcript_chunks.append(chunk)
    else:
        # Windowed full-audio mode avoids a single long blocking ASR call while
        # preserving broader context than diarized micro-segments.
        info = sf.info(str(temp_wav))
        total_duration_s = float(info.frames) / float(info.samplerate) if info and info.samplerate else 0.0
        window_s = max(30.0, _safe_float(segment_config.get("full_audio_window_s"), 180.0))
        overlap_s = max(0.0, min(window_s * 0.4, _safe_float(segment_config.get("full_audio_window_overlap_s"), 8.0)))
        step_s = max(1.0, window_s - overlap_s)

        # Build windows first so we can avoid a tiny tail-only final window,
        # which is more likely to produce unstable end-of-audio behavior.
        windows = []
        win_start = 0.0
        while win_start < total_duration_s:
            win_end = min(total_duration_s, win_start + window_s)
            if win_end - win_start <= 0.0:
                break
            windows.append((win_start, win_end))
            if win_end >= total_duration_s:
                break
            win_start += step_s

        min_tail_s = max(20.0, min(90.0, window_s * 0.5))
        if len(windows) >= 2:
            last_start, last_end = windows[-1]
            if (last_end - last_start) < min_tail_s:
                prev_start, _prev_end = windows[-2]
                windows[-2] = (prev_start, total_duration_s)
                windows.pop()

        total_windows = len(windows)
        for idx, (win_start, win_end) in enumerate(windows, start=1):
            win_dur = max(0.0, win_end - win_start)
            if win_dur <= 0.0:
                continue

            logger.info(
                f"   ⏱️ ASR window {idx}/{total_windows}: {win_start:.1f}s -> {win_end:.1f}s (/{total_duration_s:.1f}s)"
            )

            if use_tuning_decode_path:
                # Match tuning notebook behavior: decode an expanded window and
                # keep only transcript chunks that overlap the target window.
                expanded_start = max(0.0, win_start - tuning_left_context_s)
                expanded_end = min(total_duration_s, win_end + tuning_right_context_s)
                expanded_dur = max(0.05, expanded_end - expanded_start)
                y_win, _ = librosa.load(
                    temp_wav,
                    sr=16000,
                    mono=True,
                    offset=expanded_start,
                    duration=expanded_dur,
                )
                local_target_start = win_start - expanded_start
                local_target_end = win_end - expanded_start
            else:
                start_frame = int(win_start * 16000)
                frames = int(win_dur * 16000)
                y_win, _ = sf.read(str(temp_wav), start=start_frame, frames=frames)

            res = transcriber.transcribe(y_win, language=language, prompt_ids=prompt_ids)
            if res and 'chunks' in res:
                # Keep only the center region for non-edge windows to reduce overlap duplicates.
                keep_start = win_start + (overlap_s / 2.0 if win_start > 0.0 else 0.0)
                keep_end = win_end - (overlap_s / 2.0 if win_end < total_duration_s else 0.0)

                for chunk in res['chunks']:
                    start_rel, end_rel = chunk['timestamp']
                    if start_rel is None:
                        start_rel = 0.0
                    if end_rel is None:
                        end_rel = float(start_rel)

                    if use_tuning_decode_path:
                        start_rel = float(start_rel)
                        end_rel = float(end_rel)
                        overlap = max(0.0, min(end_rel, local_target_end) - max(start_rel, local_target_start))
                        if overlap <= 0.0:
                            continue
                        start_abs = start_rel + expanded_start
                        end_abs = end_rel + expanded_start
                    else:
                        start_abs = float(start_rel) + win_start
                        end_abs = float(end_rel) + win_start

                    if end_abs <= keep_start or start_abs >= keep_end:
                        continue
                    if not _chunk_overlaps_speech(start_abs, end_abs, speech_timeline):
                        continue

                    chunk['timestamp'] = (start_abs, end_abs)
                    transcript_chunks.append(chunk)

            logger.info(f"   ✅ ASR window {idx}/{total_windows} completed")

            _flush_gpu_memory()

    logger.info(f"   ✅ ASR decoding finished. Collected chunks: {len(transcript_chunks)}")

    # Precompute diarization turns ONCE to avoid repeated itertracks() calls.
    logger.info("   🗂️ Precomputing diarization turns...")
    diar_turns = [(float(turn.start), float(turn.end), speaker) for turn, _, speaker in diarization.itertracks(yield_label=True)]
    logger.info(f"   ✅ {len(diar_turns)} diarization turns precomputed")

    if split_by_speaker_turns and transcript_chunks:
        before_count = len(transcript_chunks)
        transcript_chunks = _split_chunks_by_diarization(transcript_chunks, precomputed_turns=diar_turns)
        logger.info(f"   🧩 Speaker-turn split: {before_count} -> {len(transcript_chunks)} chunks")

    # Build optional channel->speaker hints using in-memory stereo array (no disk I/O).
    stereo_hint_enabled = _safe_bool(stereo_config.get("enabled", True), True)
    stereo_dom_db = max(0.5, _safe_float(stereo_config.get("dominance_db_threshold"), 3.0))
    stereo_overlap_threshold = max(0.0, _safe_float(stereo_config.get("low_overlap_seconds"), 0.15))
    channel_speaker_counts = {"left": {}, "right": {}}
    is_stereo = y_stereo is not None and y_stereo.ndim == 2 and y_stereo.shape[0] >= 2

    if stereo_hint_enabled and is_stereo:
        logger.info(f"   📊 Building stereo channel priors ({len(diar_turns)} turns)...")
        for turn_start, turn_end, speaker in diar_turns:
            start_frame = int(turn_start * 16000)
            end_frame = min(y_stereo.shape[1], start_frame + max(1, int((turn_end - turn_start) * 16000)))
            if end_frame <= start_frame:
                continue
            seg_l = y_stereo[0, start_frame:end_frame]
            seg_r = y_stereo[1, start_frame:end_frame]
            rms_l = float(np.sqrt(np.mean(np.square(seg_l)))) + 1e-8
            rms_r = float(np.sqrt(np.mean(np.square(seg_r)))) + 1e-8
            db = 20.0 * np.log10(rms_l / rms_r)
            if db >= stereo_dom_db:
                channel_speaker_counts["left"][speaker] = channel_speaker_counts["left"].get(speaker, 0) + 1
            elif db <= -stereo_dom_db:
                channel_speaker_counts["right"][speaker] = channel_speaker_counts["right"].get(speaker, 0) + 1
        logger.info("   ✅ Stereo channel priors built.")

    dominant_left_speaker = max(channel_speaker_counts["left"], key=channel_speaker_counts["left"].get) if channel_speaker_counts["left"] else None
    dominant_right_speaker = max(channel_speaker_counts["right"], key=channel_speaker_counts["right"].get) if channel_speaker_counts["right"] else None

    # 3. Merge & Identify
    logger.info("   🔗 Merging text into paragraphs...")
    final_segments = []
    current_speaker = None
    current_text_buffer = []
    current_start = 0.0
    current_end = 0.0
    
    # Hybrid Logic: If separation is good (>5dB) AND Left channel has exactly 1 speaker (Interviewer)
    is_hybrid = sep_db > 5.0 and spk_left == 1

    for idx, chunk in enumerate(transcript_chunks):
        start, end = chunk['timestamp']
        text = chunk['text']
        
        # Find best speaker match from diarization
        best_speaker = "UNKNOWN"
        max_overlap = 0
        
        for turn_start, turn_end, speaker in diar_turns:
            overlap = max(0, min(end, turn_end) - max(start, turn_start))
            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = speaker

        if idx > 0 and idx % 2000 == 0:
            logger.info(f"   🔎 Speaker matching progress: {idx}/{len(transcript_chunks)} chunks")
        
        # Hybrid Logic Override (uses in-memory stereo array — no disk I/O)
        if is_hybrid and is_stereo:
            start_frame = int(start * 16000)
            end_frame = min(y_stereo.shape[1], start_frame + max(1, int((end - start) * 16000)))
            if end_frame > start_frame:
                seg_l = y_stereo[0, start_frame:end_frame]
                seg_r = y_stereo[1, start_frame:end_frame]
                rms_l = float(np.sqrt(np.mean(np.square(seg_l))))
                rms_r = float(np.sqrt(np.mean(np.square(seg_r))))
                if rms_l > rms_r:
                    best_speaker = "LEFT (Student)"

        # Stereo-aware fallback: if diarization overlap is weak, use channel-dominance priors.
        if stereo_hint_enabled and is_stereo and max_overlap <= stereo_overlap_threshold:
            start_frame = int(start * 16000)
            end_frame = min(y_stereo.shape[1], start_frame + max(1, int((end - start) * 16000)))
            if end_frame > start_frame:
                seg_l = y_stereo[0, start_frame:end_frame]
                seg_r = y_stereo[1, start_frame:end_frame]
                rms_l = float(np.sqrt(np.mean(np.square(seg_l)))) + 1e-8
                rms_r = float(np.sqrt(np.mean(np.square(seg_r)))) + 1e-8
                db = 20.0 * np.log10(rms_l / rms_r)
                if db >= stereo_dom_db and dominant_left_speaker:
                    best_speaker = dominant_left_speaker
                elif db <= -stereo_dom_db and dominant_right_speaker:
                    best_speaker = dominant_right_speaker
        
        gap_from_previous_chunk = max(0.0, start - current_end) if current_speaker is not None else 0.0
        projected_duration = end - current_start if current_speaker is not None else 0.0
        should_merge = (
            best_speaker == current_speaker
            and gap_from_previous_chunk <= final_merge_gap_seconds
            and projected_duration <= max_segment_duration_seconds
        )

        # Merge Logic
        if should_merge:
            current_text_buffer.append(text)
            current_end = end
        else:
            if current_speaker is not None:
                final_segments.append({
                    "Start": round(current_start, 2),
                    "End": round(current_end, 2),
                    "Speaker": current_speaker,
                    "Text": " ".join(current_text_buffer)
                })
            current_speaker = best_speaker
            current_text_buffer = [text]
            current_start = start
            current_end = end

    # Append final segment
    if current_speaker is not None:
        final_segments.append({
            "Start": round(current_start, 2),
            "End": round(current_end, 2),
            "Speaker": current_speaker,
            "Text": " ".join(current_text_buffer)
        })

    df = pd.DataFrame(final_segments)
    csv_name = output_dir / filename.replace('.flac', '.csv')
    df.to_csv(csv_name, index=False)
    logger.info(f"   🎉 Saved: {csv_name}")

    # 🧹 DEEP CLEAN MEMORY
    # (Note: `diarization` is a local variable, safe to delete.)
    try:
        del diarization
    except Exception:
        pass
    try:
        del transcript_chunks
    except Exception:
        pass
    try:
        del y_stereo
    except Exception:
        pass
    if temp_wav.exists():
        temp_wav.unlink()
    if temp_diar_wav != temp_wav and temp_diar_wav.exists():
        temp_diar_wav.unlink()

    _flush_gpu_memory()

    return csv_name

class Transcriber:
    def __init__(self, device=None, model_id="openai/whisper-large-v3", cache_dir=None, config=None):
        """
        Initializes the Whisper model.
        """
        self.config = config or {}
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
            chunk_length_s=float(self.config.get("chunk_length_s", 30)),
            stride_length_s=(
                float(self.config.get("stride_left_s", 5)),
                float(self.config.get("stride_right_s", 5)),
            ),
            batch_size=self.config.get("batch_size", 1),
            return_timestamps=True,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
        self._logged_decode_kwargs = False
        logger.info("✅ Model Loaded.")

    def transcribe(self, audio_input, language="da", prompt_ids=None):
        # logger.info(f"🎙️ Transcribing segment...") # Reduced logging for segment loop
        
        # Handle both file path and numpy array
        if isinstance(audio_input, (str, Path)):
            inputs = str(audio_input)
        else:
            # Use direct ndarray input (not dict) to avoid intermittent dict-key
            # validation failures in ASR pipeline chunking mode.
            arr = np.asarray(audio_input)
            if arr.ndim > 1:
                # Collapse potential stereo chunks to mono for Whisper.
                arr = arr.mean(axis=-1)
            arr = np.squeeze(arr)
            if arr.ndim != 1:
                arr = arr.reshape(-1)
            inputs = np.ascontiguousarray(arr, dtype=np.float32)
        
        # Build generate_kwargs from config so notebook-tuned values can be reused in pipeline.
        generate_kwargs = {
            "language": language, 
            "task": "transcribe"
        }

        if "num_beams" in self.config:
            try:
                generate_kwargs["num_beams"] = int(self.config.get("num_beams", 1))
            except Exception:
                pass

        if "condition_on_prev_tokens" in self.config:
            generate_kwargs["condition_on_prev_tokens"] = bool(self.config.get("condition_on_prev_tokens", False))

        if "repetition_penalty" in self.config:
            try:
                generate_kwargs["repetition_penalty"] = float(self.config.get("repetition_penalty", 1.0))
            except Exception:
                pass

        if "no_speech_threshold" in self.config:
            try:
                generate_kwargs["no_speech_threshold"] = float(self.config.get("no_speech_threshold"))
            except Exception:
                pass
        
        # Only add prompt_ids if provided
        if prompt_ids is not None:
            generate_kwargs["prompt_ids"] = prompt_ids

        if not self._logged_decode_kwargs:
            safe_kwargs = {k: v for k, v in generate_kwargs.items() if k != "prompt_ids"}
            logger.info(f"   🧪 Effective Whisper generate kwargs: {safe_kwargs}")
            if prompt_ids is not None:
                try:
                    prompt_len = int(prompt_ids.shape[-1]) if hasattr(prompt_ids, "shape") else len(prompt_ids)
                except Exception:
                    prompt_len = None
                logger.info(f"   🧪 Prompt IDs attached: {prompt_len if prompt_len is not None else 'yes'}")
            self._logged_decode_kwargs = True

        try:
            result = self.pipe(inputs, generate_kwargs=generate_kwargs)
        except UnboundLocalError as ex:
            # Compatibility fallback for some transformers versions where
            # no_speech_threshold can trigger a logprobs bug.
            if "logprobs" not in str(ex).lower():
                raise
            retry_kwargs = dict(generate_kwargs)
            retry_kwargs["no_speech_threshold"] = None
            result = self.pipe(inputs, generate_kwargs=retry_kwargs)
        return result

def to_markdown(csv_path, output_dir, speaker_map=None):
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
        
        # Apply Speaker Map
        if speaker_map and speaker in speaker_map:
            speaker = speaker_map[speaker]
        
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

def export_recipe(recipe_path, output_dir, config=None):
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
    
    speaker_map = config.get('speaker_map', {}) if config else {}
    
    count = 0
    for item in recipe:
        # Prefer explicit transcript_path if present; otherwise infer from output_name.
        csv_path = None
        if 'transcript_path' in item:
            csv_path = Path(item['transcript_path'])
        elif 'output_name' in item and str(item['output_name']).lower().endswith('.flac'):
            csv_path = output_dir / str(item['output_name']).replace('.flac', '.csv')

        if csv_path and csv_path.exists():
            to_markdown(csv_path, obsidian_dir, speaker_map=speaker_map)
            count += 1
    
    logger.info(f"✅ Exported {count} transcripts to Markdown.")