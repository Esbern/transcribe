import subprocess
import logging
import json
import shutil
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf
import noisereduce as nr
import torch

logger = logging.getLogger("QualiVault")

def trim_silence(y: np.ndarray, top_db: float = 30.0, ref_percentile: float = 95.0, **kwargs):
    """Trim leading/trailing silence, robust to brief loud transients (mic bumps, door slams).

    `librosa.effects.trim`'s default reference is the signal's peak amplitude, so a single
    loud moment anywhere in a recording can make `top_db` far too strict for the rest of the
    file -- on real field recordings this has silently trimmed away many minutes of genuine
    speech. Using a high percentile of per-frame RMS as the reference instead only shifts if a
    meaningful fraction of frames are that loud, not just one brief spike.
    """
    ref = lambda S: np.percentile(S, ref_percentile)
    return librosa.effects.trim(y, top_db=top_db, ref=ref, **kwargs)


def check_ffmpeg():
    """Ensures ffmpeg is installed."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("❌ ffmpeg is not found! Please install it (e.g., `brew install ffmpeg` or `conda install ffmpeg`).")

def get_audio_info(file_path):
    """
    Returns a dict with 'channels', 'duration', 'sample_rate'.
    Uses ffprobe.
    """
    check_ffmpeg()
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_entries", "stream=channels,sample_rate,duration",
        str(file_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        stream = data["streams"][0]
        return {
            "channels": int(stream["channels"]),
            "duration": float(stream.get("duration", 0)),
            "sample_rate": int(stream["sample_rate"])
        }
    except Exception as e:
        logger.error(f"Error probing audio {file_path}: {e}")
        return None

def prepare_audio(input_files, output_path):
    """
    Concatenates multiple input files and converts to a single 16kHz FLAC.
    """
    check_ffmpeg()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"🎧 Processing {len(input_files)} file(s) -> {output_path.name}")

    # Build ffmpeg command
    cmd = ["ffmpeg", "-y"]
    
    # Input files
    for f in input_files:
        cmd.extend(["-i", str(f)])
    
    # Filter for concatenation if multiple files
    if len(input_files) > 1:
        filter_str = "".join([f"[{i}:0]" for i in range(len(input_files))])
        filter_str += f"concat=n={len(input_files)}:v=0:a=1[out]"
        cmd.extend(["-filter_complex", filter_str, "-map", "[out]"])
    
    # Output settings (16kHz FLAC for Whisper)
    cmd.extend(["-ar", "16000", "-sample_fmt", "s16", "-c:a", "flac", str(output_path)])

    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path

def split_stereo(input_path, output_dir):
    """
    Splits a stereo file into two mono files (Left and Right).
    Returns a tuple of paths: (left_path, right_path)
    """
    check_ffmpeg()
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    
    left_path = output_dir / f"{input_path.stem}_L.flac"
    right_path = output_dir / f"{input_path.stem}_R.flac"

    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-filter_complex", "[0:a]channelsplit=channel_layout=stereo[left][right]",
        "-map", "[left]", str(left_path),
        "-map", "[right]", str(right_path)
    ]
    
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return left_path, right_path

def diarization_annotation(result):
    """Normalize a pyannote Pipeline() call result to a plain Annotation.

    pyannote.audio <4.0 returns an Annotation directly (has .itertracks()).
    pyannote.audio >=4.0 returns a DiarizeOutput dataclass wrapping one as
    `.speaker_diarization`. This lets calling code use .itertracks() either way.
    """
    return getattr(result, "speaker_diarization", result)


def to_pyannote_waveform(audio: np.ndarray, sr: int) -> dict:
    """Wrap an already-loaded waveform as the in-memory input pyannote.audio expects.

    `audio` may be mono (1D, shape (time,)) or multi-channel (2D, shape (channels, time)).
    Passing a file path instead forces pyannote to decode via torchcodec, which requires an
    FFmpeg build torchcodec explicitly supports (currently 4-8). Passing the waveform we
    already have in memory sidesteps that dependency entirely.
    """
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    waveform = torch.from_numpy(audio)
    return {"waveform": waveform, "sample_rate": sr}


def analyze_channel_separation(audio_path, pipeline):
    print(f"  🔍 Diagnosing: {audio_path.name}")
    stats = {"separation_status": "Unknown", "speakers_left": 0, "speakers_right": 0}
    
    try:
        # 1. Load Audio (Stereo, first 60s for speed)
        y, sr = librosa.load(audio_path, mono=False, sr=16000, duration=60)
    except Exception as e:
        print(f"  ❌ Error loading audio: {e}")
        return stats

    # Handle Mono Files
    if y.ndim < 2:
        print("  ℹ️ File is Mono. Analyzing as single channel.")
        stats["separation_status"] = "Mono"
        
        # Still count speakers in the mono file
        if pipeline:
            print("  🧠 Counting Speakers (AI)...")
            try:
                diarization = diarization_annotation(pipeline(to_pyannote_waveform(y, sr)))
                stats["speakers_left"] = len(set(label for _, _, label in diarization.itertracks(yield_label=True)))
                print(f"     Speakers: {stats['speakers_left']}")
            except Exception as e:
                print(f"  ⚠️ Diarization error: {e}")

        return stats

    left_channel = y[0]
    right_channel = y[1]
    
    # --- METRIC 1: CHANNEL DOMINANCE ---
    frame_length = 16000 
    hop_length = 8000    
    
    rms_l = librosa.feature.rms(y=left_channel, frame_length=frame_length, hop_length=hop_length)[0]
    rms_r = librosa.feature.rms(y=right_channel, frame_length=frame_length, hop_length=hop_length)[0]
    
    rms_r = np.maximum(rms_r, 1e-6)
    rms_l = np.maximum(rms_l, 1e-6)
    
    db_diff = 20 * np.log10(rms_l / rms_r)
    
    silence_thresh = 0.005
    active_mask = (rms_l > silence_thresh) | (rms_r > silence_thresh)
    
    if np.any(active_mask):
        avg_separation = np.mean(np.abs(db_diff[active_mask]))
    else:
        avg_separation = 0.0
        
    print(f"  📊 Avg Separation: {avg_separation:.1f} dB")
    stats["separation_db"] = float(avg_separation)
    
    if avg_separation > 6.0:
        stats["separation_status"] = "Excellent"
        print("  ✅ Status: EXCELLENT Separation")
    elif avg_separation > 3.0:
        stats["separation_status"] = "Moderate"
        print("  ⚠️ Status: MODERATE Bleed")
    else:
        stats["separation_status"] = "Poor"
        print("  ❌ Status: POOR Separation")

    # --- METRIC 2: SPEAKER COUNT ---
    if pipeline:
        print("  🧠 Counting Speakers (AI)...")
        try:
            diarization_l = diarization_annotation(pipeline(to_pyannote_waveform(left_channel, sr)))
            stats["speakers_left"] = len(set(label for _, _, label in diarization_l.itertracks(yield_label=True)))
            print(f"     L : {stats['speakers_left']} speakers")

            diarization_r = diarization_annotation(pipeline(to_pyannote_waveform(right_channel, sr)))
            stats["speakers_right"] = len(set(label for _, _, label in diarization_r.itertracks(yield_label=True)))
            print(f"     R :  {stats['speakers_right']} speakers")
        except Exception as e:
            print(f"  ⚠️ Diarization error: {e}")

    return stats

def apply_noise_gate(input_path, output_path, prop_decrease=1.0):
    """
    Applies stationary noise reduction to the audio file.
    Useful for removing constant background hiss/hum.
    """
    check_ffmpeg()
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    logger.info(f"  🧹 Applying Noise Gate: {input_path.name}")
    
    try:
        # Load audio
        y, sr = librosa.load(input_path, sr=None, mono=False)
        
        # Apply reduction (stationary mode)
        y_reduced = nr.reduce_noise(y=y, sr=sr, prop_decrease=prop_decrease, stationary=True)
        
        # Save
        sf.write(output_path, y_reduced.T if y.ndim > 1 else y_reduced, sr)
        logger.info(f"  ✅ Saved cleaned audio: {output_path.name}")
        return True
    except Exception as e:
        logger.error(f"  ❌ Noise Gate failed: {e}")
        return False