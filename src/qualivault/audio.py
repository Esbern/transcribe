import subprocess
import logging
import json
import shutil
from pathlib import Path

logger = logging.getLogger("QualiVault")

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