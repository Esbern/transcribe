
import os
import subprocess
import glob
from multiprocessing import Pool, cpu_count

# --- CONFIG ---
INPUT_DIR = r"D:\ledgendary\flac audio"
OUTPUT_DIR = r"D:\ledgendary\mp3 audio"
FFMPEG_CMD = "ffmpeg" # If ffmpeg is not in PATH, put full path to ffmpeg.exe here
MAX_WORKERS = cpu_count() # Use all CPU cores, or set to a specific number

# Audio quality presets (flip ACTIVE_PROFILE to switch)
# - "talk_min": smallest speech-friendly files (mono 64kbps)
# - "max_quality": best MP3 quality for Whisper (stereo 320kbps)
PROFILES = {
    "talk_min": {"channels": 1, "bitrate": "64k"},
    "max_quality": {"channels": 2, "bitrate": "320k"},
}
ACTIVE_PROFILE = "max_quality"

PROFILE = PROFILES[ACTIVE_PROFILE]
AUDIO_CHANNELS = PROFILE["channels"]
BITRATE = PROFILE["bitrate"]

def process_single_file(file_path):
    """Process a single FLAC file to MP3."""
    filename = os.path.basename(file_path)
    name_only = os.path.splitext(filename)[0]
    output_path = os.path.join(OUTPUT_DIR, f"{name_only}.mp3")

    if os.path.exists(output_path):
        return f"⏭️ Skipped {filename} (already exists)"

    # --- THE AUDIO CHAIN EXPLAINED ---
    # 1. highpass=f=100: Removes rumble/hum below 100Hz (AC units, mic handling)
    # 2. afftdn=nr=15: Reduces background noise by 15 decibels (Safe amount to avoid robotic voice)
    # 3. loudnorm: Normalizes perceived loudness to broadcast standard (-16 LUFS)
    audio_filter = "highpass=f=100, afftdn=nr=15, loudnorm=I=-16:TP=-1.5:LRA=11"

    cmd = [
        FFMPEG_CMD, 
        '-i', file_path,
        '-af', audio_filter, # Apply our enhancement chain
        '-ac', str(AUDIO_CHANNELS),  # Stereo for Whisper
        '-b:a', BITRATE,      # High quality MP3
        output_path
    ]
    
    try:
        # We use a slight trick here to hide the massive ffmpeg log but show errors
        subprocess.run(cmd, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        return f"✅ Processed {filename}"
    except subprocess.CalledProcessError as e:
        return f"❌ Error converting {filename}: {e.stderr.decode()}"

def compress_and_enhance():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    files = glob.glob(os.path.join(INPUT_DIR, "*.flac"))
    print(f"Found {len(files)} files to process...")
    print(f"Using {MAX_WORKERS} parallel workers...")
    print(f"Audio profile: {ACTIVE_PROFILE} ({AUDIO_CHANNELS}ch @ {BITRATE})")

    # Process files in parallel
    with Pool(MAX_WORKERS) as pool:
        results = pool.map(process_single_file, files)
    
    # Print all results
    for result in results:
        print(result)

    print(f"\n✅ All done! Enhanced files are in {OUTPUT_DIR}")

if __name__ == "__main__":
    compress_and_enhance()