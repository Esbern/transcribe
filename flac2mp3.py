
import os
import subprocess
import glob

# --- CONFIG ---
INPUT_DIR = r"E:\ledgendary\flac audio"
OUTPUT_DIR = r"E:\ledgendary\mp3 audio"
FFMPEG_CMD = "ffmpeg" # If ffmpeg is not in PATH, put full path to ffmpeg.exe here

def compress_and_enhance():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    files = glob.glob(os.path.join(INPUT_DIR, "*.flac"))
    print(f"Found {len(files)} files to process...")

    for file_path in files:
        filename = os.path.basename(file_path)
        name_only = os.path.splitext(filename)[0]
        output_path = os.path.join(OUTPUT_DIR, f"{name_only}.mp3")

        if os.path.exists(output_path):
            print(f"Skipping {filename} (already exists)")
            continue

        print(f"Enhancing & Compressing {filename}...")
        
        # --- THE AUDIO CHAIN EXPLAINED ---
        # 1. highpass=f=100: Removes rumble/hum below 100Hz (AC units, mic handling)
        # 2. afftdn=nr=15: Reduces background noise by 15 decibels (Safe amount to avoid robotic voice)
        # 3. loudnorm: Normalizes perceived loudness to broadcast standard (-16 LUFS)
        audio_filter = "highpass=f=100, afftdn=nr=15, loudnorm=I=-16:TP=-1.5:LRA=11"

        cmd = [
            FFMPEG_CMD, 
            '-i', file_path,
            '-af', audio_filter, # Apply our enhancement chain
            '-ac', '1',          # Mono
            '-b:a', '96k',       # 96kbps MP3
            output_path
        ]
        
        try:
            # We use a slight trick here to hide the massive ffmpeg log but show errors
            subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error converting {filename}: {e.stderr.decode()}")
            continue

    print("✅ All done! Enhanced files are in", OUTPUT_DIR)

if __name__ == "__main__":
    compress_and_enhance()