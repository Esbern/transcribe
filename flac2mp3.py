import os
import subprocess
import glob

# --- CONFIG ---
INPUT_DIR = r"E:\ledgendary\flac audio"
OUTPUT_DIR = r"E:\ledgendary\mp3 audio"
FFMPEG_CMD = "ffmpeg" # If ffmpeg is not in PATH, put full path to ffmpeg.exe here

def compress_files():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    files = glob.glob(os.path.join(INPUT_DIR, "*.flac"))
    print(f"Found {len(files)} files to compress...")

    for file_path in files:
        filename = os.path.basename(file_path)
        name_only = os.path.splitext(filename)[0]
        output_path = os.path.join(OUTPUT_DIR, f"{name_only}.mp3")

        if os.path.exists(output_path):
            print(f"Skipping {filename} (already exists)")
            continue

        print(f"Compressing {filename} -> mp3...")
        
        # Command explanation:
        # -ac 1: Convert to Mono (Stereo is wasteful for interviews)
        # -b:a 96k: Bitrate 96kbps (Plenty for voice)
        # -map_metadata 0: Keep metadata if any
        cmd = [
            FFMPEG_CMD, 
            '-i', file_path, 
            '-ac', '1',       
            '-b:a', '96k',    
            output_path
        ]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"Error converting {filename}: {e}")
            # Fallback: Check if ffmpeg is actually installed
            print("Make sure FFmpeg is installed and added to your system PATH.")
            return

    print("All done! Files are now in", OUTPUT_DIR)

if __name__ == "__main__":
    compress_files()