import json
import os
import glob

# --- CONFIGURATION ---
# Use raw strings (r'...') to handle Windows backslashes correctly
INPUT_DIR_TRANSCRIPTS = r"E:\ledgendary\\interview_cleaned"
INPUT_DIR_AUDIO = r"E:\ledgendary\mp3 audio_small"
OUTPUT_DIR = r"E:\ledgendary\labek_studio_format"

# This matches the structure inside your Docker container
# If you mount C:\data\audio_interview to /label-studio/files, keep this empty/relative
# Label Studio Local Files usually expects just the filename if pointed to root
DOCKER_MOUNT_PREFIX = "" 

def find_audio_file(audio_dir, file_id):
    """
    Looks for interview_{id}.wav, .mp3, .m4a, etc.
    Returns the filename (e.g., 'interview_23.wav') or None.
    """
    search_pattern = os.path.join(audio_dir, f"interview_{file_id}.*")
    matches = glob.glob(search_pattern)
    if matches:
        # Return just the filename, not the full windows path
        return os.path.basename(matches[0])
    return None

def convert_folder():
    # 1. Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Get all JSON files from the Whisper folder
    transcript_files = glob.glob(os.path.join(INPUT_DIR_TRANSCRIPTS, "*.json"))
    
    print(f"Found {len(transcript_files)} transcripts to process...")

    for file_path in transcript_files:
        filename = os.path.basename(file_path) # e.g., "23.json"
        file_id = os.path.splitext(filename)[0] # e.g., "23"

        # 3. Find the matching audio file extension
        audio_filename = find_audio_file(INPUT_DIR_AUDIO, file_id)
        
        if not audio_filename:
            print(f"⚠️ Warning: No audio found for ID {file_id} (Skipping)")
            continue

        # 4. Load the Whisper Data
        with open(file_path, 'r', encoding='utf-8') as f:
            whisper_data = json.load(f)

        # 5. Build Label Studio Regions
        ls_predictions = []
        
        # WhisperX usually puts segments in a 'segments' list, 
        # but your sample showed a list of objects directly. 
        # We handle both cases:
        segments = whisper_data.get('segments', whisper_data) if isinstance(whisper_data, dict) else whisper_data

        for segment in segments:
            # Skip empty segments if any
            if 'start' not in segment: continue
            
            # Speaker Label Region
            speaker_result = {
                "id": f"spk_{file_id}_{segment['start']}",
                "from_name": "speaker",
                "to_name": "audio",
                "type": "labels",
                "value": {
                    "start": segment['start'],
                    "end": segment['end'],
                    "labels": [segment.get('speaker', 'Unknown')] 
                }
            }

            # Text Transcription Region
            text_result = {
                "id": f"txt_{file_id}_{segment['start']}",
                "from_name": "transcription",
                "to_name": "audio",
                "type": "textarea",
                "value": {
                    "start": segment['start'],
                    "end": segment['end'],
                    "text": [segment['text'].strip()]
                }
            }
            ls_predictions.extend([speaker_result, text_result])

        # 6. Construct the Label Studio Task Structure
        # Note: The "/data/local-files/?d=" prefix is required for LS to fetch 
        # the file from the mounted volume storage.
        final_output = [{
            "data": {
                # We add "files/" here so the full path becomes /label-studio/files/...
               "audio": f"/data/local-files/?d=audio/{audio_filename}"
            },
            "predictions": [{
                "model_version": "whisperx_v1",
                "score": 0.5,
                "result": ls_predictions
            }]
        }]

        # 7. Save to the Output Folder
        output_filename = f"import_{file_id}.json"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=2)
            
        print(f"✅ Converted: {filename} -> {output_filename}")

if __name__ == "__main__":
    convert_folder()