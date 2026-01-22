import json
import os
import glob

# --- CONFIGURATION ---
INPUT_DIR_TRANSCRIPTS = r"E:\ledgendary\transcriptions- - wisper"
INPUT_DIR_AUDIO = r"e:\ledgendary\mp3 audio_small" # Make sure this matches your actual audio folder
OUTPUT_DIR = r"e:\ledgendary\\interview_cleaned"
DOCKER_MOUNT_PREFIX = "audio"

# Granularity Setting:
SPLIT_THRESHOLD = 0.5 

def find_audio_file(audio_dir, file_id):
    search_pattern = os.path.join(audio_dir, f"interview_{file_id}.*")
    matches = glob.glob(search_pattern)
    return os.path.basename(matches[0]) if matches else None

def granular_segments(word_list):
    if not word_list:
        return []

    clean_segments = []
    current_segment = None
    last_known_speaker = "Unknown"

    for i, word_obj in enumerate(word_list):
        if 'start' not in word_obj: continue
        
        w_start = word_obj['start']
        w_end = word_obj['end']
        w_text = word_obj['word']
        
        # 1. Update Speaker Memory
        if 'speaker' in word_obj:
            last_known_speaker = word_obj['speaker']
        w_effective_speaker = last_known_speaker

        # Initialize
        if current_segment is None:
            current_segment = {
                "start": w_start,
                "end": w_end,
                "text": w_text,
                "speaker": w_effective_speaker
            }
            continue

        # --- LOGIC CHECKS ---
        gap = w_start - current_segment['end']
        speaker_changed = (w_effective_speaker != current_segment['speaker'])
        
        # SMART CHECK: Does the previous segment end with punctuation?
        # If NOT, it's an unfinished sentence. We should probably keep merging 
        # to fix the broken flow, unless the gap is huge.
        last_text = current_segment['text'].strip()
        sentence_ended = last_text.endswith(('.', '?', '!'))

        # DECISION: SPLIT OR MERGE?
        # We split ONLY if:
        # 1. The gap is huge (New topic/silence)
        # 2. OR: The speaker changed AND the previous sentence actually finished.
        should_split = (gap > SPLIT_THRESHOLD) or (speaker_changed and sentence_ended)

        if should_split:
            clean_segments.append(current_segment)
            current_segment = {
                "start": w_start,
                "end": w_end,
                "text": w_text,
                "speaker": w_effective_speaker
            }
        else:
            # MERGE (Glue this word onto the previous bubble)
            current_segment['end'] = w_end
            current_segment['text'] += " " + w_text

            # Note: If we merge despite a speaker change (because sentence didn't end),
            # the bubble keeps the ORIGINAL speaker. This effectively "corrects" 
            # the AI's wrong speaker tag for the second half of the sentence.

    if current_segment:
        clean_segments.append(current_segment)

    return clean_segments

def convert_folder():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    transcript_files = glob.glob(os.path.join(INPUT_DIR_TRANSCRIPTS, "*.json"))
    
    print(f"Processing {len(transcript_files)} transcripts...")

    for file_path in transcript_files:
        filename = os.path.basename(file_path)
        file_id = os.path.splitext(filename)[0]
        audio_filename = find_audio_file(INPUT_DIR_AUDIO, file_id)
        
        if not audio_filename:
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        # Handle list vs dict (The fix from before)
        if isinstance(raw_data, list):
            raw_segments = raw_data
        else:
            raw_segments = raw_data.get('segments', [])

        all_words = []
        for seg in raw_segments:
            if 'words' in seg:
                all_words.extend(seg['words'])
        
        # Fallback if no words found
        if not all_words and raw_segments:
             print(f"Warning: No word-level timestamps for {filename}. Using segments.")
             processed_segments = []
             for seg in raw_segments:
                 processed_segments.append({
                     "start": seg['start'],
                     "end": seg['end'],
                     "text": seg['text'].strip(),
                     "speaker": seg.get('speaker', 'Unknown')
                 })
        else:
             processed_segments = granular_segments(all_words)
        
        print(f"  {file_id}: Generated {len(processed_segments)} bubbles.")

        ls_predictions = []
        for segment in processed_segments:
            region_id = f"seg_{file_id}_{int(segment['start']*1000)}"
            
            ls_predictions.append({
                "id": region_id,
                "from_name": "speaker",
                "to_name": "audio",
                "type": "labels",
                "value": {
                    "start": segment['start'],
                    "end": segment['end'],
                    "labels": [segment['speaker']] 
                }
            })
            ls_predictions.append({
                "id": region_id,
                "from_name": "transcription",
                "to_name": "audio",
                "type": "textarea",
                "value": {
                    "start": segment['start'],
                    "end": segment['end'],
                    "text": [segment['text']]
                }
            })

        final_output = [{
            "data": {
                "audio": f"/data/local-files/?d={DOCKER_MOUNT_PREFIX}/{audio_filename}"
            },
            "predictions": [{
                "model_version": "whisper_filled_v2",
                "result": ls_predictions
            }]
        }]

        output_path = os.path.join(OUTPUT_DIR, f"import_{file_id}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=2)

if __name__ == "__main__":
    convert_folder()