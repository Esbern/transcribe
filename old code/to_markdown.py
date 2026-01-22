import json
import os
import glob
import datetime
import shutil

# --- CONFIGURATION ---
INPUT_DIR_JSON = r"e:\ledgendary\interview_cleaned"
INPUT_DIR_AUDIO = r"E:\ledgendary\mp3 audio_small"
OUTPUT_DIR = r"e:\ledgendary\interview_md"
TARGET_SPEAKER = "SPEAKER_00"

# --- THE QUESTIONNAIRE SCHEMA ---
# We define your questions here to keep the main code clean
QUESTION_SECTIONS = {
    "Section Q2: Demographics": [
        "Q2_1_person_interviewed", "Q2_2_birthyear", "Q2_3_residents", 
        "Q2_4_year_aqcuisition", "Q2_5_income_outside_farm"
    ],
    "Section Q3: Farm Structure": [
        "Q3_1_farmsize", "Q3_2_leased_area", "Q3_3_leased_out_area", 
        "Q3_4_ownership_of_other_farms", "Q3_5_started_farming", 
        "Q3_6_ownership_motivation", "Q3_7_organic", "Q3_8_local_farming_conditions"
    ],
    "Section Q4: Crops & Livestock": [
        "Q4_1_crop_types", "Q4_2_crop_reason", "Q4_3_livestock", 
        "Q4_4_livestock_type_count", "Q4_5_livestock_changes_past", 
        "Q4_6_livestock_changes_future"
    ],
    "Section Q5: Land Use Changes": [
        "Q5_1_arable_to_grass", "Q5_2_grass_to_arable", "Q5_3_arable_to_nature", 
        "Q5_4_nature_to_arable", "Q5_5_grass_to_nature", "Q5_6_nature_to_grass", 
        "Q5_7_other_changes", "Q5_8_crop_changes_past", "Q5_9_birds", 
        "Q5_10_insects", "Q5_11_other_animals"
    ],
    "Section Q6: Decision Making": [
        "Q6_1_planning_crop_rotation", "Q6_2_considerations_crops", 
        "Q6_3_crop_changes_future", "Q6_4_soil_variation_crops", 
        "Q6_5_soil_impact_cropchoice", "Q6_6_terrain_impact_cropchoice", 
        "Q6_7_drainage_impact_cropchoice", "Q6_8_changes_fieldplanning", 
        "Q6_9_good_agricultural_year"
    ],
    "Section Q7: Legumes": [
        "Q7_1_1_Legumes_crop_rotation", "Q7_1_2_Challenges_legumes", 
        "Q7_1_3_pros_cons_legumes", "Q7_1_4_perception_legumes", 
        "Q7_1_5_Perception_changes_legumes", "Q7_1_6_advantages_disadvantages_legumes", 
        "Q7_1_7_barriers_legumes", "Q7_1_8_benefits_legumes", "Q7_1_9_cooperation_sales",
        "Q7_2_1_experience_legumes", "Q7_2_2_perception_legumes", 
        "Q7_2_3_pros_cons_legumes", "Q7_2_4_advantages_disadvantages_legumes", 
        "Q7_2_5_expected_challenges_legumes", "Q7_2_6_incentives_legumes", 
        "Q7_2_7_benefits_legumes"
    ],
    "Section Q8: Field Conditions": [
        "Q8_1_best_worst_fields", "Q8_2_drainage_system", "Q8_3_waterlogged_fields", 
        "Q8_4_drought_fields", "Q8_5_plantprotection_fields", 
        "Q8_6_plantprotection_use_change", "Q8_7_production_conditions_fields", 
        "Q8_8_production_security_fields", "Q8_9_yield_data"
    ],
    "Section Q9: Society & Admin": [
        "Q9_1_farmtype", "Q9_2_admin_factors_cropchoice", "Q9_3_env_impact_choice", 
        "Q9_4_societal_expectations", "Q9_5_changes_societal_expectations", 
        "Q9_6_traditions_crops", "Q9_7_knowlegde_info_sources"
    ],
    "Section Q10: Landscape": [
        "Q10_1_landscape_role", "Q10_2_landscape_role_changes", "Q10_3_landscape_steward"
    ],
    "Section Q11: Future": [
        "Q11_1_future_farm_wishes", "Q11_2_future_legumes", 
        "Q11_3_biodiversity_considerations", "Q11_4_climate_adaptation", 
        "Q11_5_nitrogen_management", "Q11_6_retrospective_farm", 
        "Q11_7_Green_tripartite"
    ]
}

def format_timestamp(seconds):
    seconds = int(round(seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02}:{minutes:02}:{secs:02}"

def find_audio_file(base_id):
    patterns = [
        f"{base_id}.*",
        f"interview_{base_id}.*",
        f"Interview_{base_id}.*",
        f"interview {base_id}.*",
    ]
    for pattern in patterns:
        search_path = os.path.join(INPUT_DIR_AUDIO, pattern)
        matches = glob.glob(search_path)
        if matches:
            return matches[0]
    return None

def get_transcript_lines(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        results = data[0]['predictions'][0]['result']
    except (KeyError, IndexError, json.JSONDecodeError):
        return [], []

    regions = {}
    found_speakers = set()

    for item in results:
        r_id = item.get('id')
        if not r_id: continue

        if r_id not in regions:
            regions[r_id] = {'start': 0, 'speaker': 'Unknown', 'text': ''}

        if item.get('value', {}).get('start'):
             regions[r_id]['start'] = item['value']['start']

        if item.get('from_name') == 'speaker':
            spk = item['value']['labels'][0]
            regions[r_id]['speaker'] = spk
            found_speakers.add(spk)
        elif item.get('from_name') == 'transcription':
            regions[r_id]['text'] = item['value']['text'][0]

    return sorted(regions.values(), key=lambda x: x['start']), list(found_speakers)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    audio_output_dir = os.path.join(OUTPUT_DIR, "assets")
    if not os.path.exists(audio_output_dir):
        os.makedirs(audio_output_dir)

    all_files = glob.glob(os.path.join(INPUT_DIR_JSON, "import_*.json"))
    print(f"Found {len(all_files)} files.")

    today_str = datetime.date.today().strftime("%Y-%m-%d")

    for json_file in all_files:
        filename = os.path.basename(json_file)
        base_id = filename.replace("import_", "").replace(".json", "")
        readable_title = base_id.replace("interview_", "").replace("_", " ").title()
        if readable_title.isdigit():
            readable_title = f"Interview {readable_title}"

        output_path = os.path.join(OUTPUT_DIR, f"{base_id}.md")

        # --- AUDIO FINDER ---
        source_audio_path = find_audio_file(base_id)
        if not source_audio_path and "interview_" in base_id:
            source_audio_path = find_audio_file(base_id.replace("interview_", ""))

        audio_filename = None
        if source_audio_path:
            audio_filename = os.path.basename(source_audio_path)
            dest_audio_path = os.path.join(audio_output_dir, audio_filename)
            if not os.path.exists(dest_audio_path):
                shutil.copy2(source_audio_path, dest_audio_path)
        
        # --- GENERATE MARKDOWN ---
        lines, speakers = get_transcript_lines(json_file)
        if not lines: continue

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("---\n")
            f.write(f"title: \"{readable_title}\"\n")
            f.write(f"date: {today_str}\n")
            f.write("tags: [interview]\n")
            f.write("---\n\n")

            if audio_filename:
                f.write(f"# 🎧 Recording\n")
                f.write(f"![[assets/{audio_filename}]]\n\n")
                f.write("---\n\n")

            f.write(f"# Transcript\n\n")

            for line in lines:
                speaker = line['speaker']
                text = line['text']
                start_seconds = int(line['start']) 
                timestamp_text = format_timestamp(line['start'])
                
                if audio_filename:
                    time_link = f"[[assets/{audio_filename}#t={start_seconds}|{timestamp_text}]]"
                else:
                    time_link = f"`[{timestamp_text}]`"

                if speaker == TARGET_SPEAKER:
                    f.write(f"### **{speaker}** {time_link}\n")
                else:
                    f.write(f"### {speaker} {time_link}\n")
                
                f.write(f"{text}\n\n")

            # --- ADD QUESTIONNAIRE AT THE BOTTOM ---
            f.write("\n\n---\n")
            f.write("# 📊 Interview Data\n\n")
            f.write("> [!INFO]- Questionnaire\n")
            f.write("> Click the arrow above to expand and fill in the data.\n>\n")
            
            for section_name, questions in QUESTION_SECTIONS.items():
                f.write(f"> %% --- {section_name} --- %%\n")
                for q in questions:
                    # Format: > **Key**:: 
                    f.write(f"> **{q}**:: \n")
                f.write(">\n") # Spacer between sections

    print(f"\n✅ Finished! Questionnaires added to {len(all_files)} files.")

if __name__ == "__main__":
    main()