import os
import re
import yaml

def load_existing_recipe(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return yaml.safe_load(f) or []
    return []

def scan():
    # Load config
    with open("./scanner.yml", 'r') as f:
        config = yaml.safe_load(f)

    recipe = load_existing_recipe(config['paths']['recipe_file'])
    existing_ids = {item['id'] for item in recipe}
    
    new_entries = []
    folder_re = re.compile(config['patterns']['folder_id_regex'])
    file_re = re.compile(config['patterns']['file_regex'], re.IGNORECASE)

    for root, dirs, files in os.walk(config['paths']['master_folder']):
        match = folder_re.search(os.path.basename(root))
        if match:
            interview_id = match.group('id')
            
            # Skip if already in recipe
            if interview_id in existing_ids:
                continue
            
            # Find matching audio files in this folder
            matching_files = [
                os.path.join(root, f) for f in files if file_re.match(f)
            ]

            if matching_files:
                new_entries.append({
                    'id': interview_id,
                    'status': 'pending',
                    'output_name': f"interview_{interview_id}.flac",
                    'files': sorted(matching_files) # Default alphabetical order
                })

    # Append new finds and save
    recipe.extend(new_entries)
    with open(config['paths']['recipe_file'], 'w') as f:
        yaml.dump(recipe, f, sort_keys=False)

if __name__ == "__main__":
    scan()