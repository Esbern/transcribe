micromamba activate agri_transcribe
pip uninstall -y torch torchaudio torchvision whisperx
pip install whisperx --extra-index-url https://download.pytorch.org/whl/cu128
pip install whisperx --extra-index-url https://download.pytorch.org/whl/cu130
