# transcribe
audio cleanup and ai transcription
the enviorment used strict channel priority so run

micromamba config set channel_priority strict

to create the enviorment run 

micromamba create -f .\environment.yml
due to cpmpatability problmens with wisperx wisperx and pytorch must both be installed using pip


pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

pip install whisperx

pip install -U label-studio

to run label studio run the following from command line
label-studio