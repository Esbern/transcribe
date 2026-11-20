# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

QualiVault: a local-first, GDPR-compliant pipeline that transcribes and diarizes qualitative interviews (mainly Danish) and exports an Obsidian vault. The installable package is `src/qualivault/`. You run it through Jupyter notebooks that sit inside each project folder.

## Setup and commands

```bash
# Pick the env for your hardware: environment_cuda.yml, environment_mac.yml or environment_cpu.yml (env name: qualivault)
conda env create -f environment_cuda.yml && conda activate qualivault
pip uninstall -y torchcodec      # CUDA only: a broken torchcodec crashes the transformers ASR pipeline; nothing needs it
pip install whisperx             # optional, CUDA
pip install -e .

python -m qualivault.cli --init <project>      # scaffold projects/<project>/ (config.yml + notebooks from templates/)
python -m qualivault.cli --notebook <project>  # jupyter lab in projects/<project>/
ollama serve                                   # needed for notebook 04 (LLM transcript validation)
```

There are no tests, linter or build step. To check a change, import the module or run the relevant notebook.

## Architecture

**Projects are separate from the package.** `projects/` is gitignored in this repo. Each project (for example `projects/Legundary/`) has its own git repo, a `config.yml`, a git-ignored `.env` with `HF_TOKEN` (`core.get_hf_token` falls back to it when `config.yml`'s `hf_token` is empty), a `processing_recipe.yaml` and its own copies of the notebooks. `--init` substitutes `{{PROJECT_NAME}}` in the template notebooks and `config.yml`. Notebooks find the repo root with `paths.find_workspace_root()`, which walks up to `pyproject.toml`. If you edit `src/qualivault/templates/*.ipynb`, existing projects do not pick up the change, because their notebooks are copies.

**Notebook pipeline (templates 00–05):**
1. `00_Setup_and_Scan`: regex-scans `paths.org_audio_folder` (`patterns.folder_id_regex` / `target_audio_regex`) and writes `processing_recipe.yaml` with `recipe.generate_recipe` (one item per interview: `id`, `speakers` (filled in by hand), `files` relative to `org_audio_folder`, `output_name: Interview_<id>.flac`). Expected folder layout is `<name>_<id>`. A folder with several audio files becomes one recipe item per file (`recipe.split_into_parts`: `id: 20-1`, `interview_id`, `part`, `output_name: Interview_20_part1.flac`), so each part is calibrated and diarized separately; part CSVs go to `<output_base_folder>/_parts/` and `parts.merge_part_transcripts` (run at the end of `transcribe_recipe`) joins them into `Interview_<id>.csv` with `Part`/`Audio`/`PartSpeaker` columns, matching speakers across parts by pyannote embedding. The Obsidian export writes one section per part. `recipe.check_recipe` catches duplicate ids/output names, one id spanning several folders, and missing files; `prepare_audio_and_analyze` and `transcribe_recipe` refuse to run when it reports errors.
2. `01_Prepare_Audio` → `pipeline.prepare_audio_and_analyze`: merges the source files into FLAC (`audio.prepare_audio`), then `audio.analyze_channel_separation` writes `separation_status`, `separation_db`, `speakers_left` and `speakers_right` **back into the recipe**. Transcription relies on these fields later.
3. `02_Transcribe` → `pipeline.transcribe_recipe`: writes one CSV per interview to `paths.output_base_folder`.
4. `03_Export_Obsidian` → `core.export_recipe` (older, simple CSV→MD export).
5. `04_Validate_Transcripts` → `validation.OllamaValidator` / `validation_workflow`: writes per-interview JSON to `validation_reports/`. Config `text_corrections` is applied here, in the notebook.
6. `05_Export_To_Obsidian` → `obsidian_workflow.export_transcripts_to_obsidian`: the full export, with timestamps, audio links, questionnaire prefill and validation notes merged in.

**Two status systems.** `pipeline.py` (the current path) tracks progress per interview and per step in `<project>/.qualivault/run_state.json` (`state.RunState`). It skips a step when its output file already exists (FLAC or CSV), so delete the output to force a rerun. The `*_status` fields in `processing_recipe.yaml` (`convert_status`, `transcribe_status`, …) come from the older `core.process_recipe` loop and are not kept in sync, so `transcribe_status: pending` in the recipe does not mean the interview is untranscribed.

**Transcription backends** (`transcription.backend`): `auto` resolves to `whisperx` on CUDA and `transformers` everywhere else. `whisperx` and `faster_whisper` work on whole files (`pipeline._transcribe_with_*`). `transformers` is the main, most tunable path: `core.Transcriber` with the HF Whisper pipeline, plus pyannote `speaker-diarization-3.1`, plus `core.process_item`. `process_item` does the speaker work: speaker count from the recipe (`diarization.speaker_count_source`), stereo L/R dominance as a fallback hint, splitting chunks by diarization turns, segment merge/duration limits, and adaptive tightening of those limits for interviews with `Poor` separation.

**Config layering.** `transcribe_recipe` deep-merges `transcription.interview_overrides[<key>]` over `transcription`. The key can be `Interview_<id>`, the FLAC stem or the bare id. Models are cached in `paths.models_folder` (relative to the project; `core.setup_environment` also sets `HF_HOME` to it). Paths in config are resolved relative to the project root, with `~` expanded.

`core.py` is the oldest module and mixes the legacy entry points (`process_recipe`, `export_recipe`, `to_markdown`) with the core transcription logic (`Transcriber`, `process_item`) that `pipeline.py` depends on. New orchestration goes in `pipeline.py` / `*_workflow.py`.

## Environment notes

- Development runs in WSL2. The source audio is on a Windows OneDrive mount (`/mnt/c/...`), and some old recipes contain Windows-absolute paths, which `_resolve_recipe_files` passes through unchanged.
- `context.md` is out of date: it says "no WhisperX", but WhisperX is now a supported backend.
