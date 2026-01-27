# -*- coding: utf-8 -*-

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from qualivault.audio import analyze_channel_separation, prepare_audio
from qualivault.core import Transcriber, get_device, process_item
from qualivault.recipe import expected_csv_name, expected_flac_name, interview_key, load_recipe, save_recipe
from qualivault.state import RunState

logger = logging.getLogger("QualiVault")


def _init_pyannote_pipeline(hf_token: Optional[str], device: str, cache_dir: Optional[Path] = None):
    if not hf_token:
        return None

    from pyannote.audio import Pipeline

    logger.info("⏳ Initializing Pyannote diarization...")
    kwargs = {"use_auth_token": hf_token}
    if cache_dir:
        kwargs["cache_dir"] = str(cache_dir)

    pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", **kwargs)
    pipe.to(torch.device(device))
    logger.info("✅ Pyannote diarization ready")
    return pipe


def _init_transcriber(config: Dict[str, Any], device: str, cache_dir: Optional[Path] = None):
    transcription_cfg = (config or {}).get("transcription", {})

    backend = transcription_cfg.get("backend", "auto")
    model_id = transcription_cfg.get("model_id", "openai/whisper-large-v3")

    # NOTE: Some backends are handled at the pipeline level (file-based, not chunk-based).
    if backend in {"whisperx", "faster_whisper", "mlx_whisper"}:
        return None

    # Default: transformers-based Whisper
    return Transcriber(device=device, model_id=model_id, cache_dir=cache_dir, config=transcription_cfg)


def _transcribe_with_whisperx(
    audio_path: Path,
    *,
    language: str,
    device: str,
    hf_token: Optional[str],
    model_id: str,
    output_csv: Path,
) -> Path:
    """Transcribe a single audio file with WhisperX and write a CSV.

    This is intentionally file-based (WhisperX expects file inputs and has its own VAD/alignment).
    """

    import importlib
    import pandas as pd

    whisperx = importlib.import_module("whisperx")

    compute_type = "float16" if device == "cuda" else "int8"
    model = whisperx.load_model(model_id, device, compute_type=compute_type)
    result = model.transcribe(str(audio_path), language=language)

    segments = result.get("segments", [])

    # Optional diarization (requires HF token)
    if hf_token:
        try:
            diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
            diarize_segments = diarize_model(str(audio_path))
            # Assign speakers where possible (works best if word-level timestamps exist)
            try:
                result = whisperx.assign_word_speakers(diarize_segments, result)
                segments = result.get("segments", segments)
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"⚠️ WhisperX diarization not available: {e}")

    rows = []
    for seg in segments:
        rows.append(
            {
                "Start": round(float(seg.get("start", 0.0)), 2),
                "End": round(float(seg.get("end", 0.0)), 2),
                "Speaker": seg.get("speaker", "UNKNOWN"),
                "Text": (seg.get("text") or "").strip(),
            }
        )

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return output_csv


def _transcribe_with_faster_whisper(
    audio_path: Path,
    *,
    language: str,
    device: str,
    model_id: str,
    output_csv: Path,
) -> Path:
    """Transcribe a single audio file with faster-whisper and write a CSV.

    Speaker attribution is not included (Speaker=UNKNOWN) unless you run a separate diarization step.
    """

    import importlib
    import pandas as pd

    faster_whisper = importlib.import_module("faster_whisper")
    WhisperModel = faster_whisper.WhisperModel

    fw_device = "cuda" if device == "cuda" else "cpu"
    compute_type = "float16" if fw_device == "cuda" else "int8"
    model = WhisperModel(model_id, device=fw_device, compute_type=compute_type)

    segments, _info = model.transcribe(str(audio_path), language=language)

    rows = []
    for seg in segments:
        rows.append(
            {
                "Start": round(float(seg.start), 2),
                "End": round(float(seg.end), 2),
                "Speaker": "UNKNOWN",
                "Text": (seg.text or "").strip(),
            }
        )

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return output_csv


def prepare_audio_and_analyze(
    project_root: Path,
    config: Dict[str, Any],
    recipe_path: Path,
    *,
    resume: bool = True,
    max_items: Optional[int] = None,
) -> None:
    """Convert raw audio to FLAC and compute analysis stats.

    Resume behavior is tracked in `.qualivault/run_state.json`.
    Analysis results (separation + speaker counts) are written back into the recipe because
    later transcription logic uses them.
    """

    project_root = Path(project_root)
    recipe_path = Path(recipe_path)

    flac_dir = (project_root / config["paths"]["flac_output_folder"]).resolve()
    flac_dir.mkdir(parents=True, exist_ok=True)

    models_dir = config.get("paths", {}).get("models_folder")
    cache_dir = (project_root / models_dir).resolve() if models_dir else None

    hf_token = config.get("hf_token")
    device = get_device()

    state = RunState(RunState.default_path(project_root)).load()

    diarization_pipe = None
    if hf_token:
        try:
            diarization_pipe = _init_pyannote_pipeline(hf_token, device=device, cache_dir=cache_dir)
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize pyannote (speaker counting disabled): {e}")
            diarization_pipe = None

    recipe = load_recipe(recipe_path)
    if max_items is not None:
        recipe = recipe[:max_items]

    if not recipe:
        logger.warning(f"No recipe items found at: {recipe_path}")
        return

    converted = analyzed = failed = skipped = 0

    for item in recipe:
        key = interview_key(item)
        output_name = expected_flac_name(item)
        out_path = flac_dir / output_name

        # Step: convert
        if resume and out_path.exists():
            # Migration-friendly: existing FLAC counts as success even if state is new.
            if state.get_step_status(key, "convert") != "success":
                state.mark_step_success(key, "convert", flac=str(out_path))
            skipped += 1
        elif state.should_skip(key, "convert", resume=resume) and out_path.exists():
            skipped += 1
        else:
            try:
                state.mark_step_running(key, "convert")

                if out_path.exists():
                    state.mark_step_success(key, "convert", flac=str(out_path))
                else:
                    prepare_audio(item["files"], out_path)
                    state.mark_step_success(key, "convert", flac=str(out_path))
                converted += 1
            except Exception as e:
                failed += 1
                state.mark_step_failed(key, "convert", e, flac=str(out_path))
                logger.error(f"❌ Convert failed for {key}: {e}")
                continue

        # Step: analyze
        # Treat as done if recipe already contains analysis stats.
        already_has_analysis = any(k in item for k in ["separation_status", "speakers_left", "speakers_right", "separation_db"])
        if resume and already_has_analysis:
            # Migration-friendly: if analysis already exists in recipe, mark success and skip.
            if state.get_step_status(key, "analyze") != "success":
                state.mark_step_success(key, "analyze")
            skipped += 1
            continue
        if state.should_skip(key, "analyze", resume=resume) and already_has_analysis:
            skipped += 1
            continue

        try:
            state.mark_step_running(key, "analyze")
            stats = analyze_channel_separation(out_path, diarization_pipe)
            item.update(stats)
            item["analysis_completed_at"] = datetime.now().isoformat()
            save_recipe(recipe_path, recipe)
            state.mark_step_success(key, "analyze", **{k: stats.get(k) for k in stats.keys()})
            analyzed += 1
        except Exception as e:
            failed += 1
            state.mark_step_failed(key, "analyze", e, flac=str(out_path))
            logger.error(f"❌ Analyze failed for {key}: {e}")
            continue

    logger.info("=" * 70)
    logger.info(f"✅ Prepare complete. Converted: {converted}, Analyzed: {analyzed}, Skipped: {skipped}, Failed: {failed}")
    logger.info(f"📍 State file: {state.state_path}")


def transcribe_recipe(
    project_root: Path,
    config: Dict[str, Any],
    recipe_path: Path,
    *,
    resume: bool = True,
    max_items: Optional[int] = None,
) -> None:
    """Run diarization + transcription and write per-interview CSV transcripts.

    This uses `.qualivault/run_state.json` for resumability.
    """

    project_root = Path(project_root)
    recipe_path = Path(recipe_path)

    flac_dir = (project_root / config["paths"]["flac_output_folder"]).resolve()
    output_dir = (project_root / config["paths"]["output_base_folder"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    models_dir = config.get("paths", {}).get("models_folder")
    cache_dir = (project_root / models_dir).resolve() if models_dir else None

    hf_token = config.get("hf_token")
    device = get_device()

    state = RunState(RunState.default_path(project_root)).load()

    transcription_cfg = (config or {}).get("transcription", {})
    backend = transcription_cfg.get("backend", "auto")
    model_id = transcription_cfg.get("model_id", "openai/whisper-large-v3")
    language = transcription_cfg.get("language", "da")

    # Backend auto-selection
    if backend == "auto":
        # CUDA: prefer WhisperX (fast, strong) if installed; otherwise transformers.
        # Non-CUDA: default to transformers (MPS/CPU).
        backend = "whisperx" if device == "cuda" else "transformers"

    diarization_pipe = None
    transcriber = None
    if backend == "transformers":
        diarization_pipe = _init_pyannote_pipeline(hf_token, device=device, cache_dir=cache_dir)
        transcriber = _init_transcriber(config, device=device, cache_dir=cache_dir)

    recipe = load_recipe(recipe_path)
    if max_items is not None:
        recipe = recipe[:max_items]

    done = failed = skipped = 0

    for item in recipe:
        key = interview_key(item)
        flac_name = expected_flac_name(item)
        csv_name = expected_csv_name(item)

        flac_path = flac_dir / flac_name
        csv_path = output_dir / csv_name

        if resume and csv_path.exists():
            # Migration-friendly: existing transcript counts as success even if state is new.
            if state.get_step_status(key, "transcribe") != "success":
                state.mark_step_success(key, "transcribe", csv=str(csv_path))
            skipped += 1
            continue
        if csv_path.exists() and state.should_skip(key, "transcribe", resume=resume):
            skipped += 1
            continue

        if not flac_path.exists():
            failed += 1
            state.mark_step_failed(key, "transcribe", FileNotFoundError(f"Missing FLAC: {flac_path}"), flac=str(flac_path))
            logger.error(f"❌ Missing FLAC for {key}: {flac_path}")
            continue

        try:
            state.mark_step_running(key, "transcribe")

            if backend == "whisperx":
                out_csv = _transcribe_with_whisperx(
                    flac_path,
                    language=language,
                    device=device,
                    hf_token=hf_token,
                    model_id=model_id,
                    output_csv=csv_path,
                )
            elif backend == "faster_whisper":
                out_csv = _transcribe_with_faster_whisper(
                    flac_path,
                    language=language,
                    device=device,
                    model_id=model_id,
                    output_csv=csv_path,
                )
            else:
                # process_item expects analysis fields on the item dict (speakers/separation).
                item_local = dict(item)
                item_local["output_name"] = flac_name

                out_csv = process_item(
                    item_local,
                    flac_dir=flac_dir,
                    output_dir=output_dir,
                    transcriber=transcriber,
                    diarization_pipeline=diarization_pipe,
                    language=language,
                    prompt_ids=None,
                )

            if not out_csv:
                raise RuntimeError("transcription produced no output")

            state.mark_step_success(key, "transcribe", csv=str(out_csv), backend=backend, model_id=model_id)
            done += 1
            logger.info(f"✅ Transcribed {key} -> {out_csv}")
        except Exception as e:
            failed += 1
            state.mark_step_failed(key, "transcribe", e, flac=str(flac_path), csv=str(csv_path), backend=backend)
            logger.error(f"❌ Transcribe failed for {key}: {e}")
            continue

    logger.info("=" * 70)
    logger.info(f"✅ Transcription complete. Done: {done}, Skipped: {skipped}, Failed: {failed}")
    logger.info(f"📍 State file: {state.state_path}")
