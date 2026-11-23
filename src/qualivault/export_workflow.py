# -*- coding: utf-8 -*-
"""Everything after transcription, without notebooks or Ollama:

1. `clean_transcripts`: rule-based cleanup + `text_corrections` from config.yml
   -> `<output_base_folder>/_cleaned_for_validation/*.csv` (rebuilt every run, so new
   corrections always apply; the folder name is kept for notebook 05 compatibility)
2. `build_compact_audio`: small mp3 per FLAC, with the same trim/denoise/normalize as
   transcription so timestamps line up -> `<obsidian_vault_folder>/<compact_audio_folder>/`
3. `export_transcripts_to_obsidian` on the cleaned CSVs

`run_export(project_root, config)` runs all three.
"""

from __future__ import annotations

import csv
import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("QualiVault")

CLEANED_SUBFOLDER = "_cleaned_for_validation"

KNOWN_SUBTITLE_PHRASES = [
    "danske tekster af nicolai winther",
    "danske tekster af jesper buhl scandinavian text service 2018",
]


def _resolve(project_root: Path, value: str) -> Path:
    return (Path(project_root) / Path(value).expanduser()).resolve()


def _strip_subtitles(text: str):
    removed = False
    for phrase in KNOWN_SUBTITLE_PHRASES:
        if phrase in text.casefold():
            text = re.sub(re.escape(phrase), " ", text, flags=re.IGNORECASE)
            removed = True
    return text, removed


def _apply_corrections(text: str, corrections: Dict[str, str]):
    hits = 0
    # Longest first so short keys don't rewrite parts of longer ones.
    for wrong, correct in sorted(corrections.items(), key=lambda kv: len(str(kv[0])), reverse=True):
        if wrong is None or correct is None or not str(wrong).strip():
            continue
        text, n = re.subn(rf"\b{re.escape(str(wrong).strip())}\b", str(correct).strip(), text, flags=re.IGNORECASE)
        hits += n
    return text, hits


def is_repetitive_pattern_hallucination(text: str) -> bool:
    """Whisper loop output like "hhhhh", "nanananana" or "og og og og"."""
    if not text:
        return False
    s = re.sub(r"[^A-Za-zÆØÅæøå]", "", text).casefold()
    if len(s) < 4:
        return False
    if re.fullmatch(r"(.)\1{3,}", s):
        return True
    for unit_len in range(1, 6):
        if len(s) >= unit_len * 4 and len(s) % unit_len == 0 and s[:unit_len] * (len(s) // unit_len) == s:
            return True
    tokens = re.findall(r"[A-Za-zÆØÅæøå]+", text.casefold())
    return len(tokens) >= 4 and len(set(tokens)) == 1


def _norm(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


def merge_same_speaker(rows: list, max_gap_seconds: float = 3.0, max_duration_seconds: float = 60.0) -> list:
    """Join consecutive rows of the same speaker (and same part audio) into one block.

    Diarization and the ASR windows cut one speaker's turn into many short rows, each with
    its own heading in Obsidian. Rows are joined when the pause between them is at most
    `max_gap_seconds` and the joined block stays within `max_duration_seconds` (so long
    monologues still get a timestamp now and then). Returns new rows (pandas Series).
    """
    out: list = []
    for r in rows:
        if out:
            prev = out[-1]
            same = (
                str(prev.get("Speaker", "")) == str(r.get("Speaker", ""))
                and str(prev.get("Audio", "")) == str(r.get("Audio", ""))
            )
            if (
                same
                and float(r["Start"]) - float(prev["End"]) <= max_gap_seconds
                and float(r["End"]) - float(prev["Start"]) <= max_duration_seconds
            ):
                prev["Text"] = f"{prev['Text']} {r['Text']}"
                prev["End"] = max(float(prev["End"]), float(r["End"]))
                continue
        out.append(r.copy())
    return out


def clean_transcripts(
    transcripts_dir: Path,
    corrections: Optional[Dict[str, str]] = None,
    merge: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write cleaned copies of every `*.csv` in `transcripts_dir` to `_cleaned_for_validation/`.

    Removes Whisper subtitle credits, repetitive-loop segments, consecutive duplicates and
    rows with impossible timestamps; applies `corrections` (wrong -> right, whole words,
    case-insensitive). A segment starting before the previous one ended is dropped only if
    its text repeats what was just said; otherwise its start is moved to the previous end
    (the text is kept). Timelines restart per part (`Audio` column) in joined interviews.

    `merge` (config `cleaning.merge_same_speaker`) joins consecutive rows of one speaker, see
    `merge_same_speaker`; keys `enabled` (default true), `max_gap_seconds`, `max_duration_seconds`.
    """
    transcripts_dir = Path(transcripts_dir)
    corrections = corrections or {}
    merge = dict(merge or {})
    merge_on = bool(merge.get("enabled", True))
    merge_gap = float(merge.get("max_gap_seconds", 3.0))
    merge_dur = float(merge.get("max_duration_seconds", 60.0))
    out_dir = transcripts_dir / CLEANED_SUBFOLDER
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.csv"):
        old.unlink()

    stats = dict(files=0, end_fixed=0, subtitles=0, repetitive=0, duplicates=0, bad_time=0, rollback_dropped=0, rollback_kept=0, corrections=0, merged=0)
    for csv_path in sorted(transcripts_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)
        if "Text" not in df.columns:
            df.to_csv(out_dir / csv_path.name, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
            continue
        rows, prev_key, cursor, cursor_audio, recent = [], None, None, object(), []
        for _, row in df.iterrows():
            try:
                start, end = float(row["Start"]), float(row["End"])
            except (TypeError, ValueError):
                stats["bad_time"] += 1
                continue
            if not (np.isfinite(start) and np.isfinite(end)):
                stats["bad_time"] += 1
                continue
            if end < start:  # text is real; only the end time is off
                end = start
                stats["end_fixed"] += 1
            audio = row.get("Audio", None)
            if audio != cursor_audio:  # new part of a joined interview: its own timeline
                cursor, cursor_audio, recent, prev_key = None, audio, [], None

            text = "" if pd.isna(row.get("Text")) else str(row.get("Text")).strip()
            if len(text) >= 2 and text[0] == text[-1] == '"':
                text = text[1:-1].strip()
            text, subtitle = _strip_subtitles(text)
            text, hits = _apply_corrections(text, corrections)
            stats["corrections"] += hits
            text = " ".join(text.split())
            if not text:
                stats["subtitles"] += int(subtitle)
                continue
            if is_repetitive_pattern_hallucination(text):
                stats["repetitive"] += 1
                continue

            if cursor is not None and start < cursor:
                n = _norm(text)
                if len(n.split()) >= 3 and f" {n} " in f" {' '.join(recent)} ":
                    stats["rollback_dropped"] += 1
                    continue
                start = cursor
                end = max(end, start)
                stats["rollback_kept"] += 1

            key = (str(row.get("Speaker", "")).strip().casefold(), text.casefold())
            if key == prev_key:
                stats["duplicates"] += 1
                continue
            prev_key = key

            r = row.copy()
            r["Text"], r["Start"], r["End"] = text, start, end
            rows.append(r)
            cursor = max(cursor if cursor is not None else end, end)
            recent = (recent + [_norm(text)])[-5:]

        if merge_on:
            before = len(rows)
            rows = merge_same_speaker(rows, merge_gap, merge_dur)
            stats["merged"] += before - len(rows)
        pd.DataFrame(rows, columns=df.columns).to_csv(out_dir / csv_path.name, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
        stats["files"] += 1

    logger.info(
        f"🧹 Cleaned {stats['files']} transcripts -> {out_dir} | removed: subtitles {stats['subtitles']}, "
        f"loops {stats['repetitive']}, duplicates {stats['duplicates']}, bad timestamps {stats['bad_time']}, "
        f"repeated overlaps {stats['rollback_dropped']} | kept with fixed times: overlaps {stats['rollback_kept']}, "
        f"end<start {stats['end_fixed']} | joined same-speaker rows {stats['merged']} | "
        f"corrections applied {stats['corrections']} ({len(corrections)} rules)"
    )
    return out_dir


def build_compact_audio(flac_dir: Path, audio_out_dir: Path, preprocessing: Dict[str, Any], bitrate: str = "96k", force: bool = False) -> int:
    """mp3 per FLAC for Obsidian, preprocessed like transcription so timestamps match.

    Skips mp3s newer than their FLAC unless `force`. Returns the number written.
    """
    import librosa
    import soundfile as sf

    from qualivault.audio import check_ffmpeg, trim_silence

    check_ffmpeg()
    audio_out_dir = Path(audio_out_dir)
    audio_out_dir.mkdir(parents=True, exist_ok=True)
    trim = bool(preprocessing.get("trim_silence", True))
    top_db = max(10.0, float(preprocessing.get("trim_silence_top_db", 30.0)))
    normalize = bool(preprocessing.get("normalize_audio", True))
    peak_target = max(0.01, float(preprocessing.get("normalize_target_peak", 0.95)))
    denoise = max(0.0, min(1.0, float(preprocessing.get("denoise_strength", 0.0))))

    written = 0
    for flac in sorted(Path(flac_dir).glob("*.flac")):
        out = audio_out_dir / f"{flac.stem}.mp3"
        if out.exists() and not force and out.stat().st_mtime >= flac.stat().st_mtime:
            continue
        try:
            y, sr = librosa.load(flac, sr=16000, mono=True)
            if trim:
                try:
                    y, _ = trim_silence(y, top_db=top_db)
                except Exception:
                    pass
            if denoise > 0:
                import noisereduce as nr

                y = nr.reduce_noise(y=y, sr=sr, prop_decrease=denoise, stationary=True)
            if normalize and len(y) and np.max(np.abs(y)) > 0:
                y = np.clip(y * (peak_target / float(np.max(np.abs(y)))), -1.0, 1.0)
            if not len(y):
                y, sr = librosa.load(flac, sr=16000, mono=True)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_wav = Path(tmp.name)
            try:
                sf.write(tmp_wav, y, sr)
                subprocess.run(
                    ["ffmpeg", "-y", "-i", str(tmp_wav), "-vn", "-c:a", "libmp3lame", "-b:a", bitrate, str(out)],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            finally:
                tmp_wav.unlink(missing_ok=True)
            written += 1
            logger.info(f"🎧 {out.name}")
        except Exception as e:
            logger.error(f"❌ Compact audio failed for {flac.name}: {e}")
    return written


def run_export(project_root: Path, config: Dict[str, Any], *, target_speaker: str = "SPEAKER_00", force_notes: bool = True) -> None:
    """Clean transcripts, build compact audio and (re)write the Obsidian notes."""
    from qualivault.obsidian_workflow import export_transcripts_to_obsidian

    project_root = Path(project_root)
    paths = config.get("paths", {})
    transcripts_dir = _resolve(project_root, paths["output_base_folder"])
    vault = _resolve(project_root, paths.get("obsidian_vault_folder", str(project_root / "obsidian_vault")))
    audio_sub = paths.get("compact_audio_folder", "compact_audio")
    # paths.obsidian_notes_subfolder: where notes go inside the vault (e.g. "01_Raw_Transcripts"
    # in a vault set up for coding). Audio links stay relative to the vault root either way.
    notes_dir = vault / paths.get("obsidian_notes_subfolder", "")

    cleaned = clean_transcripts(
        transcripts_dir,
        config.get("text_corrections") or {},
        (config.get("cleaning") or {}).get("merge_same_speaker"),
    )
    # paths.compact_audio_bitrate: mp3 bitrate; the audio is 16 kHz mono speech, so 32k is plenty.
    # paths.compact_audio_in_git: true when the vault repo stores the mp3s (e.g. via Git LFS);
    # otherwise the folder is added to the vault's .gitignore.
    n = build_compact_audio(
        _resolve(project_root, paths["flac_output_folder"]),
        vault / audio_sub,
        config.get("transcription", {}).get("preprocessing", {}),
        bitrate=str(paths.get("compact_audio_bitrate", "96k")),
    )
    logger.info(f"🎧 Compact audio: {n} mp3 written")

    gitignore = vault / ".gitignore"
    existing = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
    if not paths.get("compact_audio_in_git", False) and f"{audio_sub}/" not in existing.splitlines():
        gitignore.write_text(existing + ("" if not existing or existing.endswith("\n") else "\n") + f"{audio_sub}/\n", encoding="utf-8")

    questionnaire = paths.get("questionnaire_csv_path")
    export_transcripts_to_obsidian(
        transcripts_dir=cleaned,
        output_dir=notes_dir,
        obsidian_vault=vault,
        compact_audio_subfolder=audio_sub,
        target_speaker=target_speaker,
        question_sections={},
        questionnaire_csv_path=_resolve(project_root, questionnaire) if questionnaire else None,
        processing_recipe_path=project_root / "processing_recipe.yaml",
        validation_reports_dir=None,
        legacy_validation_report_path=None,
        force=force_notes,
    )
