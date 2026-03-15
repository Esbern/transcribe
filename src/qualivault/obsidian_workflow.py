# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import re
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml


def format_ts(seconds: float) -> str:
    seconds = int(round(seconds))
    h, m = divmod(seconds, 3600)
    m, s = divmod(m, 60)
    return f"{h:02}:{m:02}:{s:02}"


def find_audio_file(base_id: str, audio_dir: Path) -> Optional[str]:
    """Find an audio file matching transcript basename.

    Flexible matching: Interview_1 matches interview_1.mp3, interview 1.mp3, etc.
    """

    audio_dir = Path(audio_dir)
    if not audio_dir.exists():
        return None

    all_audio = [f for f in audio_dir.glob("*") if f.is_file()]
    base_lower = base_id.lower()

    num_match = re.search(r"\d+", base_id)
    numeric_id = num_match.group(0) if num_match else None

    for audio in all_audio:
        if audio.stem.lower() == base_lower:
            return audio.name

    if numeric_id:
        base_prefix = base_lower.split("_")[0] if "_" in base_lower else ""
        for audio in all_audio:
            audio_lower = audio.stem.lower()
            if numeric_id in audio_lower:
                if any(prefix and prefix in audio_lower for prefix in ["interview", base_prefix]):
                    return audio.name

    for audio in all_audio:
        stem_lower = audio.stem.lower()
        if base_lower in stem_lower or stem_lower in base_lower:
            return audio.name

    if numeric_id:
        for audio in all_audio:
            if numeric_id in audio.stem:
                return audio.name

    return None


def _index_validation_report(index: Dict[str, Dict[str, Any]], report: Dict[str, Any]) -> None:
    csv_file = report.get("csv_file")
    if csv_file:
        csv_path = Path(csv_file)
        index[csv_path.name] = report
        index[csv_path.stem] = report


def load_validation_index(
    *,
    validation_reports_dir: Optional[Path] = None,
    legacy_validation_report_path: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """Load validation reports into a lookup map.

    Supports:
    - New format: per-interview JSON files in a directory: `*_validation.json`
    - Legacy format: a single `validation_report.json` with a `results` array

    Returns a dict keyed by csv filename and stem.
    """

    index: Dict[str, Dict[str, Any]] = {}

    if validation_reports_dir is not None:
        validation_reports_dir = Path(validation_reports_dir)
        if validation_reports_dir.exists():
            for report_file in sorted(validation_reports_dir.glob("*_validation.json")):
                try:
                    with open(report_file, "r", encoding="utf-8") as f:
                        report = json.load(f)
                    if isinstance(report, dict):
                        _index_validation_report(index, report)
                except Exception:
                    continue

    if legacy_validation_report_path is not None:
        legacy_validation_report_path = Path(legacy_validation_report_path)
        if legacy_validation_report_path.exists():
            try:
                with open(legacy_validation_report_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    for item in data.get("results", []):
                        if isinstance(item, dict):
                            _index_validation_report(index, item)
            except Exception:
                pass

    return index


def debug_audio_matching(
    *,
    transcripts_dir: Path,
    audio_vault_dir: Path,
    sample_files: int = 3,
) -> None:
    transcripts_dir = Path(transcripts_dir)
    audio_vault_dir = Path(audio_vault_dir)

    print("\n🔍 DETAILED AUDIO FILE MATCHING DEBUG\n")
    print(f"Audio vault: {audio_vault_dir}")
    print(f"Exists: {audio_vault_dir.exists()}\n")

    if not audio_vault_dir.exists():
        print("❌ Audio vault directory does not exist!")
        return

    all_audio = sorted([f for f in audio_vault_dir.glob("*") if f.is_file()])
    print(f"Audio files found ({len(all_audio)}):")
    for audio in all_audio[:20]:
        print(f"  • {audio.name} (stem: {audio.stem})")

    csv_files_sample = sorted(transcripts_dir.glob("*.csv"))[:sample_files]
    print(f"\n\nTesting matching for first {len(csv_files_sample)} CSVs:\n")

    for csv_path in csv_files_sample:
        base = csv_path.stem
        print(f"CSV: {csv_path.name}")
        print(f"  Stem: '{base}'")
        match = find_audio_file(base, audio_vault_dir)
        if match:
            print(f"    ✅ Matched: {match}")
        else:
            print("    ❌ NO MATCH FOUND")
        print()


def _read_csv_with_encoding_fallback(path: Path, **kwargs: Any) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError as e:
            last_err = e
            continue
    if last_err is not None:
        raise last_err
    return pd.read_csv(path, **kwargs)


def _extract_numeric_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    m = re.search(r"\d+", s)
    if not m:
        return None
    return str(int(m.group(0)))


def _load_recipe_speakers_map(processing_recipe_path: Optional[Path]) -> Dict[str, str]:
    if not processing_recipe_path:
        return {}

    path = Path(processing_recipe_path)
    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            recipe = yaml.safe_load(f)
    except Exception:
        return {}

    if not isinstance(recipe, list):
        return {}

    out: Dict[str, str] = {}
    for item in recipe:
        if not isinstance(item, dict):
            continue
        rid = _extract_numeric_id(item.get("id"))
        if rid is None:
            continue
        speakers = item.get("speakers")
        if speakers is None:
            continue
        out[rid] = str(speakers)
    return out


def _load_questionnaire_structure(
    questionnaire_csv_path: Optional[Path],
) -> Optional[Dict[str, Any]]:
    if not questionnaire_csv_path:
        return None

    path = Path(questionnaire_csv_path)
    if not path.exists():
        return None

    try:
        df = _read_csv_with_encoding_fallback(
            path,
            header=None,
            dtype=str,
            keep_default_na=False,
            engine="python",
        )
    except Exception:
        return None

    if df.empty:
        return None

    lb_row_idx: Optional[int] = None
    for idx in range(len(df)):
        first_col = str(df.iat[idx, 0]).strip().casefold() if df.shape[1] > 0 else ""
        if first_col in {"lb_id", "lb id"}:
            lb_row_idx = idx
            break

    id_to_col: Dict[str, int] = {}
    if lb_row_idx is not None:
        for col in range(2, df.shape[1]):
            interview_id = _extract_numeric_id(df.iat[lb_row_idx, col])
            if interview_id:
                id_to_col[interview_id] = col

    sections: List[Tuple[str, List[str]]] = []
    current_section = "Questionnaire"
    current_questions: List[str] = []
    response_type_by_key: Dict[str, str] = {}
    row_index_by_key: Dict[str, int] = {}

    for idx in range(len(df)):
        key = str(df.iat[idx, 0]).strip() if df.shape[1] > 0 else ""
        resp = str(df.iat[idx, 1]).strip() if df.shape[1] > 1 else ""

        if not key:
            continue
        if key.casefold() in {"variable", "lb_id", "lb id"}:
            continue

        if key.startswith("Section "):
            if current_questions:
                sections.append((current_section, current_questions))
            current_section = key
            current_questions = []
            continue

        # Keep questionnaire rows that look like question variables (Q...) or free rows.
        current_questions.append(key)
        response_type_by_key[key] = resp
        row_index_by_key[key] = idx

    if current_questions:
        sections.append((current_section, current_questions))

    return {
        "dataframe": df,
        "id_to_col": id_to_col,
        "sections": sections,
        "response_type_by_key": response_type_by_key,
        "row_index_by_key": row_index_by_key,
    }


def _get_prefilled_answer(
    questionnaire: Optional[Dict[str, Any]],
    question_key: str,
    interview_id: Optional[str],
) -> str:
    if not questionnaire or interview_id is None:
        return ""

    row_index_by_key: Dict[str, int] = questionnaire.get("row_index_by_key", {})
    id_to_col: Dict[str, int] = questionnaire.get("id_to_col", {})
    df: pd.DataFrame = questionnaire.get("dataframe")

    row_idx = row_index_by_key.get(question_key)
    col_idx = id_to_col.get(interview_id)
    if row_idx is None or col_idx is None:
        return ""

    try:
        value = str(df.iat[row_idx, col_idx]).strip()
        return "" if value.lower() == "nan" else value
    except Exception:
        return ""


def _has_prefill_column_for_interview(
    questionnaire: Optional[Dict[str, Any]],
    interview_id: Optional[str],
) -> bool:
    if not questionnaire or interview_id is None:
        return False
    id_to_col: Dict[str, int] = questionnaire.get("id_to_col", {})
    return interview_id in id_to_col


def export_transcripts_to_obsidian(
    *,
    transcripts_dir: Path,
    output_dir: Path,
    obsidian_vault: Path,
    compact_audio_subfolder: str,
    target_speaker: str,
    question_sections: Optional[Dict[str, List[str]]] = None,
    questionnaire_csv_path: Optional[Path] = None,
    processing_recipe_path: Optional[Path] = None,
    validation_reports_dir: Optional[Path] = None,
    legacy_validation_report_path: Optional[Path] = None,
    force: bool = False,
) -> Tuple[int, int]:
    """Export CSV transcripts to Obsidian Markdown notes.

    Returns (exported_count, skipped_count).
    """

    transcripts_dir = Path(transcripts_dir)
    output_dir = Path(output_dir)
    obsidian_vault = Path(obsidian_vault)

    audio_vault_dir = obsidian_vault / compact_audio_subfolder

    output_dir.mkdir(parents=True, exist_ok=True)
    audio_vault_dir.mkdir(parents=True, exist_ok=True)

    validation_index = load_validation_index(
        validation_reports_dir=validation_reports_dir,
        legacy_validation_report_path=legacy_validation_report_path,
    )

    questionnaire = _load_questionnaire_structure(questionnaire_csv_path)
    speakers_by_interview_id = _load_recipe_speakers_map(processing_recipe_path)

    csv_files = sorted(transcripts_dir.glob("*.csv"))
    print(f"\n📝 EXPORTING {len(csv_files)} TRANSCRIPTS\n")

    exported = 0
    skipped = 0

    for csv_path in csv_files:
        base = csv_path.stem
        title = base.replace("_", " ").title()
        interview_id = _extract_numeric_id(base)
        out_path = output_dir / f"{base}.md"

        if out_path.exists() and not force:
            print(f"⏭️  {out_path.name} (already exists)")
            skipped += 1
            continue

        df = pd.read_csv(csv_path)

        audio_name = find_audio_file(base, audio_vault_dir)
        audio_rel_path = f"{compact_audio_subfolder}/{audio_name}" if audio_name else None
        # Keep timestamp links clickable even when matching fails by using a stable fallback filename.
        audio_rel_path_for_links = audio_rel_path or f"{compact_audio_subfolder}/{base}.mp3"

        validation_data = (
            validation_index.get(csv_path.name)
            or validation_index.get(csv_path.stem)
            or {}
        )
        flagged_segments: Dict[int, Dict[str, Any]] = {}
        for seg in validation_data.get("flagged_segments", []):
            if not isinstance(seg, dict):
                continue
            seg_index = seg.get("segment_index")
            if seg_index is None:
                continue
            try:
                flagged_segments[int(seg_index)] = seg
            except Exception:
                continue

        lines: List[str] = []
        lines.append("---")
        lines.append(f'title: "{title}"')
        if interview_id is not None:
            lines.append(f"LB_id: {interview_id}")
        lines.append(f"date: {datetime.date.today().isoformat()}")
        lines.append("tags: [interview]")
        lines.append("---\n")

        if audio_rel_path:
            lines.append("# 🎧 Recording")
            lines.append(f"![[{audio_rel_path}]]\n")
            lines.append("---\n")

        lines.append("# Transcript\n")

        for idx, row in df.iterrows():
            speaker = str(row.get("Speaker", "Unknown"))
            text = str(row.get("Text", ""))
            start = float(row.get("Start", 0))
            ts_label = format_ts(start)
            ts_int = int(round(start))

            ts_link = f"[{ts_label}]({audio_rel_path_for_links}#t={ts_int})"

            heading = (
                f"### **{speaker}** {ts_link}"
                if speaker == target_speaker
                else f"### {speaker} {ts_link}"
            )
            lines.append(heading)
            lines.append(text)

            idx_int: Optional[int]
            try:
                idx_int = int(str(idx))
            except Exception:
                idx_int = None

            if idx_int is not None and idx_int in flagged_segments:
                val_seg = flagged_segments[idx_int]
                lines.append("")
                lines.append("> [!warning]- _Validation Issues (AI-detected, may contain false positives)_")

                issues = val_seg.get("issues", [])
                if issues:
                    for issue in issues:
                        if isinstance(issue, dict):
                            issue_text = issue.get("description", str(issue))
                        else:
                            issue_text = str(issue)
                        lines.append(f"> - *#Issue:* _{issue_text}_")

                suggestions = val_seg.get("suggestions")
                if suggestions:
                    lines.append(f"> - *#Suggestions:* _{suggestions}_")

                confidence = val_seg.get("confidence")
                if confidence is not None:
                    try:
                        conf_float = float(confidence)
                        lines.append(f"> - *Confidence:* {conf_float:.0%}")
                    except Exception:
                        pass

            lines.append("")

        lines.append("---")
        lines.append("# 📊 Interview Data")
        lines.append("> [!INFO]- Questionnaire")
        lines.append("> Click to expand. Prefilled values are loaded when available; blank fields are expected for interviews completed directly in Obsidian.")
        lines.append(">")

        if interview_id is not None:
            lines.append(f"> **LB_id**:: {interview_id}")
        else:
            lines.append("> **LB_id**:: ")
        lines.append(">")

        if questionnaire and questionnaire.get("sections"):
            response_type_by_key: Dict[str, str] = questionnaire.get("response_type_by_key", {})
            sections: List[Tuple[str, List[str]]] = questionnaire.get("sections", [])
            has_prefill_column = _has_prefill_column_for_interview(questionnaire, interview_id)

            if interview_id is not None and not has_prefill_column:
                lines.append("> _No prefilled questionnaire column found for this LB_id in Questionnaire.csv. This is expected when the interview is completed directly in Obsidian._")
                lines.append(">")

            for section, questions in sections:
                lines.append(f"> %% {section} %%")
                for q in questions:
                    response_type = response_type_by_key.get(q, "")
                    prefilled = _get_prefilled_answer(questionnaire, q, interview_id)

                    if q.strip() == "Q1_2_number_of_participants" and interview_id is not None:
                        prefilled = speakers_by_interview_id.get(interview_id, prefilled)

                    helper = f" ({response_type})" if response_type else ""
                    lines.append(f"> **{q}**{helper}:: {prefilled}")
                lines.append(">")
        else:
            for section, questions in (question_sections or {}).items():
                lines.append(f"> %% {section} %%")
                for q in questions:
                    prefilled = ""
                    if q.strip() == "Q1_2_number_of_participants" and interview_id is not None:
                        prefilled = speakers_by_interview_id.get(interview_id, "")
                    lines.append(f"> **{q}**:: {prefilled}")
                lines.append(">")

        out_path.write_text("\n".join(lines), encoding="utf-8")

        audio_status = "🎧" if audio_rel_path else "⚠️ "
        val_count = len(flagged_segments)
        val_status = f"✅({val_count})" if val_count > 0 else "  "
        print(f"{audio_status} {val_status} {out_path.name}")

        exported += 1

    print(f"\n✨ Exported {exported}/{len(csv_files)} files to {output_dir} (skipped {skipped})")
    return exported, skipped
