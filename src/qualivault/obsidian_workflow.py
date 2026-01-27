# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import re
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


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


def export_transcripts_to_obsidian(
    *,
    transcripts_dir: Path,
    output_dir: Path,
    obsidian_vault: Path,
    compact_audio_subfolder: str,
    target_speaker: str,
    question_sections: Dict[str, List[str]],
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

    csv_files = sorted(transcripts_dir.glob("*.csv"))
    print(f"\n📝 EXPORTING {len(csv_files)} TRANSCRIPTS\n")

    exported = 0
    skipped = 0

    for csv_path in csv_files:
        base = csv_path.stem
        title = base.replace("_", " ").title()
        out_path = output_dir / f"{base}.md"

        if out_path.exists() and not force:
            print(f"⏭️  {out_path.name} (already exists)")
            skipped += 1
            continue

        df = pd.read_csv(csv_path)

        audio_name = find_audio_file(base, audio_vault_dir)
        audio_rel_path = f"{compact_audio_subfolder}/{audio_name}" if audio_name else None

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

            if audio_rel_path:
                ts_link = f"[[{audio_rel_path}#t={ts_int}|{ts_label}]]"
            else:
                ts_link = f"`[{ts_label}]`"

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
        lines.append("> Click to expand and fill in answers.")
        lines.append(">")

        for section, questions in question_sections.items():
            lines.append(f"> %% {section} %%")
            for q in questions:
                lines.append(f"> **{q}**:: ")
            lines.append(">")

        out_path.write_text("\n".join(lines), encoding="utf-8")

        audio_status = "🎧" if audio_rel_path else "⚠️ "
        val_count = len(flagged_segments)
        val_status = f"✅({val_count})" if val_count > 0 else "  "
        print(f"{audio_status} {val_status} {out_path.name}")

        exported += 1

    print(f"\n✨ Exported {exported}/{len(csv_files)} files to {output_dir} (skipped {skipped})")
    return exported, skipped
