# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from qualivault.validation import OllamaValidator


def validate_transcripts_to_individual_reports(
    *,
    transcripts_dir: Path,
    reports_dir: Path,
    model: str,
    ollama_url: str,
    language: str,
    validation_params: Dict[str, Any],
    resume: bool = True,
    max_files: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Validate CSV transcripts and write one JSON report per interview.

    Resume behavior:
    - If `resume=True`, existing `*_validation.json` files are loaded and skipped.
    - If `resume=False`, all files are revalidated and overwritten.

    Returns a list of reports (loaded + newly generated).
    """

    transcripts_dir = Path(transcripts_dir)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(transcripts_dir.glob("*.csv"))
    if max_files is not None:
        csv_files = csv_files[:max_files]

    results: List[Dict[str, Any]] = []

    validator = OllamaValidator(
        model=model,
        ollama_url=ollama_url,
        timeout=validation_params.get("timeout", 120),
    )

    for csv_file in csv_files:
        interview_name = csv_file.stem
        report_file = reports_dir / f"{interview_name}_validation.json"

        if resume and report_file.exists():
            try:
                with open(report_file, "r", encoding="utf-8") as f:
                    results.append(json.load(f))
                continue
            except Exception:
                # If existing report is corrupt, re-generate it.
                pass

        report = validator.validate_transcript(
            csv_file,
            sample_rate=validation_params.get("sample_rate", 1.0),
            language=language,
            validation_params=validation_params,
        )

        if report:
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            results.append(report)

    return results


def load_validation_reports(reports_dir: Path) -> List[Dict[str, Any]]:
    reports_dir = Path(reports_dir)
    report_files = sorted(reports_dir.glob("*_validation.json"))

    results: List[Dict[str, Any]] = []
    for report_file in report_files:
        try:
            with open(report_file, "r", encoding="utf-8") as f:
                results.append(json.load(f))
        except Exception:
            continue

    return results


def aggregate_validation_reports(
    *,
    reports_dir: Path,
    all_results: List[Dict[str, Any]],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Create a per-file DataFrame and totals dict, and write SUMMARY.csv."""

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    total_flagged = sum(r.get("flagged_count", len(r.get("flagged_segments", []))) for r in all_results)
    total_checked = sum(r.get("segments_checked", 0) for r in all_results)

    df = pd.DataFrame(
        [
            {
                "File": Path(r.get("csv_file", "unknown")).stem,
                "Checked": r.get("segments_checked", 0),
                "Flagged": r.get("flagged_count", len(r.get("flagged_segments", []))),
                "Rate": (
                    f"{(r.get('flagged_count', len(r.get('flagged_segments', []))) / max(r.get('segments_checked', 1), 1) * 100):.1f}%"
                    if r.get("segments_checked", 0) > 0
                    else "N/A"
                ),
            }
            for r in all_results
        ]
    )

    summary_file = reports_dir / "SUMMARY.csv"
    df.to_csv(summary_file, index=False)

    totals = {
        "files_validated": len(all_results),
        "total_segments_checked": total_checked,
        "total_segments_flagged": total_flagged,
        "flag_rate": (total_flagged / total_checked) if total_checked > 0 else 0.0,
        "summary_csv": str(summary_file),
    }

    return df, totals


def write_master_validation_summary(
    *,
    reports_dir: Path,
    all_results: List[Dict[str, Any]],
    model: str,
    sample_rate: float,
    language: str,
) -> Path:
    """Write VALIDATION_SUMMARY.json that references individual report files."""

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    total_flagged = sum(r.get("flagged_count", len(r.get("flagged_segments", []))) for r in all_results)
    total_checked = sum(r.get("segments_checked", 0) for r in all_results)

    master_summary = {
        "validation_date": datetime.now().isoformat(),
        "model": model,
        "sample_rate": sample_rate,
        "language": language,
        "summary": {
            "total_files_validated": len(all_results),
            "total_segments_checked": total_checked,
            "total_segments_flagged": total_flagged,
            "flag_rate": f"{(total_flagged / total_checked * 100):.1f}%" if total_checked > 0 else "0%",
        },
        "per_file_reports": [
            {
                "file": Path(r.get("csv_file", "unknown")).name,
                "segments_checked": r.get("segments_checked", 0),
                "segments_flagged": r.get("flagged_count", len(r.get("flagged_segments", []))),
                "report_file": f"{Path(r.get('csv_file', 'unknown')).stem}_validation.json",
            }
            for r in all_results
        ],
    }

    summary_file = reports_dir / "VALIDATION_SUMMARY.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2, ensure_ascii=False)

    return summary_file
